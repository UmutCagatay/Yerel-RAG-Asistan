import gc
import json
import logging
import os
import time
from typing import List, Optional

import chromadb
import numpy as np
import onnxruntime as ort
from core.config import AppConfig
from core.file_utils import atomic_write_json
from llama_index.core import Settings, StorageContext, VectorStoreIndex
from llama_index.core.embeddings import BaseEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer

log = logging.getLogger(__name__)

# Embedding inference sırasında VRAM kullanımını sınırlayan iç batch boyutu.
# Tek seferde 32 metin tokenize edilip ORT'a verilir; binlerce chunk gelse de
# VRAM kullanımı sabit kalır. 8GB VRAM kartında 32 güvenli; daha büyük batch
# hızlandırır ama VRAM taşma riskini artırır.
_EMBED_BATCH_SIZE: int = 32


class JinaEmbeddings(BaseEmbedding):
    """
    Jina V5 Nano ONNX embedding sarmalayıcısı (numpy I/O).

    device:
        "cuda" → CUDAExecutionProvider, VRAM'e yüklenir.
        "cpu"  → CPUExecutionProvider, RAM'de kalır.

    PyTorch CUDA build bağımlılığı yok; ORT host↔device kopyayı yönetir.
    """

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, device: str = "cuda"):
        # embed_batch_size=10: LlamaIndex'in default'u. 4 dosyada peak ~3.4 GB
        # ölçüldü, 6 GB cap'le bol pay var. Daha yüksek batch ingestion'ı
        # hızlandırır ama attention bellek karesinde büyür; 10 dengeli nokta.
        super().__init__(model_name="jina-v5-nano-onnx", embed_batch_size=10)
        if device not in ("cuda", "cpu"):
            raise ValueError(f"device 'cuda' veya 'cpu' olmalı, aldı: {device}")

        self._device = device
        self._model_dir = str(AppConfig.EMBED_MODEL_DIR)
        self._tokenizer: Optional[AutoTokenizer] = None
        self._model: Optional[ORTModelForFeatureExtraction] = None
        self._load()

    # ── Yükleme / Tahliye ────────────────────────────────────────────────
    def _load(self) -> None:
        load_start = time.time()
        log.info(f"Jina V5 Nano ONNX ({self._device.upper()}) yükleniyor...")

        self._tokenizer = AutoTokenizer.from_pretrained(
            self._model_dir, trust_remote_code=True, local_files_only=True
        )

        if self._device == "cuda":
            provider = "CUDAExecutionProvider"
            provider_options = [
                {
                    "device_id": 0,
                    # ÖNEMLİ: kSameAsRequested birikim yapar — ORT GitHub issue
                    # #14474 ve #13500'de doğrulanmış, eski tensor'lar geri
                    # alınmıyor. kNextPowerOfTwo pool'dan reuse ediyor: aynı
                    # boyutta yeni alloc istense bile var olan blok yeniden
                    # kullanılıyor. Bizim için tek dosya/4 dosya farkını
                    # ortadan kaldıracak ana ayar.
                    "arena_extend_strategy": "kNextPowerOfTwo",
                    # Hard cap: embedder VRAM'in en fazla 6 GB'ını alabilir.
                    # 4 dosyada peak ölçümü 3.4 GB olduğundan 6 GB cap bol
                    # rahat pay tanır ama beklenmedik leak/sıçrama olursa
                    # paylaşımlı belleğe taşmadan burada tepelenir.
                    "gpu_mem_limit": 6 * 1024 * 1024 * 1024,
                    "cudnn_conv_algo_search": "DEFAULT",
                    "do_copy_in_default_stream": True,
                }
            ]
        else:
            provider = "CPUExecutionProvider"
            provider_options = None

        # SessionOptions — peak memory için kritik.
        # enable_mem_pattern=True (default) intermediate tensor'ları reuse
        # için tutar; ONNX kaynak kodundaki kendi yorumlarına göre:
        # "can lead to memory being held for longer than needed and can
        # impact peak memory consumption". Bizim senaryomuzda peak'i
        # düşürmek için kapatıyoruz. Hız etkisi minimal.
        sess_options = ort.SessionOptions()
        if self._device == "cuda":
            sess_options.enable_mem_pattern = False

        self._model = ORTModelForFeatureExtraction.from_pretrained(
            self._model_dir,
            subfolder="onnx",
            file_name="model.onnx",
            provider=provider,
            provider_options=provider_options,
            session_options=sess_options,
            trust_remote_code=True,
            local_files_only=True,
        )

        # Sessiz CPU fallback kontrolü — ORT yanlış kuruluysa CUDA ister,
        # uyarı bile vermeden CPU'ya düşebilir.
        active = self._model.model.get_providers()
        if self._device == "cuda" and "CUDAExecutionProvider" not in active:
            raise RuntimeError(
                f"CUDA provider yüklenemedi. Aktif provider'lar: {active}\n"
                "Kontrol listesi:\n"
                "  1) pip uninstall onnxruntime onnxruntime-gpu -y\n"
                "  2) pip install onnxruntime-gpu\n"
                "  3) nvidia-smi → sürücü 525+ olmalı (CUDA 12.x için)\n"
                "  4) Windows'ta cuDNN bin klasörü PATH'e eklenmiş olmalı"
            )

        log.info(
            f"Jina V5 Nano ONNX yüklendi ({time.time() - load_start:.2f} sn). "
            f"Aktif provider: {active[0]}"
        )

    def unload(self) -> None:
        """ORT InferenceSession'ı serbest bırakır; CUDA buffer'lar destructor'da temizlenir."""
        log.info(f"Jina embedding ({self._device.upper()}) tahliye ediliyor...")
        self._model = None
        self._tokenizer = None
        gc.collect()
        # Not: torch.cuda.empty_cache() ÇAĞIRILMIYOR.
        # ORT'un CUDA bellek arenası torch'tan ayrıdır; ORT session destructor'u
        # kendi buffer'larını serbest bırakır. del + gc yeterli.
        log.info("Jina embedding belleği temizlendi.")

    # ── Encode (numpy I/O) ───────────────────────────────────────────────
    def _encode(self, texts: List[str]) -> List[List[float]]:
        """
        Verilen metinleri embed eder. İç batch ile VRAM güvenli:
        gelen liste ne kadar uzun olursa olsun, tek seferde en fazla
        _EMBED_BATCH_SIZE kadar metin tokenize edilip inference yapılır.

        Önemli: LlamaIndex VectorStoreIndex tüm chunk'ları (binlerce olabilir)
        tek çağrıda gönderebilir; eski tek-batch sürümü bu durumda VRAM'i
        7-8 GB doldurup paylaşımlı belleğe taşma yapıyordu.
        """
        if self._model is None or self._tokenizer is None:
            raise RuntimeError(
                "Model tahliye edilmiş; tekrar JinaEmbeddings() oluştur."
            )

        all_embeddings: List[List[float]] = []

        for start in range(0, len(texts), _EMBED_BATCH_SIZE):
            batch = texts[start : start + _EMBED_BATCH_SIZE]

            # Tokenizer'dan numpy iste — torch tensor üretimini atla.
            # max_length=1536: VLM blokları (VLM_MAX_TOKENS=1536) tek chunk
            # olarak gelebildiği için cap'i ona göre tutuyoruz, bilgi kaybı
            # olmasın. Attention seq^2 ile büyüdüğü için embed_batch_size=5
            # ile dengeliyoruz (init'te ayarlı). VLM olmayan kısa chunk'lar
            # padding=True ile zaten asıl uzunluklarında kalır.
            enc = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=1536,
                return_tensors="np",
            )

            # ORT InferenceSession'a doğrudan numpy ver.
            # CUDA provider aktifse host→device kopya ORT içinde yapılır.
            ort_inputs = {
                "input_ids": enc["input_ids"].astype(np.int64),
                "attention_mask": enc["attention_mask"].astype(np.int64),
            }
            outputs = self._model.model.run(None, ort_inputs)
            last_hidden = outputs[0]  # (batch, seq, hidden), numpy

            # Last-token pooling (resmi Jina kullanımı)
            attention_mask = enc["attention_mask"]
            seq_lengths = attention_mask.sum(axis=1) - 1
            batch_idx = np.arange(last_hidden.shape[0])
            embeddings = last_hidden[batch_idx, seq_lengths]

            # L2 normalize
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-8)
            embeddings = embeddings / norms

            all_embeddings.extend(embeddings.tolist())

        return all_embeddings

    # ── BaseEmbedding kontratı ───────────────────────────────────────────
    def _get_text_embedding(self, text: str) -> List[float]:
        return self._encode([f"Document: {text}"])[0]

    def _get_query_embedding(self, query: str) -> List[float]:
        return self._encode([f"Query: {query}"])[0]

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return self._encode([f"Document: {t}" for t in texts])


class VectorStoreEngine:
    def __init__(
        self,
        persist_dir: str = str(AppConfig.DATABASE_DIR),
        collection_name: str = "default",
        embed_device: str = "cuda",
    ):
        self.persist_dir = persist_dir
        self.collection_name = collection_name

        log.info(f"VectorStoreEngine başlatılıyor ({embed_device.upper()})...")

        self.embed_model = JinaEmbeddings(device=embed_device)
        Settings.embed_model = self.embed_model
        Settings.llm = None

        self.db_client = chromadb.PersistentClient(path=self.persist_dir)
        self.chroma_collection = self.db_client.get_or_create_collection(
            self.collection_name
        )
        self.vector_store = ChromaVectorStore(chroma_collection=self.chroma_collection)
        self.storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store
        )

    def _save_sections(self, parent_nodes: list, file_name: str) -> None:
        """
        Section parent node'larını JSON dosyasına kaydeder.

        Neden JSON, neden ChromaDB değil?
            Section parent'lar hiçbir zaman similarity araması ile bulunmaz.
            Erişim her zaman section_id ile yapılır: sections_map[section_id].
            Bu ID bazlı erişim için vektör veritabanı gerekmiyor; saf bir
            dict (JSON) aynı işi O(1) ile yapar, embedding maliyeti sıfır,
            HNSW indeksi oluşmaz.

        Dosya yapısı:
            { "<section_id>": {"text": "...", "metadata": {...}}, ... }

        Yazma atomik (.tmp + os.replace) — çökme sırasında bozuk dosya kalmaz.
        """

        sections_file = os.path.join(self.persist_dir, "sections.json")

        # Mevcut dosyayı yükle (başka PDF'lerden gelen section'lar korunur)
        existing: dict = {}
        if os.path.exists(sections_file):
            try:
                with open(sections_file, "r", encoding="utf-8") as f:
                    existing = json.load(f)
            except Exception as e:
                log.error(
                    f"sections.json okunamadı, mevcut kayıtlar kaybolabilir: {e}",
                    exc_info=True,
                )
                return

        # Yeni section'ları ekle
        for node in parent_nodes:
            existing[node.node_id] = {
                "text": node.text,
                "metadata": node.metadata,
            }

        try:
            atomic_write_json(sections_file, existing)
            log.debug(f"sections.json güncellendi: {len(parent_nodes)} yeni section")
        except Exception as e:
            log.error(f"sections.json yazılamadı: {e}", exc_info=True)

    def add_nodes(self, nodes, file_name: str) -> tuple[int, int]:
        """
        Dönüş: (child_count, section_count)
        """
        if not nodes:
            log.warning("Eklenecek node bulunamadı.")
            return 0, 0

        child_nodes = [n for n in nodes if n.metadata.get("node_type") != "section"]
        parent_nodes = [n for n in nodes if n.metadata.get("node_type") == "section"]

        log.info(f"{len(child_nodes)} child node embedding'leniyor...")
        VectorStoreIndex(nodes=child_nodes, storage_context=self.storage_context)

        # Parent'ları JSON'a kaydet — embedding hesabı yok, salt metin deposu
        if parent_nodes:
            log.debug(f"{len(parent_nodes)} section parent JSON'a kaydediliyor...")
            self._save_sections(parent_nodes, file_name)

        log.info(
            f"Kayıt tamamlandı: {len(child_nodes)} child → '{self.collection_name}', "
            f"{len(parent_nodes)} section → sections.json"
        )
        return len(child_nodes), len(parent_nodes)

    def unload(self):
        """
        Jina embedding modelini VRAM'den serbest bırakır.

        Settings.embed_model global state olduğu için onu da temizliyoruz,
        yoksa lokal referanslar silinse bile model bellekte kalır.
        """
        log.info("Embedding modeli bellekten tahliye ediliyor...")
        # 1. Önce Jina'nın kendi unload'unu çağır — ORT session ve tokenizer
        # açıkça None'lanıyor, destructor garantili çalışıyor.
        try:
            jina = Settings.embed_model
            if jina is not None and hasattr(jina, "unload"):
                jina.unload()
        except Exception as e:
            log.warning(f"Jina unload sırasında: {e}", exc_info=True)

        # 2. LlamaIndex global state'i temizle
        try:
            Settings.embed_model = None
        except Exception:
            pass

        # 3. Tüm ağır referansları sil — embed_model dahil olmak üzere tüm kardeşler
        if hasattr(self, "embed_model"):
            del self.embed_model
        if hasattr(self, "vector_store"):
            del self.vector_store
        if hasattr(self, "storage_context"):
            del self.storage_context
        if hasattr(self, "chroma_collection"):
            del self.chroma_collection
        if hasattr(self, "db_client"):
            del self.db_client

        # Not: torch.cuda.empty_cache() çağrılmıyor — ChromaDB ve Jina ONNX
        # PyTorch GPU kullanmıyor. Jina'nın kendi unload'u zaten ORT
        # session'ı temizliyor.
        gc.collect()

        log.info("Embedding belleği temizlendi.")

    # Context manager protokolü — 'with VectorStoreEngine() as ve:' kullanımı için.
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.unload()
        return False
