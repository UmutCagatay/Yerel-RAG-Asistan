import json
import logging
import os
import time

from core.config import AppConfig
from core.vector_store import JinaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document as LCDocument
from llama_cpp.llama_cpp import LLAMA_POOLING_TYPE_RANK
from llama_cpp.llama_embedding import LlamaEmbedding

log = logging.getLogger(__name__)


class RetrieverEngine:
    def __init__(
        self,
        collection_name: str = "default",
        persist_dir: str = str(AppConfig.DATABASE_DIR),
    ):
        self.persist_dir = persist_dir
        self.collection_name = collection_name

        log.info("Retriever modelleri belleğe alınıyor...")

        jina = JinaEmbeddings(device="cpu")

        class _LCAdapter:
            def embed_documents(self, texts):
                return jina._get_text_embeddings(texts)

            def embed_query(self, text):
                return jina._get_query_embedding(text)

        reranker_start = time.time()
        self.reranker = LlamaEmbedding(
            model_path=str(AppConfig.RERANKER_MODEL_PATH),
            pooling_type=LLAMA_POOLING_TYPE_RANK,
            n_gpu_layers=0,
            n_ctx=0,
            n_batch=4096,
            n_ubatch=4096,
            verbose=False,
        )
        log.info(f"BGE Reranker GGUF yüklendi ({time.time() - reranker_start:.2f} sn).")

        self.vectorstore = Chroma(
            persist_directory=self.persist_dir,
            embedding_function=_LCAdapter(),
            collection_name=self.collection_name,
        )

        sections_file = os.path.join(self.persist_dir, AppConfig.SECTIONS_FILENAME)
        self.sections_map: dict = {}
        if os.path.exists(sections_file):
            try:
                with open(sections_file, "r", encoding="utf-8") as _f:
                    self.sections_map = json.load(_f)
                log.info(f"{len(self.sections_map)} section parent belleğe yüklendi.")
            except Exception as e:
                log.error(f"sections.json okunamadı: {e}", exc_info=True)
        else:
            # Eski mesaj 'ingest.py çalıştırın' diyordu, artık ingest.py yok.
            # Kullanıcıyı doğru yere yönlendirelim.
            log.warning(
                "sections.json bulunamadı. Bu koleksiyona henüz "
                "doküman eklenmemiş olabilir."
            )

        # filter'lı base_retriever kalktı → Python tarafında filtreleme güvenilir
        self.base_retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": AppConfig.RETRIEVER_K}
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Komşu Node Genişletici
    # ──────────────────────────────────────────────────────────────────────────

    def _expand_with_neighbors(self, docs: list) -> list:
        """
        Retriever'ın döndürdüğü node'ları belge sırasına göre genişletir.

        Kural 1 — VLM node bulunduysa:
            Sol komşu (index-1) ve sağ komşu (index+1) da getirilir.
            Hedef: Başlık → VLM veya VLM → metin geçişlerinin korunması.
            Örnek: node_index=6 ("## OSI Referans Modeli") + node_index=7
                   (VLM şema) → ikisi birlikte LLM'e gider.

        Kural 2 — Metin node bulunduysa:
            Sadece sağ komşu (index+1) çekilir.
            Hedef: Metnin hemen ardından gelen VLM node'u da bağlama katmak.

        Sonuç liste node_index'e göre sıralanır ve tekrarlar temizlenir.
        Böylece LLM, belgedeki orijinal sırayı görür.
        """
        if not docs:
            return docs

        # Mevcut indeksleri kayıt altına al
        existing_indices: dict[int, LCDocument] = {}
        for doc in docs:
            idx = doc.metadata.get("node_index")
            if idx is not None:
                existing_indices[int(idx)] = doc

        # Hangi komşuları getireceğimizi belirle
        neighbor_indices: set[int] = set()
        for doc in docs:
            idx = doc.metadata.get("node_index")
            if idx is None:
                continue
            idx = int(idx)
            ntype = doc.metadata.get("node_type", "text")

            if ntype == "vlm":
                # Her iki komşu
                if idx - 1 >= 0:
                    neighbor_indices.add(idx - 1)
                neighbor_indices.add(idx + 1)
            else:
                # Sadece sağ komşu: yanındaki VLM olabilir
                neighbor_indices.add(idx + 1)

        # Zaten elimizde olanları çıkar
        to_fetch = [i for i in neighbor_indices if i not in existing_indices]

        if not to_fetch:
            return sorted(docs, key=lambda d: int(d.metadata.get("node_index", 9999)))

        # ChromaDB'den komşuları çek — sadece aynı dosyadaki node'lardan
        # Filtre yoksa farklı dosyalardan node sızar (bkz. bug raporu).
        file_names = {
            doc.metadata.get("file_name")
            for doc in docs
            if doc.metadata.get("file_name")
        }

        where_clause: dict = {"node_index": {"$in": to_fetch}}
        if file_names:
            where_clause = {
                "$and": [
                    {"node_index": {"$in": to_fetch}},
                    {"file_name": {"$in": list(file_names)}},
                ]
            }

        extra_docs: list[LCDocument] = []
        try:
            results = self.vectorstore._collection.get(
                where=where_clause,
                include=["documents", "metadatas"],
            )
            if results and results.get("documents"):
                for text, meta in zip(results["documents"], results["metadatas"]):
                    if text and meta is not None:
                        extra_docs.append(LCDocument(page_content=text, metadata=meta))
        except Exception as e:
            log.warning(f"Komşu node'lar alınamadı: {e}", exc_info=True)

        # Birleştir → tekilleştir → belge sırasına göre sırala
        all_docs = docs + extra_docs
        seen: set[int] = set()
        final: list[LCDocument] = []

        for doc in sorted(
            all_docs,
            key=lambda d: int(d.metadata.get("node_index", 9999)),
        ):
            idx = int(doc.metadata.get("node_index", id(doc)))
            if idx not in seen:
                seen.add(idx)
                final.append(doc)

        return final

    def _expand_with_section_context(self, docs: list) -> list:
        """
        Child node'ları section parent'larına genişletir.

        Aynı section'dan birden fazla child gelirse parent yalnızca BİR kez
        eklenir; sonraki child'lar atlanır çünkü parent metni o child'ları
        zaten kapsar. Yalnızca section_id yoksa veya parent JSON'da kayıpsa
        komşu fallback'e düşülür.
        """
        seen_section_ids: set[str] = set()
        result_docs: list[LCDocument] = []

        for doc in docs:
            section_id = doc.metadata.get("section_id")

            # Aynı section daha önce eklendiyse atla — parent o child'ı içeriyor
            if section_id and section_id in seen_section_ids:
                continue

            if section_id:
                seen_section_ids.add(section_id)
                section_data = self.sections_map.get(section_id)
                if section_data:
                    result_docs.append(
                        LCDocument(
                            page_content=section_data["text"],
                            metadata=section_data.get("metadata", {}),
                        )
                    )
                    continue
                log.warning(
                    f"Section bulunamadı (id={section_id[:8]}...): "
                    "sections.json'da yok, komşu fallback'ine düşülüyor."
                )

            # section_id yok ya da parent kayıp — fallback
            result_docs.extend(self._expand_with_neighbors([doc]))

        # node_index'e göre sırala, section_id'ye göre tekilleştir
        seen: set = set()
        final: list[LCDocument] = []
        for doc in sorted(
            result_docs, key=lambda d: int(d.metadata.get("node_index", 9999))
        ):
            key = doc.metadata.get("section_id") or id(doc)
            if key not in seen:
                seen.add(key)
                final.append(doc)
        return final

    # ──────────────────────────────────────────────────────────────────────────
    # Ana Bağlam Getirici
    # ──────────────────────────────────────────────────────────────────────────

    def get_relevant_context(
        self,
        query: str,
        top_n: int = 3,
        threshold: float = 0.0,
        file_name: str | None = None,
    ):
        """
        Sorguya en uygun bağlamı döndürür.

        Akış:
            1. k=10 ile tez_koleksiyonu'ndan child node'lar çek (section'lar ayrı koleksiyonda)
            2. Reranker ile yeniden sırala, top_n al
            3. Section parent'larıyla genişlet (_expand_with_section_context)
            4. Debug çıktısı yaz
            5. Birleşik parent metinleri döndür
        """
        # ── 1. Ham getirme ─────────────────────────────────────
        # Doküman filtresi varsa retriever'ı tek seferlik filter'lı kur.
        # Constructor'daki base_retriever sade kalır, oturum içinde
        # farklı dokümanlara sorgu atılabilir.
        search_start = time.time()
        if file_name:
            filtered_retriever = self.vectorstore.as_retriever(
                search_kwargs={
                    "k": AppConfig.RETRIEVER_K,
                    "filter": {"file_name": file_name},
                }
            )
            raw_docs = filtered_retriever.invoke(query)
        else:
            raw_docs = self.base_retriever.invoke(query)
        search_ms = (time.time() - search_start) * 1000

        if not raw_docs:
            log.warning("Hiç sonuç bulunamadı.")
            return ""

        # ── 2. Reranker ───────────────────────────────────────────────────────────
        temp_time = time.time()
        documents = [doc.page_content for doc in raw_docs]
        scores = self.reranker.rank(query, documents)
        reranker_ms = (time.time() - temp_time) * 1000

        scored_docs = sorted(zip(scores, raw_docs), key=lambda x: x[0], reverse=True)
        best_docs = [doc for _, doc in scored_docs[:top_n]]

        if not best_docs:
            log.warning(f"Reranker tüm sonuçları threshold={threshold} altında eledi.")
            return ""

        # ── 3. Debug: Reranker sonuçları ──────────────────────────────────────────
        # Çok satırlı debug çıktısı tek bir DEBUG mesajı olarak yazılır.
        # Böylece dosyada blok atomik kalır, başka modüllerin logları araya girmez.
        # isEnabledFor kontrolü: DEBUG kapalıysa string birleştirme ve döngüye
        # hiç girilmez, sorgu performansı etkilenmez.
        if log.isEnabledFor(logging.DEBUG):
            debug_lines = [
                "═" * 70,
                "  RETRIEVER DEBUG",
                f"  Sorgu  : {query[:60]}",
                f"  Ham    : {len(raw_docs)} node çekildi",
                f"  Search  : {search_ms:.0f}ms (vektör araması, top-10)",
                f"  Reranker: {reranker_ms:.0f}ms — top-{top_n} seçildi",
                "─" * 70,
            ]
            for i, (score, doc) in enumerate(scored_docs[:top_n]):
                ntype = doc.metadata.get("node_type", "?")
                nidx = doc.metadata.get("node_index", "?")
                ss = doc.metadata.get("section_start_index", "?")
                se = doc.metadata.get("section_end_index", "?")
                sid = doc.metadata.get("section_id") or "yok"
                fname = doc.metadata.get("file_name", "?")
                prev = doc.page_content[:90].replace("\n", " ")
                marker = "✓" if doc in best_docs else "✗"
                debug_lines.append(
                    f"  [{i + 1}]{marker} score={score:+.4f} | file={fname} | "
                    f"{ntype}[{nidx}] | sec=[{ss}-{se}] | section_id:{sid}"
                )
                debug_lines.append(f"       {prev}...")
            # Başa ekstra \n: log formatter prefix'i (timestamp + seviye)
            # tek satırda bırakır, blok altta hizalı başlar.
            log.debug("\n" + "\n".join(debug_lines))

        # ── 4. Section genişletme ─────────────────────────────────────────────────
        expanded_docs = self._expand_with_section_context(best_docs)

        # Debug: döndürülen section'lar — yine tek bir DEBUG mesajı
        if log.isEnabledFor(logging.DEBUG):
            debug_lines = [
                "─" * 70,
                f"  Section genişletme → {len(expanded_docs)} doc LLM'e gönderilecek:",
            ]
            for doc in expanded_docs:
                ntype = doc.metadata.get("node_type", "?")
                nidx = doc.metadata.get("node_index", "?")
                ss = doc.metadata.get("section_start_index", "?")
                se = doc.metadata.get("section_end_index", "?")
                fname = doc.metadata.get("file_name", "?")
                clen = len(doc.page_content)
                # Overlap prefix varsa atla — asıl içerikten önizleme göster.
                # "[Önceki Bölüm Bağlamı: ...]" prefixleri debug'ı yanıltır.
                content_for_preview = doc.page_content
                if content_for_preview.startswith("[Önceki Bölüm Bağlamı:"):
                    skip = content_for_preview.find("\n\n")
                    if skip > 0:
                        content_for_preview = content_for_preview[skip + 2 :]

                prev = content_for_preview[:60].replace("\n", " ")
                debug_lines.append(
                    f"    file={fname} | {ntype}[{nidx}] | sec=[{ss}-{se}] | "
                    f"{clen} karakter | {prev}..."
                )
            debug_lines.append("═" * 70)
            # Başa ekstra \n: log formatter prefix'i (timestamp + seviye)
            # tek satırda bırakır, blok altta hizalı başlar.
            log.debug("\n" + "\n".join(debug_lines))

        # ── 5. Bağlam oluştur ─────────────────────────────────────────────────────
        parts = []
        for doc in expanded_docs:
            content = doc.page_content.strip()
            if content:
                parts.append(content)

        return "\n\n---\n\n".join(parts)

    def unload(self):
        """
        Reranker + vectorstore referanslarını temizler.

        Reranker LlamaEmbedding (llama-cpp-python, C++); vectorstore
        langchain_chroma + ONNX Jina. Hiçbiri PyTorch GPU kullanmıyor,
        torch.cuda.empty_cache() yararsız.
        """
        import gc

        log.info("Reranker bellekten tahliye ediliyor...")
        if hasattr(self, "base_retriever"):
            del self.base_retriever
        if hasattr(self, "vectorstore"):
            del self.vectorstore
        if hasattr(self, "reranker"):
            del self.reranker
        gc.collect()
        log.info("Reranker belleği temizlendi.")

    # Context manager protokolü — 'with RetrieverEngine(...) as ret:' kullanımı için.
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.unload()
        return False
