import gc
import logging
import os
import time
from datetime import datetime

from core.config import AppConfig
from core.document_parser import DocumentParser
from core.vector_store import VectorStoreEngine
from core.vlm_engine import VLMEngine

log = logging.getLogger(__name__)


class IngestionEngine:
    """
    Çoklu PDF'i sırayla işleyen ingestion orkestrasyonu.

    Donanım orkestrasyonu (8GB VRAM kısıtı):
        Faz 1 — VLM (GPU) yükle → tüm PDF'leri parse et → VLM unload
        Faz 2 — Jina (GPU) yükle → tüm chunk'ları ChromaDB'ye yaz → unload

    Modeller asla aynı anda VRAM'de durmaz. Tek dosya senaryosu da bu
    akışın özel hali — liste uzunluğu 1.

    Hata politikası (D1):
        Bir dosya hata verirse atlanır, diğerlerine devam edilir.
        Sonunda başarılı/başarısız özet raporu döndürülür.

    Bellek güvenliği: VLM optional olduğu için manuel try/finally;
    VectorStoreEngine 'with' bloğuyla. İkisinde de exception olsa bile
    unload garantili.
    """

    def __init__(self, persist_dir: str = str(AppConfig.DATABASE_DIR)):
        self.persist_dir = persist_dir

    def run(
        self,
        file_paths: list[str],
        collection_name: str = "default",
    ) -> dict:
        """
        Dönüş:
            {
                "success": [{"file_name": "...", "chunk_count": N, "section_count": M}, ...],
                "failed":  [{"file_name": "...", "reason": "..."}, ...]
            }
        """
        if not file_paths:
            log.warning("İşlenecek dosya yok.")
            return {"success": [], "failed": [], "vlm_loaded": True}

        log.info(f"Ingestion başlatıldı: {len(file_paths)} dosya")
        start_time = time.time()

        # Her dosya için: file_path → parse sonucu (chunks) veya hata
        parsed: list[dict] = []  # {"file_path", "file_name", "chunks"}
        failed: list[dict] = []

        # ── Faz 1: VLM ile parse ─────────────────────────────────────────────
        # VLM optional: yüklenemezse parser görselsiz devam eder.
        # 'with' bloğu None için çalışmaz, bu yüzden manuel try/finally.
        log.info("Faz 1: VLM yükleniyor...")
        vlm_engine = None
        try:
            vlm_engine = VLMEngine()
        except Exception as e:
            # WARNING seviyesi — sistem çökmüyor, sadece görseller atlanacak.
            # Parser vlm_engine=None durumunu zaten handle ediyor.
            log.warning(
                f"VLM yüklenemedi, görselsiz devam ediliyor: {e}",
                exc_info=True,
            )

        # VLM yükleme bayrağını şimdi yakala. Faz 1 finally bloku vlm_engine'i
        # sileceği için sonraki return'larda erişemeyiz. Bu bayrak UI'ya
        # "görseller atlandı" uyarısı göstermek için lazım.
        vlm_loaded = vlm_engine is not None

        try:
            parser = DocumentParser()

            for file_path in file_paths:
                file_name = os.path.basename(file_path)
                log.info(f"Parse ediliyor: {file_name}")

                if not os.path.exists(file_path):
                    failed.append({"file_name": file_name, "reason": "Dosya bulunamadı"})
                    log.error(f"Dosya bulunamadı: {file_path}")
                    continue

                try:
                    # Per-file parse süresini ölç — tez için performans verisi olur
                    t = time.time()
                    chunks = parser.parse(file_path=file_path, vlm_engine=vlm_engine)
                    parse_duration = time.time() - t
                    parsed.append(
                        {"file_path": file_path, "file_name": file_name, "chunks": chunks}
                    )
                    log.debug(
                        f"{file_name} parse edildi: {len(chunks)} node, "
                        f"{parse_duration:.2f} sn"
                    )
                except Exception as e:
                    failed.append({"file_name": file_name, "reason": f"Parse hatası: {e}"})
                    log.error(f"{file_name} parse edilemedi: {e}", exc_info=True)
        finally:
            # Faz 1 cleanup — parse hatası, KeyboardInterrupt vs olsa bile VLM
            # mutlaka VRAM'den temizlenir.
            if vlm_engine is not None:
                log.info("Faz 1 bitti, VLM tahliye ediliyor.")
                vlm_engine.unload()
                del vlm_engine
            gc.collect()
            log.debug("VRAM boşaltıldı (Faz 1 sonu).")

        if not parsed:
            log.warning("Hiçbir dosya başarıyla parse edilemedi, Faz 2 atlandı.")
            return {"success": [], "failed": failed, "vlm_loaded": vlm_loaded}

        # ── Faz 2: Embedding + ChromaDB yazımı ───────────────────────────────
        log.info("Faz 2: Embedding (Jina) yükleniyor...")
        success: list[dict] = []

        with VectorStoreEngine(
            persist_dir=self.persist_dir,
            collection_name=collection_name,
        ) as vector_engine:
            for item in parsed:
                file_name = item["file_name"]
                chunks = item["chunks"]

                try:
                    # Her chunk'a collection_name metadata'sı ekle (silme için kritik)
                    for node in chunks:
                        node.metadata["collection_name"] = collection_name

                    # Per-file yazma süresini ölç
                    t = time.time()
                    child_count, section_count = vector_engine.add_nodes(
                        nodes=chunks, file_name=file_name
                    )
                    write_duration = time.time() - t
                    success.append(
                        {
                            "file_name": file_name,
                            "chunk_count": child_count,
                            "section_count": section_count,
                            "added_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        }
                    )
                    log.debug(
                        f"{file_name} yazıldı: {child_count} child, "
                        f"{section_count} section, {write_duration:.2f} sn"
                    )
                except Exception as e:
                    failed.append({"file_name": file_name, "reason": f"Yazma hatası: {e}"})
                    log.error(f"{file_name} ChromaDB'ye yazılamadı: {e}", exc_info=True)
        # Vector engine burada otomatik unload — exception olsa bile.
        gc.collect()
        log.info("Faz 2 bitti, embedding modeli tahliye ediliyor.")

        # ── Özet rapor ───────────────────────────────────────────────────────
        elapsed = time.time() - start_time
        log.info(
            f"Ingestion tamamlandı ({elapsed:.1f} sn): "
            f"{len(success)} başarılı, {len(failed)} başarısız."
        )
        # Başarısız dosyaları tek tek WARNING'e dök — kullanıcı hangi dosyanın
        # neden düştüğünü terminalde de görmeli, sadece dosyada değil
        for f in failed:
            log.warning(f"Başarısız: {f['file_name']} — {f['reason']}")

        return {"success": success, "failed": failed, "vlm_loaded": vlm_loaded}
