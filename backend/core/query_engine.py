import gc
import logging
import time

from core.config import AppConfig
from core.llm_engine import LLMEngine
from core.retriever import RetrieverEngine

log = logging.getLogger(__name__)


class QueryEngine:
    """
    Sorgu orkestrasyonu: retriever + LLM.

    Eski main.py'nin sınıflaştırılmış hali. Belirli bir koleksiyon ve
    doküman üzerinde çalışır, sonucu metin olarak döndürür.

    Donanım orkestrasyonu (8GB VRAM kısıtı):
        1) Retriever yükle (Jina CPU + Reranker CPU) — VRAM kullanmaz
        2) Bağlamı çek → retriever unload
        3) LLM (GPU) yükle → cevap üret → LLM unload

    Retriever CPU'da olduğu için aslında LLM ile aynı anda durabilirdi,
    ama bellek temizliği ve tutarlılık adına eski akış korunuyor.

    Bellek güvenliği: Tüm engine'ler 'with' bloklarında kullanılır.
    Exception, GeneratorExit veya normal çıkışta otomatik unload garantili.
    """

    def __init__(
        self,
        collection_name: str = "default",
        persist_dir: str = str(AppConfig.DATABASE_DIR),
    ):
        self.collection_name = collection_name
        self.persist_dir = persist_dir

    def run(self, question: str, file_names: list[str] | None = None) -> str:
        """
        Sorguyu çalıştırır, cevabı string olarak döndürür.

        file_names: Hangi dokümanlarda arama yapılacak. RetrieverEngine'a
                    metadata filtresi olarak iletilir.
                    Boş/None ise tüm aktif koleksiyon kapsamında aranır.

        Bellek yönetimi: 'with' blokları sayesinde RetrieverEngine ve
        LLMEngine her koşulda (exception olsa bile) unload edilir.
        """
        scope_label = (
            f"doküman(lar): {file_names}" if file_names else "tüm koleksiyon"
        )
        log.info(
            f"Sorgu başlatıldı — koleksiyon: '{self.collection_name}', "
            f"{scope_label}, soru: {question!r}"
        )
        start_time = time.time()

        # ── 1. Retriever ile bağlam çek ──
        log.info("Retriever yükleniyor...")
        t = time.time()
        with RetrieverEngine(
            collection_name=self.collection_name,
            persist_dir=self.persist_dir,
        ) as retriever:
            log.debug(f"Retriever hazır ({time.time() - t:.2f} sn).")

            log.info("Veritabanında arama + reranker çalışıyor...")
            t = time.time()
            context_text = retriever.get_relevant_context(
                query=question,
                top_n=AppConfig.RERANKER_TOP_N,
                threshold=0.0,
                file_names=file_names,
            )
            log.debug(f"Bağlam hazır ({time.time() - t:.2f} sn).")
        # Retriever burada otomatik unload — exception olsa bile.
        gc.collect()

        if not context_text:
            log.warning("Bağlam boş — LLM çağrılmadan dönülüyor.")
            return "Bu doküman için sorguya uygun bir bağlam bulunamadı."

        # ── 2. LLM ile cevap üret ──
        log.info("LLM yükleniyor ve cevap üretiyor...")
        t = time.time()
        with LLMEngine() as llm:
            log.debug(f"LLM hazır ({time.time() - t:.2f} sn).")
            gen_start = time.time()
            answer = llm.generate_answer(context=context_text, question=question)
            log.info(f"LLM cevabı üretildi ({time.time() - gen_start:.2f} sn).")
        # LLM burada otomatik unload.
        gc.collect()

        log.info(f"Sorgu tamamlandı ({time.time() - start_time:.2f} sn).")
        return answer

    def run_stream(self, question: str, file_names: list[str] | None = None):
        """
        Sorguyu çalıştırır, cevabı token token yield eder.

        Akış run() ile aynı — retriever çek, LLM çağır — ama LLM
        aşamasında stream döner. 'with' blokları sayesinde frontend
        bağlantıyı koparırsa (GeneratorExit) veya exception çıkarsa bile
        her iki engine unload edilir.

        file_names boş/None: tüm aktif koleksiyon kapsamında arar.
        """
        scope_label = (
            f"doküman(lar): {file_names}" if file_names else "tüm koleksiyon"
        )
        log.info(
            f"Streaming sorgu — koleksiyon: '{self.collection_name}', "
            f"{scope_label}, soru: {question!r}"
        )
        start_time = time.time()

        # ── 1. Retriever ile bağlam çek ──
        log.info("Retriever yükleniyor...")
        with RetrieverEngine(
            collection_name=self.collection_name,
            persist_dir=self.persist_dir,
        ) as retriever:
            log.info("Veritabanında arama + reranker çalışıyor...")
            context_text = retriever.get_relevant_context(
                query=question,
                top_n=AppConfig.RERANKER_TOP_N,
                threshold=0.0,
                file_names=file_names,
            )
        gc.collect()

        if not context_text:
            log.warning("Bağlam boş — LLM çağrılmadan dönülüyor.")
            yield "Bu doküman için sorguya uygun bir bağlam bulunamadı."
            return

        # ── 2. LLM stream ──
        log.info("LLM yükleniyor ve cevap akıtılıyor...")
        gen_start = time.time()
        with LLMEngine() as llm:
            try:
                for chunk in llm.generate_answer_stream(
                    context=context_text,
                    question=question,
                ):
                    yield chunk
            finally:
                # with bloku LLM unload'u zaten garanti ediyor; bu finally
                # sadece süre log'u için. GeneratorExit'te de çalışır.
                log.info(f"LLM cevabı bitti ({time.time() - gen_start:.2f} sn).")
        gc.collect()
        log.info(f"Sorgu tamamlandı ({time.time() - start_time:.2f} sn).")
