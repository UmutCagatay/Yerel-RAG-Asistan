from pathlib import Path


class AppConfig:
    # ── Dizinler ─────────────────────────────────────────────────────────
    # backend/core/config.py → iki yukarı = proje kökü
    BASE_DIR: Path = Path(__file__).resolve().parents[2]
    BACKEND_DIR: Path = BASE_DIR / "backend"
    MODELS_DIR: Path = BACKEND_DIR / "models"
    DATA_DIR: Path = BACKEND_DIR / "data"
    DATABASE_DIR: Path = DATA_DIR / "database"
    LOGS_DIR: Path = DATA_DIR / "logs"
    TEMP_IMAGES_DIR: Path = DATA_DIR / "temp_images"
    CHATS_DIR: Path = DATA_DIR / "chats"

    # ── Sabit dosya isimleri ─────────────────────────────────────────────
    CATALOG_FILENAME: str = "documents.json"
    SECTIONS_FILENAME: str = "sections.json"
    LOG_FILENAME: str = "app.log"

    # ── Model dosya isimleri ─────────────────────────────────────────────
    LLM_MODEL_NAME: str = "Turkish-Gemma-9b-T1-Q4_K_M.gguf"
    VLM_MODEL_NAME: str = "ZwZ-4B-Q4_K_M.gguf"
    VLM_MMPROJ_NAME: str = "mmproj-ZwZ-4B-F16.gguf"
    RERANKER_MODEL_NAME: str = "bge-reranker-v2-m3-Q8_0.gguf"
    EMBED_MODEL_DIR_NAME: str = "jina-v5-nano"

    # ── Model tam yolları (sabit konum, override gereksiz) ───────────────
    LLM_MODEL_PATH: Path = MODELS_DIR / LLM_MODEL_NAME
    VLM_MODEL_PATH: Path = MODELS_DIR / VLM_MODEL_NAME
    VLM_MMPROJ_PATH: Path = MODELS_DIR / VLM_MMPROJ_NAME
    RERANKER_MODEL_PATH: Path = MODELS_DIR / RERANKER_MODEL_NAME
    EMBED_MODEL_DIR: Path = MODELS_DIR / EMBED_MODEL_DIR_NAME

    # ── Ingestion sınırları ──────────────────────────────────────────────
    # Pratikte tezde kullanılacak akademik PDF'ler 1-5 MB aralığında.
    # 15 MB cap, anormal büyük dosyaları engeller ama makul senaryoları
    # kısıtlamaz. VLM parse sırasında peak RAM dosya boyutuyla orantılı
    # arttığı için bu cap aynı zamanda RAM güvenliği sağlıyor.
    MAX_FILE_SIZE_MB: int = 15

    # ── Chunk parametreleri ──────────────────────────────────────────────
    CHUNK_SIZE: int = 1100
    CHUNK_OVERLAP: int = 200

    # ── Section/parent chunking ──────────────────────────────────────────
    # Bu uzunluğun altındaki saf metin segmentleri komşularıyla birleştirilir,
    # tek başına node olmaz.
    MIN_SEGMENT_LEN: int = 150
    # Çok kısa section'lar sonrakiyle birleştirilir.
    MIN_SECTION_CHARS: int = 600
    # ~1000 token; 3 section → ~3000 token bağlam, LLM için güvenli üst sınır.
    MAX_SECTION_CHARS: int = 5000
    # Boyut-tabanlı kesmede taşınan örtüşme.
    SECTION_OVERLAP_CHARS: int = 200

    # ── Retriever ────────────────────────────────────────────────────────
    RETRIEVER_K: int = 10
    RERANKER_TOP_N: int = 3

    # ── LLM ──────────────────────────────────────────────────────────────
    LLM_N_CTX: int = 8192
    LLM_MAX_TOKENS: int = 2048
    LLM_TEMPERATURE: float = 0.1

    # ── VLM ──────────────────────────────────────────────────────────────
    VLM_N_CTX: int = 8192
    VLM_MAX_TOKENS: int = 1536
    VLM_TEMPERATURE: float = 0.0

    # ── Ingest tahmini (deneysel; toggle ile aç/kapa) ────────────────────
    # False olduğunda /documents/estimate endpoint 404 döner ve frontend
    # eski akışı (estimate adımı atlanır) kullanır. Yeni VLM/OCR motorları
    # denenirse aşağıdaki süre sabitlerini güncellemek yeterli — tahmin
    # mantığı bu sabitler üzerinden çalışıyor, modülün kendisi değişmez.
    INGEST_ESTIMATE_ENABLED: bool = True

    # Sabitler 2026-05-20 tarihli 4 dosyalı (142 sayfa, 72 görsel)
    # kontrol çalışmasına dayanır. Doğrulama: VLM açık 448 vs 438.8 sn (%2),
    # VLM kapalı 46 vs 41.6 sn (%12). Hepsi saniye cinsinden.
    INGEST_VLM_LOAD_SECONDS: float = 2.0  # VLM motoru yüklenme
    INGEST_PARSE_PER_PAGE: float = 0.6  # pymupdf4llm layout parse
    INGEST_VLM_PER_IMAGE: float = 5.5  # filtre geçen görsel başına
    INGEST_CHUNKER_OVERHEAD: float = 0.5  # dosya başı section + chunker
    INGEST_VRAM_TRANSITION: float = 1.0  # VLM unload + GC
    INGEST_JINA_LOAD_SECONDS: float = 2.0  # Embedding modeli yüklenme
    INGEST_EMBED_PER_PAGE: float = 0.15  # sayfa bazlı embed (chunk≈page)
    INGEST_DB_WRITE_BUFFER: float = 0.5  # catalog yazma + tahliye
