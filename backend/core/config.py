from pathlib import Path


class AppConfig:
    """
    Uygulama genelinde tek ayar kaynağı (single source of truth).

    Tüm dizin yolları, model dosya adları ve chunk/retriever/LLM/VLM
    parametreleri burada toplanır. Diğer modüller bu değerleri sabit
    konumdan okur; sihirli sayılar koda dağılmaz, bir parametreyi
    değiştirmek tek yerden yapılır.

    Yollar __file__'a göre türetilir; proje başka bir klasöre taşınsa
    bile çalışır (mutlak yol gömülü değil).
    """
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
    # SentenceSplitter'ın child node hedefi: ~1100 karakter gövde, 200 karakter
    # örtüşme. Örtüşme, bir cümlenin chunk sınırında ortadan kesilip bağlam
    # kaybetmesini engeller (komşu chunk son ~200 karakteri tekrar görür).
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
    # Vektör araması k=10 aday çeker (recall için geniş tutulur), reranker
    # bunları yeniden sıralayıp en iyi 3'ünü seçer (precision). 3 section,
    # LLM'in bağlam penceresine güvenle sığan üst sınır.
    RETRIEVER_K: int = 10
    RERANKER_TOP_N: int = 3

    # ── LLM ──────────────────────────────────────────────────────────────
    # N_CTX: bağlam penceresi (prompt + cevap toplam token tavanı).
    # MAX_TOKENS: tek cevapta üretilecek en fazla token.
    # TEMPERATURE: 0.1 = neredeyse deterministik; RAG'de uydurmadan çok
    # bağlama sadık, tutarlı cevap istediğimiz için düşük tutuluyor.
    LLM_N_CTX: int = 8192
    LLM_MAX_TOKENS: int = 2048
    LLM_TEMPERATURE: float = 0.1

    # ── VLM ──────────────────────────────────────────────────────────────
    VLM_N_CTX: int = 8192
    VLM_MAX_TOKENS: int = 1536
    # Qwen3-VL greedy'de (0.0) sonsuz tekrara düşer (model kartı + Qwen rehberi:
    # "DO NOT use greedy decoding"). 0.7 = instruct reçetesi: sadık ama sampling
    # açık. top_k/top_p vlm_engine'de set ediliyor.
    VLM_TEMPERATURE: float = 0.7
