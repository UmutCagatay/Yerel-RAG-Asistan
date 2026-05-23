"""
Logging kurulumu.

Konsola sade (INFO+), dosyaya zengin (DEBUG+, rotating) iki ayrı akış.
Kullanıcı terminalde özet görür; tam teshis (traceback, süreler, retriever
debug blokları) data/logs/app.log'a gider. setup_logging() bir kez, en başta
(app.py veya api/main.py) çağrılır; idempotent olduğu için tekrar çağrılması
handler çoğaltmaz.
"""

import logging
import os
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler

from core.config import AppConfig


def setup_logging(
    log_dir: str = str(AppConfig.LOGS_DIR),
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
) -> None:
    """
    Uygulama genelinde logging'i kurar. Bir kez, en başta çağrılır.

    İki handler bağlar:
        - Konsol  → INFO ve üstü (kullanıcı için sade)
        - Dosya   → DEBUG ve üstü (debug için zengin), rotating

    Modüller `log = logging.getLogger(__name__)` ile logger alır, bu setup
    onları otomatik miras yoluyla yakalar.

    Çağrı tekrar edilirse handler'lar duplicate olmasın diye önce
    mevcut handler'lar temizlenir.
    """
    os.makedirs(log_dir, exist_ok=True)

    # Windows konsolu default cp1252; Türkçe karakterler UnicodeEncodeError
    # fırlatır ve log handler crash eder, asıl hatayı gizler.
    # UTF-8'e zorla, encode edilemeyen karakteri 'replace' ile sessizce değiştir.
    try:
        if sys.stdout and sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if sys.stderr and sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        # reconfigure mevcut değil veya stream zaten kapalı — sessizce geç
        pass

    root = logging.getLogger()
    root.setLevel(
        logging.DEBUG
    )  # ana logger her şeyi geçirir, filtreyi handler'lar yapar

    # Mevcut handler'ları temizle (idempotent çağrı)
    for handler in root.handlers[:]:
        root.removeHandler(handler)

    # ── Konsol handler — sade format ─────────────────────────────────────────
    console_fmt = logging.Formatter("[%(levelname)s] %(message)s")
    # Explicit sys.stdout: yukarıda reconfigure edilmiş UTF-8 stream'i kullansın.
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(console_fmt)
    root.addHandler(console_handler)

    # ── Dosya handler — zengin format, rotating ──────────────────────────────
    file_fmt = logging.Formatter(
        "%(asctime)s [%(levelname)-7s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler = RotatingFileHandler(
        filename=os.path.join(log_dir, AppConfig.LOG_FILENAME),
        maxBytes=5 * 1024 * 1024,  # 5 MB
        backupCount=5,  # eski log dosyalarını sakla: app.log.1, .2, ...
        encoding="utf-8",
    )
    file_handler.setLevel(file_level)
    file_handler.setFormatter(file_fmt)
    root.addHandler(file_handler)

    # Bazı kütüphaneler çok gürültücü — sustur (INFO altını dosyaya almasınlar)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("langchain").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    # PIL her PNG'de IHDR/pHYs/IDAT chunk debug'ı basıyor — ingestion'da 50+ satır gürültü
    logging.getLogger("PIL").setLevel(logging.WARNING)

    # ── Oturum ayırıcı ───────────────────────────────────────────────────────
    # Yeni çalıştırma başladığında log dosyasına görsel bir sınır yazılır.
    # Sadece dosyaya gider; konsola gerek yok çünkü terminal zaten yeni
    # başlangıçta temiz görünüyor. file_handler.stream üzerinden direkt
    # yazılır, logger üzerinden gitmediği için formatlayıcıdan etkilenmez.
    separator = (
        "\n"
        + "=" * 80
        + "\n"
        + f"YENİ OTURUM — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        + "=" * 80
        + "\n"
    )
    file_handler.stream.write(separator)
    file_handler.flush()
