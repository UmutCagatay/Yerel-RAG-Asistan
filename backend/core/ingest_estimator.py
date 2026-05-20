"""
Ingestion süre tahmini.

Bu modül VLM/OCR motorundan bağımsızdır: PDF dosyalarının sayfa ve
görsel sayısını ölçer, AppConfig.INGEST_* sabitleri üzerinden toplam
ingest süresini hesaplar. Yeni bir VLM/OCR denendiğinde sadece config
sabitleri güncellenir, bu modüle dokunulmaz.

Aktörler:
    fitz (PyMuPDF)  →  PDF açar, sayfa ve görselleri çeker.
    PIL + numpy     →  Dekoratif görsel filtresi (document_parser ile
                        aynı kurallar). Tahmin doğruluğu için birebir uyum.
    AppConfig       →  Süre sabitleri ve toggle.

Endpoint /documents/estimate buradan scan_pdf + estimate_total_seconds
çağırır; PDF processing pipeline (document_parser, ingestion_engine)
bu modüle bağımlı değildir.
"""

import io
import logging
from pathlib import Path

import fitz  # PyMuPDF
import numpy as np
from core.config import AppConfig
from PIL import Image

log = logging.getLogger(__name__)


# ── Dekoratif görsel filtresi ────────────────────────────────────────────────
#
# DocumentParser._is_decorative_image ile AYNI üç kuralı uygular. İki yerde
# aynı mantığı tutmak DRY ihlali sayılabilir ama estimator'ı document_parser'a
# bağımlı yapmamak için (PDF processing pipeline tamamen ayrı kalsın
# diye) küçük bir kopya kabul edildi.

def _is_decorative_pil(img: Image.Image) -> bool:
    """
    Bir PIL Image'ın dekoratif/gürültü mü yoksa gerçek içerik mi
    olduğunu üç bağımsız kuralla belirler.

    Kural 1 — Boyut:   Toplam piksel < 4000 veya en kısa kenar < 20px
    Kural 2 — En-boy:  Oran > 20 veya < 0.05
    Kural 3 — Renk:    Std sapma < 18 → neredeyse tek renkli
    """
    try:
        img = img.convert("RGB")
        w, h = img.size

        if w * h < 4000:
            return True
        if min(w, h) < 20:
            return True

        ratio = w / h
        if ratio > 20.0 or ratio < 0.05:
            return True

        arr = np.array(img, dtype=np.float32)
        step_y = max(1, h // 60)
        step_x = max(1, w // 60)
        sampled = arr[::step_y, ::step_x]
        if np.std(sampled) < 18.0:
            return True

        return False
    except Exception:
        # Bozuk görsel: dekoratif say (VLM'e gönderilmeyeceğini varsay).
        # Tahmin sapması kabul edilebilir.
        return True


# ── PDF tarama ───────────────────────────────────────────────────────────────

def scan_pdf(pdf_path: Path) -> tuple[int, int]:
    """
    PDF'i hızlıca tarayarak (page_count, filtered_image_count) döndürür.
    Görselleri diske yazmaz, bellek üzerinde filtreden geçirir.

    Not: PyMuPDF'in page.get_images() metodu pymupdf4llm'in tespit ettiği
    görsellerle birebir aynı sayıyı vermeyebilir; vector graphics, embedded
    SVG vb. farklı sayılabilir. Pratikte bu sapma %10-20 mertebesinde,
    tahmin için kabul edilebilir.
    """
    page_count = 0
    image_count = 0

    try:
        with fitz.open(pdf_path) as doc:
            page_count = doc.page_count

            for page in doc:
                for img_info in page.get_images(full=True):
                    xref = img_info[0]
                    try:
                        base = doc.extract_image(xref)
                        pil = Image.open(io.BytesIO(base["image"]))
                        if not _is_decorative_pil(pil):
                            image_count += 1
                    except Exception:
                        # Tek bir bozuk görsel yüzünden döngü çökmesin
                        continue
    except Exception as e:
        log.warning(f"scan_pdf hatası ({pdf_path.name}): {e}")
        # Boş değerlerle dön; üst katman dosyayı atlamış gibi davranır
        return 0, 0

    return page_count, image_count


# ── Toplam süre tahmini ──────────────────────────────────────────────────────

def estimate_total_seconds(
    files: list[tuple[str, Path]],
    use_vlm: bool,
) -> dict:
    """
    Birden fazla PDF için toplam ingest süresini tahmin eder.

    Formul (saniye):
        toplam =
            use_vlm * (VLM_LOAD + VRAM_TRANSITION)
          + sum(pages) * PARSE_PER_PAGE
          + sum(images) * VLM_PER_IMAGE * use_vlm
          + n_files * CHUNKER_OVERHEAD
          + JINA_LOAD
          + sum(pages) * EMBED_PER_PAGE
          + DB_WRITE_BUFFER

    Args:
        files: [(filename, path), ...] dosya kim ve nerede.
        use_vlm: VLM toggle durumu.

    Returns:
        {
            "files": [{"name": str, "pages": int, "images": int, "seconds": float}, ...],
            "total_pages": int,
            "total_images": int,
            "total_seconds": float,
            "use_vlm": bool,
        }
    """
    c = AppConfig
    file_details = []
    total_pages = 0
    total_images = 0

    for name, path in files:
        pages, images = scan_pdf(path)
        total_pages += pages
        total_images += images

        # Dosya başı süre — sadece parse + (varsa) VLM. Embed ve sabit
        # overhead toplama düşer, dosya başı metrik UI'da pratik bilgi.
        file_seconds = pages * c.INGEST_PARSE_PER_PAGE
        if use_vlm:
            file_seconds += images * c.INGEST_VLM_PER_IMAGE

        file_details.append({
            "name": name,
            "pages": pages,
            "images": images if use_vlm else 0,
            "seconds": round(file_seconds, 1),
        })

    n_files = len(files)
    total = 0.0

    if use_vlm:
        total += c.INGEST_VLM_LOAD_SECONDS + c.INGEST_VRAM_TRANSITION
    total += total_pages * c.INGEST_PARSE_PER_PAGE
    if use_vlm:
        total += total_images * c.INGEST_VLM_PER_IMAGE
    total += n_files * c.INGEST_CHUNKER_OVERHEAD
    total += c.INGEST_JINA_LOAD_SECONDS
    total += total_pages * c.INGEST_EMBED_PER_PAGE
    total += c.INGEST_DB_WRITE_BUFFER

    return {
        "files": file_details,
        "total_pages": total_pages,
        "total_images": total_images if use_vlm else 0,
        "total_seconds": round(total, 1),
        "use_vlm": use_vlm,
    }
