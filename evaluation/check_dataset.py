"""
Pre-check: değerlendirme seti ile yüklü koleksiyon tutarlı mı?

Ölçüme başlamadan ÖNCE çalıştırılır. Hiçbir şey ölçmez, hiçbir şeyi
değiştirmez — sadece iki dosyayı okuyup karşılaştırır:

  1. evaluation/degerlendirme_seti.json  (altın set / ground truth)
  2. backend/data/database/documents.json (yüklü dokümanların kataloğu)

Amaç: setteki her 'kaynak_dosya', hedef koleksiyonda BİREBİR aynı adla
yüklü mü? Bir harf/uzantı farkı bile retrieval ölçümünü bozar, çünkü
sistemin getirdiği dosya adı bu sete göre "doğru/yanlış" sayılacak.

Backend'i import ETMEZ (DBManager açılışta temizlik/vacuum yapıyor; salt
okuma için bunu tetiklemek istemeyiz). Sadece stdlib + JSON okuma.

Kullanım:
    python check_dataset.py
    python check_dataset.py <set_yolu> <koleksiyon_adi>

Çıkış kodu:
    0  → her şey tutarlı, ölçüme geçilebilir
    1  → engelleyici sorun var (eksik dosya, bozuk set vb.)
"""

import json
import sys
from pathlib import Path

# evaluation/check_dataset.py → bir üst = proje kökü
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = Path(__file__).resolve().parent / "degerlendirme_seti.json"
CATALOG_PATH = PROJECT_ROOT / "backend" / "data" / "database" / "documents.json"

ALLOWED_TYPES = {"metin", "gorsel", "cok_dokuman", "cevapsiz"}
REQUIRED_FIELDS = {
    "id",
    "soru",
    "ideal_cevap",
    "kaynak_dosya",
    "kaynak_konum",
    "soru_tipi",
    "not",
}


def load_dataset(path: Path) -> list[dict]:
    """Seti oku ve temel yapı doğrulaması yap. Hata varsa mesajla çıkar."""
    if not path.exists():
        sys.exit(f"[HATA] Set bulunamadı: {path}\n"
                 f"       JSON'u bu yola kaydetmelisin.")
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        sys.exit(f"[HATA] Set geçerli JSON değil: {e}")

    if not isinstance(data, list) or not data:
        sys.exit("[HATA] Set boş ya da bir dizi (list) değil.")
    return data


def load_catalog_files(path: Path, collection: str) -> set[str]:
    """Katalogtan hedef koleksiyondaki doküman adlarını oku."""
    if not path.exists():
        sys.exit(f"[HATA] Katalog bulunamadı: {path}\n"
                 f"       Henüz hiç doküman yüklenmemiş olabilir.")
    try:
        with open(path, "r", encoding="utf-8") as f:
            catalog = json.load(f)
    except json.JSONDecodeError as e:
        sys.exit(f"[HATA] Katalog bozuk: {e}")

    collections = catalog.get("collections", {})
    if collection not in collections:
        mevcut = ", ".join(collections.keys()) or "(yok)"
        sys.exit(f"[HATA] '{collection}' koleksiyonu katalogda yok.\n"
                 f"       Mevcut koleksiyonlar: {mevcut}")

    return set(collections[collection].get("documents", {}).keys())


def validate_schema(data: list[dict]) -> list[str]:
    """Her elemanın alanlarını ve tip değerlerini kontrol et. Sorunları döndür."""
    problems: list[str] = []
    seen_ids: set = set()

    for i, item in enumerate(data):
        etiket = f"#{item.get('id', f'satır {i}')}"

        missing = REQUIRED_FIELDS - set(item.keys())
        if missing:
            problems.append(f"{etiket}: eksik alan(lar): {sorted(missing)}")

        tip = item.get("soru_tipi")
        if tip not in ALLOWED_TYPES:
            problems.append(f"{etiket}: geçersiz soru_tipi: {tip!r}")

        # cevapsiz → kaynak_dosya boş olmalı; diğerleri → dolu olmalı
        kaynak = item.get("kaynak_dosya", "")
        if tip == "cevapsiz" and kaynak:
            problems.append(f"{etiket}: cevapsiz ama kaynak_dosya dolu: {kaynak!r}")
        if tip in {"metin", "gorsel", "cok_dokuman"} and not kaynak:
            problems.append(f"{etiket}: {tip} ama kaynak_dosya boş")

        # id tekrarı
        _id = item.get("id")
        if _id in seen_ids:
            problems.append(f"{etiket}: tekrarlayan id")
        seen_ids.add(_id)

    return problems


def main() -> int:
    dataset_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DATASET
    collection = sys.argv[2] if len(sys.argv) > 2 else "default"

    print("=" * 64)
    print("  DEĞERLENDİRME SETİ ÖN-KONTROL")
    print(f"  Set        : {dataset_path}")
    print(f"  Katalog    : {CATALOG_PATH}")
    print(f"  Koleksiyon : {collection}")
    print("=" * 64)

    data = load_dataset(dataset_path)
    catalog_files = load_catalog_files(CATALOG_PATH, collection)

    # ── 1. Şema doğrulaması ──
    schema_problems = validate_schema(data)

    # ── 2. Tip dağılımı ──
    dist: dict[str, int] = {}
    for item in data:
        t = item.get("soru_tipi", "?")
        dist[t] = dist.get(t, 0) + 1

    # ── 3. Dosya eşleşmesi ──
    # cevapsiz sorularda kaynak_dosya boş — onları hariç tut.
    set_files = {
        item["kaynak_dosya"]
        for item in data
        if item.get("kaynak_dosya")
    }
    missing = set_files - catalog_files   # sette var, koleksiyonda yok → ENGELLEYİCİ
    extra = catalog_files - set_files     # koleksiyonda var, sette yok → sadece bilgi

    # ── Rapor ──
    print(f"\nToplam soru     : {len(data)}")
    print(f"Tip dağılımı    : {dist}")
    print(f"Sette geçen dosya: {len(set_files)} benzersiz")
    print(f"Koleksiyondaki  : {len(catalog_files)} doküman")

    if schema_problems:
        print(f"\n[ŞEMA SORUNLARI] {len(schema_problems)} adet:")
        for p in schema_problems:
            print(f"  - {p}")

    if missing:
        print(f"\n[ENGELLEYİCİ] Sette geçen ama koleksiyonda OLMAYAN {len(missing)} dosya:")
        for f in sorted(missing):
            print(f"  - {f}")
        print("  → Bu dokümanları koleksiyona yükle, yoksa retrieval ölçümü yanlış çıkar.")

    if extra:
        print(f"\n[BİLGİ] Koleksiyonda olup sette geçmeyen {len(extra)} doküman:")
        for f in sorted(extra):
            print(f"  - {f}")
        print("  → Sorun değil; ölçümde bu dokümanlardan parça gelirse 'yanlış kaynak' sayılır.")

    # ── Sonuç ──
    print("\n" + "=" * 64)
    if missing or schema_problems:
        print("  SONUÇ: ENGELLEYİCİ SORUN VAR — düzeltmeden ölçüme geçme.")
        print("=" * 64)
        return 1

    print("  SONUÇ: Set ve koleksiyon tutarlı. Ölçüme geçilebilir. ✓")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(main())
