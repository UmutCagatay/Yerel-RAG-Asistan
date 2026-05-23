"""
Güvenli dosya yazma yardımcıları.

Projedeki tüm JSON yazma işlemleri (catalog, sections.json, sohbet dosyaları)
bu modüldeki atomik yazmadan geçer. Amaç: çökme/elektrik kesintisi anında
yarım yazılmış, parse edilemeyen JSON bırakmamak.
"""

import json
import os
import time


def atomic_write_json(
    path: str, data: dict, retries: int = 5, backoff: float = 0.1
) -> None:
    """
    JSON'u atomik yaz: önce '<path>.tmp' dosyaya yaz, sonra os.replace ile rename.

    Neden? Doğrudan path'e yazarken çökme/elektrik kesintisi olursa dosya yarım
    kalır ve JSON parse edilemez hale gelir. Bu fonksiyon ya tamamen yeni içerik
    yazılmış ya da hiç değişmemiş bir dosya garanti eder.

    Akış:
        1. Geçici dosyaya tam içerik yazılır
        2. f.flush() + os.fsync() ile disk'e fiziksel yazım garanti
        3. os.replace(): atomik rename (Windows + Unix). Yarıda olamaz —
           ya eski dosya, ya yeni dosya. Asla yarım dosya.
        4. Hata olursa .tmp temizlenir, eski dosya bozulmadan kalır.

    Kullanım:
        catalog için, sections.json için, herhangi bir JSON dosya yazımı için.
    """
    tmp_path = path + ".tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())  # disk'e fiziksel yazımı zorla

        # os.replace bazen Windows'ta WinError 5 (erişim engellendi) verir:
        # antivirüs, arama indeksleyici veya bulut senkronu (OneDrive vb.)
        # hedef dosyayı o an kısa süreliğine kilitler. Kilit geçici olduğu
        # için kısa aralıklarla birkaç kez deneriz; hâlâ açılmazsa son hatayı
        # fırlatırız (hatayı gizlemeyiz, sadece geçici kilide tolerans tanırız).
        for attempt in range(retries):
            try:
                os.replace(tmp_path, path)  # atomik rename (POSIX + Windows)
                break
            except PermissionError:
                if attempt == retries - 1:
                    raise
                time.sleep(backoff * (attempt + 1))
    except Exception:
        # Hata olursa .tmp'yi temizle ki bir dahaki açılışta artık dosya kalmasın
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        raise
