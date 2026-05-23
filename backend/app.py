"""
Terminal arayüzü — GUI'den bağımsız, menü tabanlı giriş noktası.

Tauri/React arayüzü api/main.py üzerinden çalışır; bu dosya ise aynı
çekirdeği (DBManager, QueryEngine) terminalden kullanmayı sağlar — hızlı
test ve sunucusuz kullanım için. `python app.py` ile çalışır.
"""

import logging
import os

from core.config import AppConfig
from core.db_manager import DBManager
from core.logger import setup_logging
from core.query_engine import QueryEngine

setup_logging()
log = logging.getLogger(__name__)


class App:
    """
    Terminal tabanlı RAG asistanı.

    Menü döngüsü ve kullanıcı etkileşimi. Tüm iş mantığı DBManager
    üzerinden geçer; bu sınıf sadece UI katmanıdır.
    """

    def __init__(self):
        self.db = DBManager()

    # ── Ana döngü ────────────────────────────────────────────────────────────

    def run(self) -> None:
        log.info("Uygulama başlatıldı.")
        print("\n" + "=" * 60)
        print("  YEREL RAG ASİSTANI")
        print("=" * 60)

        handlers = {
            "1": self._handle_add_documents,
            "2": self._handle_delete_document,
            "3": self._handle_list_documents,
            "4": self._handle_change_collection,
            "5": self._handle_create_collection,
            "6": self._handle_delete_collection,
            "7": self._handle_query,
        }

        while True:
            self._show_menu()
            choice = input("Seçim: ").strip()

            if choice == "0":
                log.info("Uygulama kapatıldı.")
                print("\nGörüşmek üzere.")
                break

            handler = handlers.get(choice)
            if handler is None:
                print("[HATA] Geçersiz seçim.")
                continue

            try:
                handler()
            except Exception as e:
                # Beklenmedik hata menüyü çökertmesin.
                # exc_info=True — tam traceback dosyaya gider, terminalde kısa
                # tepki mesajı görünür. Hangi menü seçiminde patladığı da kayıtta.
                log.error(f"Handler hatası (seçim={choice}): {e}", exc_info=True)
                print(f"\n[HATA] İşlem sırasında beklenmedik bir sorun oluştu: {e}")
                print("Menüye dönülüyor.")

    def _show_menu(self) -> None:
        print(f"\nAktif koleksiyon: {self.db.active_collection}")
        print("-" * 40)
        print("1) Doküman ekle (tek/çoklu PDF)")
        print("2) Doküman sil")
        print("3) Doküman listele")
        print("4) Koleksiyon değiştir")
        print("5) Yeni koleksiyon oluştur")
        print("6) Koleksiyon sil")
        print("7) Sorgu yap")
        print("0) Çıkış")

    # ── Handler'lar ──────────────────────────────────────────────────────────

    def _handle_add_documents(self) -> None:
        print("\n--- Doküman Ekle ---")
        print("PDF yollarını boşlukla ayırarak yazın. Tek dosya için tek yol.")
        raw = input("Yol(lar): ").strip()
        if not raw:
            print("[İPTAL]")
            return

        paths = self._split_paths(raw)
        if not paths:
            print("[İPTAL] Geçerli yol yok.")
            return

        # Mevcut olmayanları ve boyut sınırını aşanları erken ayıkla
        valid: list[str] = []
        missing: list[str] = []
        too_big: list[tuple[str, float]] = []  # (path, size_mb)
        for p in paths:
            if not os.path.exists(p):
                missing.append(p)
                continue
            size_mb = os.path.getsize(p) / (1024 * 1024)
            if size_mb > AppConfig.MAX_FILE_SIZE_MB:
                too_big.append((p, size_mb))
                continue
            valid.append(p)

        for m in missing:
            log.warning(f"Dosya bulunamadı, atlandı: {m}")
            print(f"[ATLA] Dosya bulunamadı: {m}")

        for p, size_mb in too_big:
            # Boyut limitini aşan dosya atlanıyor — dosyada iz bırakalım
            log.warning(
                f"Boyut limiti aşıldı, atlandı: {p} "
                f"({size_mb:.1f} MB > {AppConfig.MAX_FILE_SIZE_MB} MB)"
            )
            print(
                f"[ATLA] Dosya çok büyük "
                f"({size_mb:.1f} MB > {AppConfig.MAX_FILE_SIZE_MB} MB): {p}"
            )

        if not valid:
            print("[İPTAL] İşlenecek geçerli dosya kalmadı.")
            return

        # ── Çakışan dosyaları tespit et ──
        existing = [
            os.path.basename(p)
            for p in valid
            if self.db.document_exists(os.path.basename(p))
        ]

        # Her dosya için karar haritası — DBManager'a dict olarak geçecek.
        # Çakışmayan dosyalar bu haritada görünmez; DBManager onları zaten
        # direkt ekleme yoluna alır.
        on_conflict: dict[str, str] = {}

        if existing:
            print(f"\nBu {len(existing)} dosya koleksiyonda zaten var:")
            # Multi-select: kullanıcı üzerine yazılacakları seçer,
            # seçilmeyenler otomatik olarak skip olur
            to_overwrite = self._select_multiple_from_list(
                existing,
                "Üzerine yazılacakları seç (seçilmeyenler atlanacak)",
            )
            for fname in existing:
                on_conflict[fname] = "overwrite" if fname in to_overwrite else "skip"

            # Kullanıcıya seçimin özetini göster ki ne onayladığını bilsin
            overwrite_count = sum(1 for v in on_conflict.values() if v == "overwrite")
            skip_count = sum(1 for v in on_conflict.values() if v == "skip")
            print(
                f"\nKarar: {overwrite_count} dosya üzerine yazılacak, "
                f"{skip_count} dosya atlanacak."
            )

        # İşlemi başlat — çakışma yoksa on_conflict boş dict,
        # DBManager zaten "zaten var" durumuna düşmez
        result = self.db.add_documents(file_paths=valid, on_conflict=on_conflict)

        # Özet
        print("\n--- Özet ---")
        print(f"Başarılı : {len(result['success'])}")
        print(f"Başarısız: {len(result['failed'])}")
        print(f"Atlanan  : {len(result['skipped'])}")
        for f in result["failed"]:
            print(f"  [HATA] {f['file_name']}: {f['reason']}")

    def _handle_delete_document(self) -> None:
        print("\n--- Doküman Sil ---")
        docs = self.db.list_documents()
        if not docs:
            print("Aktif koleksiyonda doküman yok.")
            return

        selected = self._select_multiple_from_list(
            [d["file_name"] for d in docs], "Silinecek dokümanları seç"
        )
        if not selected:
            print("[İPTAL]")
            return

        print(f"\nSilinecek: {', '.join(selected)}")
        if not self._ask_yes_no("Onaylıyor musun?"):
            print("[İPTAL]")
            return

        result = self.db.delete_documents(selected)
        if result["failed"]:
            print(f"\n{len(result['failed'])} doküman silinemedi:")
            for f in result["failed"]:
                print(f"  - {f['file_name']}: {f['reason']}")

    def _handle_list_documents(self) -> None:
        print(f"\n--- Dokümanlar (koleksiyon: {self.db.active_collection}) ---")
        docs = self.db.list_documents()
        if not docs:
            print("Bu koleksiyonda doküman yok.")
            return

        for i, d in enumerate(docs, 1):
            chunk = d.get("chunk_count", "?")
            section = d.get("section_count", "?")
            added = d.get("added_at", "?")
            print(f"  {i}. {d['file_name']}")
            print(f"     chunk: {chunk}  |  section: {section}  |  eklendi: {added}")

    def _handle_change_collection(self) -> None:
        print("\n--- Koleksiyon Değiştir ---")
        collections = self.db.list_collections()
        selected = self._select_from_list(collections, "Aktif yapılacak koleksiyon")
        if selected is None:
            return
        self.db.set_active_collection(selected)

    def _handle_create_collection(self) -> None:
        print("\n--- Yeni Koleksiyon ---")
        name = input("Koleksiyon adı: ").strip()
        if not name:
            print("[İPTAL]")
            return
        self.db.create_collection(name)

    def _handle_delete_collection(self) -> None:
        print("\n--- Koleksiyon Sil ---")
        # Default'u zaten DBManager engelliyor, listede gösterelim ama uyaralım
        collections = [
            c for c in self.db.list_collections() if c != self.db.DEFAULT_COLLECTION
        ]
        if not collections:
            print("Silinebilir koleksiyon yok. (default silinemez)")
            return

        selected = self._select_from_list(collections, "Silinecek koleksiyon")
        if selected is None:
            return

        if not self._ask_yes_no(
            f"'{selected}' koleksiyonu ve içindeki tüm dokümanlar silinecek. Emin misin?"
        ):
            print("[İPTAL]")
            return

        self.db.delete_collection(selected)

    def _handle_query(self) -> None:
        print("\n--- Sorgu ---")
        docs = self.db.list_documents()
        if not docs:
            print("Aktif koleksiyonda doküman yok. Önce doküman ekle.")
            return

        selected = self._select_from_list(
            [d["file_name"] for d in docs], "Sorgu yapılacak doküman"
        )
        if selected is None:
            return

        question = input("Soru: ").strip()
        if not question:
            print("[İPTAL]")
            return

        engine = QueryEngine(collection_name=self.db.active_collection)
        answer = engine.run(question=question, file_name=selected)
        print("=" * 60)
        print(answer)
        print("=" * 60)

    # ── Yardımcılar ──────────────────────────────────────────────────────────

    def _ask_yes_no(self, prompt: str) -> bool:
        ans = input(f"{prompt} [e/h]: ").strip().lower()
        return ans in ("e", "evet", "y", "yes")

    def _select_from_list(self, items: list, prompt: str):
        """Numaralı liste göster, kullanıcı bir seçer. İptal için boş enter."""
        if not items:
            print("Liste boş.")
            return None

        print(f"\n{prompt}:")
        for i, item in enumerate(items, 1):
            print(f"  {i}) {item}")

        raw = input("Numara (boş = iptal): ").strip()
        if not raw:
            return None

        try:
            idx = int(raw) - 1
            if 0 <= idx < len(items):
                return items[idx]
            print("[HATA] Geçersiz numara.")
            return None
        except ValueError:
            print("[HATA] Sayı girilmedi.")
            return None

    def _select_multiple_from_list(self, items: list, prompt: str) -> list:
        """
        Numaralı liste göster, kullanıcı virgülle ayırarak birden fazla seçer.
        Örnek: "1,3,5" veya "1, 3, 5". Boş enter = iptal.
        """
        if not items:
            print("Liste boş.")
            return []

        print(f"\n{prompt}:")
        for i, item in enumerate(items, 1):
            print(f"  {i}) {item}")
        print("  (Virgülle ayır: '1,3,5' — boş = iptal)")

        raw = input("Numaralar: ").strip()
        if not raw:
            return []

        selected = []
        for part in raw.split(","):
            part = part.strip()
            if not part:
                continue
            try:
                idx = int(part) - 1
                if 0 <= idx < len(items):
                    selected.append(items[idx])
                else:
                    print(f"[HATA] Geçersiz numara: {part}")
            except ValueError:
                print(f"[HATA] Sayı değil: {part}")

        # Tekrarları kaldır, sırayı koru
        seen = set()
        unique = []
        for s in selected:
            if s not in seen:
                seen.add(s)
                unique.append(s)
        return unique

    def _split_paths(self, raw: str) -> list[str]:
        """
        Yol stringini parse eder. Tırnaklı yollarda boşluk korunsun
        (örn. 'C:\\My Docs\\a.pdf' "B:\\b.pdf").
        Sade tutmak için shlex kullanıyoruz.
        """
        import shlex

        try:
            return shlex.split(raw)
        except ValueError as e:
            # Eşleşmeyen tırnak vs. — naif fallback.
            # Sessiz kalmamak için dosyaya not düşüyoruz, kullanıcı arayuzünü
            # etkilemiyor (zaten path ayıklanmaya devam ediyor).
            log.warning(f"shlex.split başarısız, naif split'e düşülüyor: {e}")
            return raw.split()


if __name__ == "__main__":
    App().run()
