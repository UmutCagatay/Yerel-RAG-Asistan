import json
import logging
import os
import re

import chromadb
from core.config import AppConfig
from core.file_utils import atomic_write_json
from core.ingestion_engine import IngestionEngine

log = logging.getLogger(__name__)


class DBManager:
    """
    Veritabanı yönetiminin merkezi sınıfı.

    Tüm doküman ve koleksiyon işlemleri buradan geçer. App.py UI katmanı
    olarak kalır, iş mantığı bu sınıfta toplanır.

    Catalog (documents.json) hangi koleksiyonda hangi doküman var bilgisini
    tutar. ChromaDB ile senkron olmak zorunda — eklenen/silinen her şey
    ikisinde de güncellenmelidir.
    """

    DEFAULT_COLLECTION = "default"
    # ChromaDB kuralı: 3-512 karakter, alfanumerik + . _ -, başı/sonu alfanumerik
    COLLECTION_NAME_PATTERN = re.compile(
        r"^[a-zA-Z0-9][a-zA-Z0-9._-]{1,48}[a-zA-Z0-9]$"
    )

    def __init__(self, persist_dir: str = str(AppConfig.DATABASE_DIR)):
        self.persist_dir = persist_dir
        self.catalog_path = os.path.join(self.persist_dir, AppConfig.CATALOG_FILENAME)
        self.sections_path = os.path.join(self.persist_dir, AppConfig.SECTIONS_FILENAME)
        self.active_collection: str = self.DEFAULT_COLLECTION

        # Dizin yoksa oluştur — ChromaDB de aynı dizine yazacak
        os.makedirs(self.persist_dir, exist_ok=True)

        # ChromaDB client — başarısız olursa anlamlı log + yeniden fırlat
        try:
            self.chroma_client = chromadb.PersistentClient(path=self.persist_dir)
            log.debug(f"ChromaDB client başlatıldı: {self.persist_dir}")
        except Exception as e:
            log.critical(f"ChromaDB client başlatılamadı: {e}", exc_info=True)
            raise

        # Catalog yoksa boş bir tane oluştur, default koleksiyon hep var olsun
        if not os.path.exists(self.catalog_path):
            initial = {"collections": {self.DEFAULT_COLLECTION: {"documents": {}}}}
            self._save_catalog(initial)
            log.info(f"Yeni catalog oluşturuldu: {self.catalog_path}")

        # Önceki oturumdan kalan orphan kayıtları temizle:
        #   1) Catalog'da yok ama DB'de var olan dokümanlar (crash recovery)
        #   2) Aktif olmayan ChromaDB segment klasörleri
        self._cleanup_orphan_documents()
        self._cleanup_orphan_segments()

    # ── Catalog dosyası okuma/yazma ──────────────────────────────────────────

    def _load_catalog(self) -> dict:
        """Catalog'u diskten okur, dict olarak döndürür."""
        try:
            with open(self.catalog_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            log.critical(
                f"Catalog dosyası bozuk: {self.catalog_path} — {e}",
                exc_info=True,
            )
            raise
        except OSError as e:
            log.error(f"Catalog dosyası okunamadı: {e}", exc_info=True)
            raise

    def _save_catalog(self, catalog: dict) -> None:
        """
        Catalog'u diske yazar. Tüm yazma işlemleri buradan geçer.
        Atomik yazım: çökme/elektrik kesintisinde dosya bozulmaz.
        """
        try:
            atomic_write_json(self.catalog_path, catalog)
            log.debug(f"Catalog kaydedildi: {self.catalog_path}")
        except OSError as e:
            log.error(f"Catalog dosyası yazılamadı: {e}", exc_info=True)
            raise

    # ── Koleksiyon yönetimi ──────────────────────────────────────────────────

    def list_collections(self) -> list[str]:
        """Catalog'daki tüm koleksiyon adlarını döndürür."""
        catalog = self._load_catalog()
        return list(catalog["collections"].keys())

    def create_collection(self, name: str) -> bool:
        """..."""
        name = name.strip()
        if not name:
            log.warning("Koleksiyon adı boş olamaz.")
            return False

        if not self.COLLECTION_NAME_PATTERN.match(name):
            log.warning(
                f"Geçersiz koleksiyon adı: '{name}'. Kurallar: "
                "3-50 karakter, sadece harf/rakam/._-, başı ve sonu alfanumerik."
            )
            return False

        catalog = self._load_catalog()
        if name in catalog["collections"]:
            log.warning(f"'{name}' adında bir koleksiyon zaten var.")
            return False

        try:
            self.chroma_client.get_or_create_collection(name)
        except Exception as e:
            log.error(f"ChromaDB koleksiyonu oluşturulamadı: {e}", exc_info=True)
            return False

        catalog["collections"][name] = {"documents": {}}
        self._save_catalog(catalog)
        log.info(f"Koleksiyon oluşturuldu: '{name}'")
        return True

    def delete_collection(self, name: str, chat_manager=None) -> bool:
        """..."""
        if name == self.DEFAULT_COLLECTION:
            log.warning(f"'{self.DEFAULT_COLLECTION}' koleksiyonu silinemez.")
            return False

        catalog = self._load_catalog()
        if name not in catalog["collections"]:
            log.warning(f"'{name}' adında koleksiyon yok.")
            return False

        log.info(f"Koleksiyon silme başlatıldı: '{name}'")

        try:
            self.chroma_client.delete_collection(name)
        except Exception as e:
            log.warning(f"ChromaDB tarafında silme hatası: {e}", exc_info=True)

        self._delete_sections_by_collection(name)

        del catalog["collections"][name]
        self._save_catalog(catalog)

        if self.active_collection == name:
            self.active_collection = self.DEFAULT_COLLECTION
            log.info(f"Aktif koleksiyon '{self.DEFAULT_COLLECTION}'a alındı.")

        # Bu koleksiyona ait sohbetleri de sil — aksi halde diskte yetim kalır.
        # ChatManager verilmezse (eski çağrılar) atlanır, non-fatal.
        if chat_manager is not None:
            try:
                chat_manager.delete_chats_by_collection(name)
            except Exception as e:
                log.warning(
                    f"Koleksiyon sohbetleri silinirken hata: {e}", exc_info=True
                )

        self._vacuum_db()
        log.info(f"Koleksiyon silindi: '{name}'")
        log.info("Segment klasörleri programı yeniden başlattığında temizlenecek.")
        return True

    def rename_collection(
        self,
        old_name: str,
        new_name: str,
        chat_manager=None,
    ) -> dict:
        """
        Bir koleksiyonu yeniden adlandırır. Atomik olmayan çok-fazlı işlem,
        her faz log'a yazılır.

        Fazlar:
            1) ChromaDB collection.modify(name=new_name) + chunks
               metadata'larındaki collection_name update
            2) sections.json collection_name güncelle
            3) Catalog anahtar rename
            4) Aktif koleksiyon ise self.active_collection güncelle
            5) ChatManager varsa sohbetlerin collection field update

        Kurallar:
            - 'default' yeniden adlandırılamaz
            - new_name ChromaDB pattern'ine uymalı
            - new_name zaten varsa çakışma

        Dönüş:
            {"renamed": True, "new_name": str, "chats_updated": int}
            {"renamed": False, "reason": str}
        """
        old_name = (old_name or "").strip()
        new_name = (new_name or "").strip()

        if not old_name or not new_name:
            return {"renamed": False, "reason": "İsimler boş olamaz."}

        if old_name == new_name:
            return {"renamed": True, "new_name": new_name, "chats_updated": 0}

        if old_name == self.DEFAULT_COLLECTION:
            return {
                "renamed": False,
                "reason": f"'{self.DEFAULT_COLLECTION}' yeniden adlandırılamaz.",
            }

        if not self.COLLECTION_NAME_PATTERN.match(new_name):
            return {
                "renamed": False,
                "reason": (
                    "Geçersiz koleksiyon adı. "
                    "3-50 karakter, harf/rakam/._-, başı ve sonu alfanumerik olmalı."
                ),
            }

        catalog = self._load_catalog()

        if old_name not in catalog["collections"]:
            return {
                "renamed": False,
                "reason": f"'{old_name}' adında koleksiyon yok.",
            }

        if new_name in catalog["collections"]:
            return {
                "renamed": False,
                "reason": f"'{new_name}' zaten var, çakışma.",
            }

        log.info(f"Koleksiyon yeniden adlandırılıyor: '{old_name}' -> '{new_name}'")

        # 1) ChromaDB collection rename + chunks metadata update
        # Önce collection adını değiştir, sonra yeni isimle alıp chunks
        # metadata'larındaki collection_name'i güncelle.
        try:
            chroma_col = self.chroma_client.get_collection(old_name)
            chroma_col.modify(name=new_name)
            log.debug(f"ChromaDB collection rename başarılı: '{new_name}'")

            # Yeni isimle tekrar al — referans davranışı garantisiz, defansif
            chroma_col = self.chroma_client.get_collection(new_name)
            result = chroma_col.get(include=["metadatas"])
            ids = result.get("ids", []) or []
            metadatas = result.get("metadatas", []) or []

            if ids:
                updated_metas = [
                    {**(md or {}), "collection_name": new_name} for md in metadatas
                ]
                chroma_col.update(ids=ids, metadatas=updated_metas)
                log.info(f"{len(ids)} chunk için collection_name metadata güncellendi.")
        except Exception as e:
            log.error(
                f"ChromaDB koleksiyon güncellemesi başarısız: {e}",
                exc_info=True,
            )
            return {"renamed": False, "reason": f"ChromaDB hatası: {e}"}

        # 2) sections.json collection_name update
        self._rename_sections_collection(old_name, new_name)

        # 3) Catalog anahtar rename
        catalog["collections"][new_name] = catalog["collections"].pop(old_name)
        self._save_catalog(catalog)

        # 4) Aktif koleksiyon ise state güncelle
        if self.active_collection == old_name:
            self.active_collection = new_name
            log.info(f"Aktif koleksiyon da yeniden adlandırıldı: '{new_name}'")

        # 5) Sohbetlerin collection field update (ChatManager varsa)
        chats_updated = 0
        if chat_manager is not None:
            try:
                chats_updated = chat_manager.rename_collection_in_chats(
                    old_name, new_name
                )
            except Exception as e:
                # Sohbet güncellemesi non-fatal — ana operasyon başarılı
                log.warning(
                    f"Sohbet güncellemesi sırasında hata: {e}",
                    exc_info=True,
                )

        log.info(
            f"Koleksiyon yeniden adlandırma tamamlandı: "
            f"'{old_name}' -> '{new_name}', {chats_updated} sohbet güncellendi."
        )
        return {
            "renamed": True,
            "new_name": new_name,
            "chats_updated": chats_updated,
        }

    def _rename_sections_collection(self, old_name: str, new_name: str) -> None:
        """sections.json'da bir koleksiyon adının tüm geçişlerini günceller."""
        if not os.path.exists(self.sections_path):
            return

        try:
            with open(self.sections_path, "r", encoding="utf-8") as f:
                sections = json.load(f)
        except Exception as e:
            log.error(f"sections.json okunamadı: {e}", exc_info=True)
            return

        changed = 0
        for data in sections.values():
            md = data.get("metadata", {})
            if md.get("collection_name") == old_name:
                md["collection_name"] = new_name
                changed += 1

        if changed == 0:
            return

        try:
            atomic_write_json(self.sections_path, sections)
            log.debug(
                f"{changed} section collection_name güncellendi: "
                f"'{old_name}' -> '{new_name}'"
            )
        except Exception as e:
            log.error(f"sections.json yazılamadı: {e}", exc_info=True)

    def set_active_collection(self, name: str) -> bool:
        """Aktif koleksiyonu değiştirir. Hedef koleksiyon var olmak zorunda."""
        catalog = self._load_catalog()
        if name not in catalog["collections"]:
            log.warning(f"'{name}' adında koleksiyon yok.")
            return False
        self.active_collection = name
        log.info(f"Aktif koleksiyon: '{name}'")
        return True

    # ── Yardımcılar ──────────────────────────────────────────────────────────

    def _delete_sections_by_collection(self, collection_name: str) -> None:
        """sections.json'dan belirtilen koleksiyona ait section'ları siler."""
        if not os.path.exists(self.sections_path):
            log.debug("sections.json yok, koleksiyon section silme atlandı.")
            return

        try:
            with open(self.sections_path, "r", encoding="utf-8") as f:
                sections = json.load(f)
        except Exception as e:
            log.error(f"sections.json okunamadı: {e}", exc_info=True)
            return

        # Silmeden önce ve sonra kayıt sayısı — silme operasyonunun gerçekten
        # bir şey yaptığını dosyada görmek için
        before = len(sections)
        filtered = {
            sid: data
            for sid, data in sections.items()
            if data.get("metadata", {}).get("collection_name") != collection_name
        }
        removed = before - len(filtered)

        try:
            atomic_write_json(self.sections_path, filtered)
            log.debug(
                f"sections.json'dan {removed} section silindi "
                f"(koleksiyon: {collection_name})"
            )
        except Exception as e:
            log.error(f"sections.json yazılamadı: {e}", exc_info=True)

    # ── Doküman yönetimi ─────────────────────────────────────────────────────

    def add_documents(
        self,
        file_paths: list[str],
        collection: str | None = None,
        on_conflict: dict[str, str] | str = "ask",
        use_vlm: bool = True,
    ) -> dict:
        """
        Çoklu PDF ekleme. IngestionEngine'i çağırır, catalog'u günceller.

        on_conflict:
            str biçiminde verilirse tüm çakışan dosyalara aynı karar uygulanır:
                "ask"       → çağıran tarafa bırak (app.py kullanıcıya sorar)
                "overwrite" → mevcut dokümanı sil, yeniyi yaz
                "skip"      → mevcut dokümanı atla

            dict biçiminde verilirse her dosya için ayrı karar:
                {"a.pdf": "overwrite", "b.pdf": "skip", ...}
                Çakışmayan dosyalar için dict'te entry olmasa da olur.
        """
        collection = collection or self.active_collection
        catalog = self._load_catalog()

        if collection not in catalog["collections"]:
            log.error(f"'{collection}' adında koleksiyon yok.")
            return {"success": [], "failed": [], "skipped": []}

        # Başlangıç logu — operasyonun nereden tetiklendiği dosyaya işlenir
        log.info(
            f"Doküman ekleme başlatıldı: {len(file_paths)} dosya, "
            f"koleksiyon: '{collection}', on_conflict: {on_conflict!r}"
        )

        # ── Çakışma filtresi ──
        to_process: list[str] = []
        skipped: list[dict] = []

        for path in file_paths:
            file_name = os.path.basename(path)
            exists = file_name in catalog["collections"][collection]["documents"]

            if not exists:
                to_process.append(path)
                continue

            # Bu dosya için geçerli karar nedir?
            # dict ise dosyaya özel karar, yoksa toplu kararı uygula
            if isinstance(on_conflict, dict):
                decision = on_conflict.get(file_name, "ask")
            else:
                decision = on_conflict

            if decision == "overwrite":
                # Üzerine yazma kararı kritik — sonradan "neden silindi?" sorusunda
                # bu satır cevap verecek. Toplu silme metodunu tek elemanlı
                # listeyle çağırıyoruz — tek bir silme yolu olsun.
                log.info(f"'{file_name}' üzerine yazılacak (eski kayıt siliniyor).")
                self.delete_documents([file_name], collection=collection)
                to_process.append(path)
            elif decision == "skip":
                skipped.append({"file_name": file_name, "reason": "Zaten var"})
                log.info(f"'{file_name}' zaten var, atlanıyor.")
            else:  # "ask" — app.py burayı çağırmadan önce karar vermeli
                log.error(
                    f"'{file_name}' zaten var. App katmanı on_conflict kararı "
                    "vermeden bu method çağrılmamalı."
                )
                skipped.append(
                    {"file_name": file_name, "reason": "Çakışma — karar verilmemiş"}
                )

        if not to_process:
            log.warning("İşlenecek dosya kalmadı.")
            return {"success": [], "failed": [], "skipped": skipped}

        # ── IngestionEngine'i çalıştır ──
        engine = IngestionEngine(persist_dir=self.persist_dir)
        result = engine.run(
            file_paths=to_process,
            collection_name=collection,
            use_vlm=use_vlm,
        )

        # ── Catalog güncelle ──
        catalog = self._load_catalog()  # yeniden oku (delete_documents yazmış olabilir)
        for item in result["success"]:
            file_name = item.pop("file_name")
            catalog["collections"][collection]["documents"][file_name] = item
        self._save_catalog(catalog)

        # Bitiş özeti — bir bakışta operasyonun sonucu
        log.info(
            f"Doküman ekleme tamamlandı: "
            f"{len(result['success'])} başarılı, "
            f"{len(result['failed'])} başarısız, {len(skipped)} atlanan."
        )

        return {
            "success": result["success"],
            "failed": result["failed"],
            "skipped": skipped,
            # VLM yüklenebildi mi bayrağı — UI görsellerin atlandığını
            # kullanıcıya uyarı olarak gösterir.
            "vlm_loaded": result.get("vlm_loaded", True),
        }

    def list_documents(self, collection: str | None = None) -> list[dict]:
        """
        Belirtilen koleksiyondaki tüm dokümanları döndürür.
        collection=None ise aktif koleksiyon kullanılır.

        Dönüş örneği:
            [
                {"file_name": "test1.pdf", "added_at": "...", "chunk_count": 42, ...},
                ...
            ]
        """
        collection = collection or self.active_collection
        catalog = self._load_catalog()

        if collection not in catalog["collections"]:
            log.warning(f"'{collection}' adında koleksiyon yok.")
            return []

        docs = catalog["collections"][collection]["documents"]
        return [{"file_name": name, **info} for name, info in docs.items()]

    def document_exists(self, file_name: str, collection: str | None = None) -> bool:
        """Doküman aktif/belirtilen koleksiyonda kayıtlı mı?"""
        collection = collection or self.active_collection
        catalog = self._load_catalog()
        if collection not in catalog["collections"]:
            return False
        return file_name in catalog["collections"][collection]["documents"]

    def delete_documents(
        self, file_names: list[str], collection: str | None = None
    ) -> dict:
        """
        Birden fazla dokümanı toplu siler. VACUUM ve segment temizliği
        en sonda bir kez çalışır.

        Dönüş: {"deleted": [...], "failed": [...]}
        """
        collection = collection or self.active_collection
        deleted: list[str] = []
        failed: list[dict] = []

        catalog = self._load_catalog()
        if collection not in catalog["collections"]:
            log.error(f"'{collection}' adında koleksiyon yok.")
            return {"deleted": [], "failed": []}

        # Operasyon başlangıcı — kaç dosyanın hangi koleksiyonda silinmeye
        # çalışıldığı dosyaya kaydoluyor
        log.info(
            f"Toplu doküman silme başlatıldı: {len(file_names)} dosya, "
            f"koleksiyon: '{collection}'"
        )

        for file_name in file_names:
            if file_name not in catalog["collections"][collection]["documents"]:
                failed.append({"file_name": file_name, "reason": "Koleksiyonda yok"})
                log.warning(f"'{file_name}' koleksiyonda yok, atlanıyor.")
                continue

            try:
                chroma_col = self.chroma_client.get_or_create_collection(collection)
                chroma_col.delete(where={"file_name": file_name})
            except Exception as e:
                failed.append({"file_name": file_name, "reason": f"ChromaDB: {e}"})
                # ERROR seviyesinde — kullanıcı verisi yarım silinmiş olabilir,
                # bu bilgi mutlaka dosyaya zengin biçimde gitmeli
                log.error(f"'{file_name}' ChromaDB'den silinemedi: {e}", exc_info=True)
                continue

            self._delete_sections_by_document(file_name, collection)
            del catalog["collections"][collection]["documents"][file_name]
            deleted.append(file_name)
            # Her başarılı silmeyi tek tek DEBUG'a yaz — dosyada tam sıralı iz olsun
            log.debug(f"Doküman silindi: '{file_name}'")

        self._save_catalog(catalog)

        if deleted:
            self._vacuum_db()
            log.info(f"{len(deleted)} doküman silindi, {len(failed)} başarısız.")

        return {"deleted": deleted, "failed": failed}

    def rename_document(
        self,
        old_name: str,
        new_name: str,
        collection: str | None = None,
    ) -> dict:
        """
        Bir dokümanı yeniden adlandırır. Chunk'lar ve embedding'ler yerinde
        kalır — sadece metadata'daki file_name değişir, sections.json'da
        ilgili kayıtlar güncellenir ve catalog'da anahtar değişir.

        Embedding hesaplaması yok, bu yüzden hızlı ve risksiz.

        Dönüş:
            {"renamed": True, "new_name": "..."} başarılı
            {"renamed": False, "reason": "..."} başarısız
        """
        old_name = (old_name or "").strip()
        new_name = (new_name or "").strip()

        if not old_name or not new_name:
            return {"renamed": False, "reason": "İsimler boş olamaz."}

        if old_name == new_name:
            # No-op — başarılı say
            return {"renamed": True, "new_name": new_name}

        collection = collection or self.active_collection
        catalog = self._load_catalog()

        if collection not in catalog["collections"]:
            return {"renamed": False, "reason": f"'{collection}' koleksiyonu yok."}

        docs = catalog["collections"][collection]["documents"]

        if old_name not in docs:
            return {
                "renamed": False,
                "reason": f"'{old_name}' '{collection}' içinde yok.",
            }

        if new_name in docs:
            return {
                "renamed": False,
                "reason": f"'{new_name}' zaten var, çakışma.",
            }

        log.info(
            f"Doküman yeniden adlandırılıyor: '{old_name}' → '{new_name}' "
            f"(koleksiyon: '{collection}')"
        )

        # 1) ChromaDB chunk metadata update
        # Embedding'ler yerinde kalır, sadece metadata.file_name değişir.
        # ChromaDB update(ids=..., metadatas=...) ile yapılır.
        try:
            chroma_col = self.chroma_client.get_or_create_collection(collection)
            result = chroma_col.get(
                where={"file_name": old_name},
                include=["metadatas"],
            )
            ids = result.get("ids", []) or []
            old_metas = result.get("metadatas", []) or []

            if ids:
                new_metas = [{**(md or {}), "file_name": new_name} for md in old_metas]
                chroma_col.update(ids=ids, metadatas=new_metas)
                log.info(f"{len(ids)} chunk metadata güncellendi.")
            else:
                log.warning(
                    f"'{old_name}' için ChromaDB'de chunk bulunamadı — "
                    f"yine de catalog/sections güncellenecek."
                )
        except Exception as e:
            log.error(f"ChromaDB metadata güncellenemedi: {e}", exc_info=True)
            return {"renamed": False, "reason": f"ChromaDB hatası: {e}"}

        # 2) sections.json file_name update
        self._rename_sections_document(old_name, new_name, collection)

        # 3) Catalog anahtar rename
        docs[new_name] = docs.pop(old_name)
        self._save_catalog(catalog)

        log.info(f"Yeniden adlandırma tamamlandı: '{old_name}' → '{new_name}'")
        return {"renamed": True, "new_name": new_name}

    def reorder_documents(
        self,
        file_names_in_order: list[str],
        collection: str | None = None,
    ) -> dict:
        """
        Bir koleksiyondaki dokümanların sırasını verilen listeye göre
        yeniden düzenler. Catalog dict'i Python 3.7+ insertion-order'a sahip;
        dict'i yeni sırayla yeniden inşa edip yazıyoruz.

        ChromaDB ve sections.json'a dokunulmaz — sıralama sadece UI için
        catalog metadata'ı üzerinden yürütülüyor.

        Validation: gelen liste koleksiyonun tüm dokümanlarını içermeli
        (ne eksik ne fazla). Aksi takdirde reddedilir.

        Dönüş:
            {"reordered": True, "count": int}
            {"reordered": False, "reason": str}
        """
        collection = collection or self.active_collection
        catalog = self._load_catalog()

        if collection not in catalog["collections"]:
            return {
                "reordered": False,
                "reason": f"'{collection}' koleksiyonu yok.",
            }

        docs = catalog["collections"][collection]["documents"]
        current = set(docs.keys())
        provided = set(file_names_in_order)

        if current != provided:
            missing = current - provided
            extra = provided - current
            return {
                "reordered": False,
                "reason": (
                    f"Liste tutarsız. "
                    f"Eksik: {sorted(missing)}, Fazla: {sorted(extra)}"
                ),
            }

        # Yeni sırayla dict'i yeniden inşa et. Mevcut info'lar korunur,
        # sadece anahtar sırası değişir.
        catalog["collections"][collection]["documents"] = {
            name: docs[name] for name in file_names_in_order
        }
        self._save_catalog(catalog)

        log.info(
            f"Doküman sırası güncellendi (koleksiyon: '{collection}', "
            f"{len(file_names_in_order)} doküman)."
        )
        return {"reordered": True, "count": len(file_names_in_order)}

    # ── Yardımcılar ──────────────────────────────────────────────────────────

    def _delete_sections_by_document(
        self, file_name: str, collection_name: str
    ) -> None:
        """
        sections.json'dan belirtilen doküman+koleksiyon kombinasyonuna ait
        section'ları siler.
        """
        if not os.path.exists(self.sections_path):
            log.debug("sections.json yok, doküman section silme atlandı.")
            return

        try:
            with open(self.sections_path, "r", encoding="utf-8") as f:
                sections = json.load(f)
        except Exception as e:
            log.error(f"sections.json okunamadı: {e}", exc_info=True)
            return

        before = len(sections)
        filtered = {
            sid: data
            for sid, data in sections.items()
            if not (
                data.get("metadata", {}).get("file_name") == file_name
                and data.get("metadata", {}).get("collection_name") == collection_name
            )
        }
        removed = before - len(filtered)

        try:
            atomic_write_json(self.sections_path, filtered)
            log.debug(
                f"sections.json'dan {removed} section silindi "
                f"(doküman: {file_name}, koleksiyon: {collection_name})"
            )
        except Exception as e:
            log.error(f"sections.json yazılamadı: {e}", exc_info=True)

    def _rename_sections_document(
        self,
        old_name: str,
        new_name: str,
        collection_name: str,
    ) -> None:
        """
        sections.json'da bir (doküman + koleksiyon) kombinasyonunun
        tüm section kayıtlarında file_name'i günceller. Atomik yazım.
        """
        if not os.path.exists(self.sections_path):
            return

        try:
            with open(self.sections_path, "r", encoding="utf-8") as f:
                sections = json.load(f)
        except Exception as e:
            log.error(f"sections.json okunamadı: {e}", exc_info=True)
            return

        changed = 0
        for data in sections.values():
            md = data.get("metadata", {})
            if (
                md.get("file_name") == old_name
                and md.get("collection_name") == collection_name
            ):
                md["file_name"] = new_name
                changed += 1

        if changed == 0:
            log.debug(f"sections.json'da '{old_name}' için güncellenecek kayıt yok.")
            return

        try:
            atomic_write_json(self.sections_path, sections)
            log.debug(
                f"{changed} section file_name güncellendi "
                f"(doküman: '{old_name}' → '{new_name}', koleksiyon: '{collection_name}')"
            )
        except Exception as e:
            log.error(f"sections.json yazılamadı: {e}", exc_info=True)

    def _vacuum_db(self) -> None:
        """
        SQLite freelist'i temizler, dosyayı kompakt yapar.

        ChromaDB silme yapsa bile SQLite sayfaları "freelist"e atıyor,
        dosya boyutunu küçültmüyor. VACUUM bunu çözer. Hızlı operasyon
        (küçük dosyalarda ~100ms), her silme sonrası çağırmak güvenli.
        """
        import sqlite3

        sqlite_path = os.path.join(self.persist_dir, "chroma.sqlite3")
        if not os.path.exists(sqlite_path):
            log.debug("VACUUM atlandı: chroma.sqlite3 yok.")
            return
        try:
            # Önce/sonra boyut karşılaştırması debug açısından çok değerli;
            # silmenin gerçekten dosyayı küçülttüğünü dosyada görebilirsin
            size_before = os.path.getsize(sqlite_path)
            conn = sqlite3.connect(sqlite_path)
            conn.execute("VACUUM")
            conn.close()
            size_after = os.path.getsize(sqlite_path)
            log.debug(
                f"VACUUM tamamlandı: {size_before / 1024:.0f} KB → "
                f"{size_after / 1024:.0f} KB"
            )
        except Exception as e:
            log.warning(f"VACUUM sırasında hata: {e}", exc_info=True)

    def _cleanup_orphan_documents(self) -> None:
        """
        Catalog'da olmayan ama ChromaDB'de chunk'ları olan yetim dokümanları
        temizler. Çökme/elektrik kesintisi sonrası tutarsızlığı düzeltir.

        Senaryo: ingestion sırasında DB'ye chunk'lar yazıldı ama catalog'a
        yazılamadan crash oldu. Sonuç: DB'de yetim chunk'lar, catalog'da kayıt yok.
        Eğer temizlenmezse sorgular bu yetim chunk'ları sonuçlara katıp
        yanlış cevap üretir, daha kötüsü kullanıcı dosyayı silseler bile
        "hayalet" sonuçlar geri gelir.

        Bu metod startup'ta çalışır. Her koleksiyonun ChromaDB metadata'larını
        tarar, catalog'da olmayan file_name'leri tespit eder ve siler.
        """
        if not os.path.exists(self.catalog_path):
            return

        try:
            catalog = self._load_catalog()
        except Exception as e:
            log.warning(f"Catalog okunamadı, orphan doc check atlandı: {e}")
            return

        total_removed = 0

        for collection_name in list(catalog["collections"].keys()):
            catalog_files = set(
                catalog["collections"][collection_name]["documents"].keys()
            )

            try:
                chroma_col = self.chroma_client.get_or_create_collection(
                    collection_name
                )
                # Embedding'i atla, sadece metadata'ları çek — çok daha hızlı
                result = chroma_col.get(include=["metadatas"])
                metadatas = result.get("metadatas", []) or []
            except Exception as e:
                log.warning(
                    f"'{collection_name}' için metadata okunamadı, atlandı: {e}",
                    exc_info=True,
                )
                continue

            # DB'deki unique file_name'leri çıkar
            db_files = {md["file_name"] for md in metadatas if md and "file_name" in md}

            # Catalog'da olmayan = yetim
            orphan_files = db_files - catalog_files

            for orphan in orphan_files:
                try:
                    chroma_col.delete(where={"file_name": orphan})
                    self._delete_sections_by_document(orphan, collection_name)
                    log.info(
                        f"Yetim doküman temizlendi: '{orphan}' "
                        f"(koleksiyon: '{collection_name}')"
                    )
                    total_removed += 1
                except Exception as e:
                    log.warning(f"Yetim '{orphan}' silinemedi: {e}", exc_info=True)

        if total_removed:
            log.info(
                f"Toplam {total_removed} yetim doküman temizlendi (çökme recovery)."
            )
            # Sildiğimiz kayıtlar için SQLite freelist'i sıkıştır
            self._vacuum_db()

    def _cleanup_orphan_segments(self) -> None:
        import shutil
        import sqlite3

        sqlite_path = os.path.join(self.persist_dir, "chroma.sqlite3")
        if not os.path.exists(sqlite_path):
            return

        # Aktif segment ID'lerini SQLite'tan oku
        try:
            conn = sqlite3.connect(sqlite_path)
            cur = conn.cursor()
            cur.execute("SELECT id FROM segments")
            active_ids = {row[0] for row in cur.fetchall()}
            conn.close()
        except Exception as e:
            log.warning(f"Segment ID'leri okunamadı: {e}", exc_info=True)
            return

        # persist_dir altındaki UUID klasörlerini tara
        uuid_pattern = re.compile(
            r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
            re.IGNORECASE,
        )
        removed = 0
        failed = 0
        for entry in os.listdir(self.persist_dir):
            full_path = os.path.join(self.persist_dir, entry)
            if not os.path.isdir(full_path):
                continue
            if not uuid_pattern.match(entry):
                continue
            if entry in active_ids:
                continue
            # Orphan — sil
            try:
                shutil.rmtree(full_path)
                removed += 1
                log.debug(f"Orphan segment silindi: {entry}")
            except Exception as e:
                # Beklenen davranış: Windows mmap kilidi, çalışma sırasında
                # silinemez, sonraki açılışta otomatik temizlenir.
                # Bu yüzden DEBUG seviyesi yeterli, terminal kirletmiyor.
                failed += 1
                log.debug(f"Orphan segment silinemedi ({entry}): {e}")

        if removed:
            log.info(f"{removed} orphan segment klasörü temizlendi.")
        if failed:
            log.debug(f"{failed} orphan segment kilitli (sonraki açılışta denenecek).")
