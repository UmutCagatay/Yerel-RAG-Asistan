import json
import logging
import os
import uuid
from datetime import datetime
from typing import Optional

from core.config import AppConfig
from core.file_utils import atomic_write_json

log = logging.getLogger(__name__)


class ChatManager:
    """
    Sohbet yönetiminin merkezi sınıfı.

    Her sohbet ayrı bir JSON dosyası: data/chats/<id>.json.
    Sohbet bir koleksiyona kilitlidir (koleksiyon değişirse yeni sohbet
    açılır). Mesajlar sırayla append edilir; user mesajları kendi
    scope'unu (hangi dokümana / tüm koleksiyona) saklar, böylece
    geriye dönük baktığında "bu soru hangi kapsamda soruldu" görülür.

    Dosya yapısı:
        {
            "id": "uuid",
            "title": "...",
            "collection": "default",
            "messages": [
                {"role": "user", "content": "...", "scope": {"type": "document", "file_name": "x.pdf"}},
                {"role": "assistant", "content": "..."}
            ],
            "created_at": "YYYY-MM-DD HH:MM:SS",
            "updated_at": "YYYY-MM-DD HH:MM:SS"
        }
    """

    def __init__(self, chats_dir: str = str(AppConfig.CHATS_DIR)):
        self.chats_dir = chats_dir
        os.makedirs(self.chats_dir, exist_ok=True)
        log.debug(f"ChatManager hazır: {self.chats_dir}")

    # ── Dosya I/O ────────────────────────────────────────────────────────────

    def _chat_path(self, chat_id: str) -> str:
        return os.path.join(self.chats_dir, f"{chat_id}.json")

    def _load_chat(self, chat_id: str) -> Optional[dict]:
        path = self._chat_path(chat_id)
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            log.error(f"Sohbet dosyası okunamadı '{chat_id}': {e}", exc_info=True)
            return None

    def _save_chat(self, chat: dict) -> None:
        path = self._chat_path(chat["id"])
        atomic_write_json(path, chat)

    @staticmethod
    def _now() -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ── CRUD ─────────────────────────────────────────────────────────────────

    def list_chats(self, collection: Optional[str] = None) -> list[dict]:
        """
        Tüm sohbetlerin özetini döner — mesajsız, hızlı liste için.
        collection verilirse o koleksiyona ait olanları filtreler.
        En son güncellenenler önce gelir.
        """
        chats: list[dict] = []
        try:
            entries = os.listdir(self.chats_dir)
        except OSError as e:
            log.error(f"Sohbet dizini okunamadı: {e}", exc_info=True)
            return []

        for fname in entries:
            # .tmp uzantılı dosyalar yarıda kalmış yazma kalıntısı, listeleme
            if not fname.endswith(".json") or fname.endswith(".tmp"):
                continue
            chat_id = fname[:-5]  # ".json" sonekini at
            chat = self._load_chat(chat_id)
            if chat is None:
                continue
            if collection is not None and chat.get("collection") != collection:
                continue
            chats.append(
                {
                    "id": chat["id"],
                    "title": chat.get("title", "İsimsiz sohbet"),
                    "collection": chat.get("collection"),
                    "created_at": chat.get("created_at"),
                    "updated_at": chat.get("updated_at"),
                    "message_count": len(chat.get("messages", [])),
                }
            )

        # En son güncellenen üstte
        chats.sort(key=lambda c: c.get("updated_at") or "", reverse=True)
        return chats

    def get_chat(self, chat_id: str) -> Optional[dict]:
        """Tek bir sohbeti tüm mesajlarıyla döner. Bulunamazsa None."""
        return self._load_chat(chat_id)

    def create_chat(self, collection: str) -> dict:
        """Yeni boş sohbet oluşturur, diske yazar, döner."""
        now = self._now()
        chat = {
            "id": str(uuid.uuid4()),
            "title": "Yeni Sohbet",
            "collection": collection,
            "messages": [],
            "created_at": now,
            "updated_at": now,
        }
        self._save_chat(chat)
        log.info(f"Yeni sohbet oluşturuldu: {chat['id']} (koleksiyon: '{collection}')")
        return chat

    def delete_chat(self, chat_id: str) -> bool:
        """Sohbet dosyasını siler. Yoksa False döner."""
        path = self._chat_path(chat_id)
        if not os.path.exists(path):
            return False
        try:
            os.remove(path)
            log.info(f"Sohbet silindi: {chat_id}")
            return True
        except OSError as e:
            log.error(f"Sohbet silinemedi '{chat_id}': {e}", exc_info=True)
            return False

    def update_title(self, chat_id: str, title: str) -> Optional[dict]:
        """Başlığı günceller. Sohbet yoksa None."""
        chat = self._load_chat(chat_id)
        if chat is None:
            return None
        chat["title"] = title.strip() or "İsimsiz sohbet"
        chat["updated_at"] = self._now()
        self._save_chat(chat)
        return chat

    def rename_collection_in_chats(
        self, old_collection: str, new_collection: str
    ) -> int:
        """
        Belirli bir koleksiyona ait tüm sohbetlerin collection field'ını
        günceller. Koleksiyon yeniden adlandırıldığında DBManager tarafından
        çağrılır — aksi takdirde sohbetler yetim koleksiyon adına referans tutar.

        updated_at değiştirilmez (içerik değişmedi, sadece refactor),
        sidebar sıralaması bozulmasın.

        Dönüs: güncellenen sohbet sayısı.
        """
        count = 0
        try:
            entries = os.listdir(self.chats_dir)
        except OSError as e:
            log.error(f"Sohbet dizini okunamadı: {e}", exc_info=True)
            return 0

        for fname in entries:
            if not fname.endswith(".json") or fname.endswith(".tmp"):
                continue
            chat_id = fname[:-5]
            chat = self._load_chat(chat_id)
            if chat is None:
                continue
            if chat.get("collection") == old_collection:
                chat["collection"] = new_collection
                # updated_at'a dokunma — içerik değişmedi
                self._save_chat(chat)
                count += 1

        if count:
            log.info(
                f"{count} sohbet koleksiyon adı güncellendi: "
                f"'{old_collection}' -> '{new_collection}'"
            )
        return count

    def add_message(self, chat_id: str, message: dict) -> Optional[dict]:
        """
        Sohbete bir mesaj ekler. İlk user mesajında başlık otomatik üretilir
        (kullanıcı mesajının ilk 40 karakteri).
        Sohbet yoksa None döner.
        """
        chat = self._load_chat(chat_id)
        if chat is None:
            return None

        chat["messages"].append(message)
        chat["updated_at"] = self._now()

        # Başlık otomatik üretimi: ilk user mesajından, sadece sohbet hala
        # "Yeni Sohbet" başlıklıysa (kullanıcı manuel değiştirmediyse)
        if (
            message.get("role") == "user"
            and chat.get("title") == "Yeni Sohbet"
        ):
            content = (message.get("content") or "").strip()
            if content:
                title = content[:40]
                if len(content) > 40:
                    title += "..."
                chat["title"] = title

        self._save_chat(chat)
        return chat
