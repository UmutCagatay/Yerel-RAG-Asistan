import json
import logging
import os
import shutil
import tempfile
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from core.config import AppConfig
from core.chat_manager import ChatManager
from core.db_manager import DBManager
from core.ingest_estimator import estimate_total_seconds
from core.logger import setup_logging
from core.query_engine import QueryEngine
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# Logger'i kur — app.py'deki ile aynı kurulum,
# setup_logging idempotent olduğu için tekrar çağrı sorun değil.
setup_logging()
log = logging.getLogger(__name__)


# FastAPI lifespan — startup ve shutdown event'lerini tek context manager'da
# topluyor. on_event() deprecated, modern yol budur.
@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("FastAPI sunucusu başlatıldı.")
    yield
    # Buraya gelirse uvicorn düzgün shutdown yapıyor (Ctrl+C, SIGTERM).
    # Process zaten ölecek ve OS GPU memory'yi serbest bırakacak; bu hook
    # log integrity için: kapanma temiz mi yoksa crash mi anlamak için.
    log.info("FastAPI sunucusu kapanıyor.")


# FastAPI uygulaması — sunucunun "kalbi"
app = FastAPI(title="Yerel RAG Asistani API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    # Sadece Tauri ve Vite dev server origin'lerine izin ver.
    # Backend zaten 127.0.0.1'de dinliyor (uvicorn default), LAN'a kapalı.
    # CORS kısıtlaması ek bir kat: başka bir tarayıcı sekmesindeki kötü
    # site fetch("http://localhost:8000/...") ile veri çekemesin.
    #
    # Regex daha esnek: tauri:// + tauri.localhost (her sema) + localhost +
    # 127.0.0.1 (her port). Liste de geri çekilme için tutuluyor.
    allow_origins=[
        "tauri://localhost",        # Tauri v1 prod
        "http://tauri.localhost",   # Tauri v2 prod (Linux/macOS)
        "https://tauri.localhost",  # Tauri v2 prod (Windows)
        "http://localhost:1420",    # Vite dev server
        "http://127.0.0.1:1420",    # Vite dev server (127 binding)
    ],
    allow_origin_regex=(
        r"^(tauri://localhost"
        r"|https?://tauri\.localhost"
        r"|https?://localhost(:\d+)?"
        r"|https?://127\.0\.0\.1(:\d+)?)$"
    ),
    allow_credentials=True,
    allow_methods=["*"],  # GET, POST, DELETE hepsi
    allow_headers=["*"],
)

# DBManager tek bir tane oluşturulur, sunucu açık olduğu
# sürece yaşar. Terminal'deki App.__init__ içindeki self.db
# ne işe yarıyorsa burada da aynı görev.
db = DBManager()

# ChatManager — sohbet dosyalarını yönetir, data/chats/<id>.json
chat_manager = ChatManager()

# Sorgu lock'u — aynı anda en fazla bir sorgu çalışsın.
# Sebep: 8GB VRAM kullanıcı aynı anda iki sorgu gönderirse iki LLM yan yana
# yüklenmeye çalışır → VRAM çakması. Lock ile ikincisi 409 alıp kullanıcıya
# "birkaç saniye sonra tekrar dene" der.
query_lock = threading.Lock()


@app.get("/health")
def health():
    """Backend ayakta mı sorusuna cevap. Tauri startup'ında ping için.
    Yan bilgi: dosya boyut limiti (MB) + ingest tahmini bayrağı — frontend
    bayrağa göre estimate adımını atlar veya çalıştırır."""
    return {
        "status": "ok",
        "max_file_size_mb": AppConfig.MAX_FILE_SIZE_MB,
        "ingest_estimate_enabled": AppConfig.INGEST_ESTIMATE_ENABLED,
    }


@app.get("/collections")
def list_collections():
    """Tüm koleksiyonları ve hangisinin aktif olduğunu döner."""
    return {
        "active": db.active_collection,
        "all": db.list_collections(),
    }


@app.get("/documents")
def list_documents():
    """Aktif koleksiyondaki dokümanları döner."""
    return {
        "collection": db.active_collection,
        "documents": db.list_documents(),
    }


class CreateCollectionRequest(BaseModel):
    """Yeni koleksiyon oluştururken frontend'in yollayacağı veri şeması."""

    name: str


@app.post("/collections")
def create_collection(body: CreateCollectionRequest):
    """Yeni koleksiyon oluşturur."""
    success = db.create_collection(body.name)
    if not success:
        # DBManager False döndürdü — koleksiyon adı geçersiz veya zaten var.
        # 400 Bad Request: 'istek hatalı, sebebi şu' demek.
        raise HTTPException(
            status_code=400,
            detail=f"'{body.name}' oluşturulamadı. Geçersiz ad veya zaten var.",
        )
    return {"created": body.name}


@app.delete("/collections/{name}")
def delete_collection(name: str):
    """Koleksiyonu ve içindeki tüm dokümanları siler."""
    if name == db.DEFAULT_COLLECTION:
        raise HTTPException(
            status_code=400,
            detail=f"'{db.DEFAULT_COLLECTION}' koleksiyonu silinemez.",
        )

    success = db.delete_collection(name)
    if not success:
        raise HTTPException(
            status_code=404,
            detail=f"'{name}' adında koleksiyon bulunamadı.",
        )
    return {"deleted": name}


class RenameCollectionRequest(BaseModel):
    """Koleksiyon yeniden adlandırma isteği."""

    new_name: str


@app.patch("/collections/{name}")
def rename_collection(name: str, body: RenameCollectionRequest):
    """
    Koleksiyonu yeniden adlandırır. ChromaDB collection rename + chunks
    metadata + sections.json + catalog + aktif state + sohbetler hepsi
    senkron güncellenir.
    """
    result = db.rename_collection(
        name, body.new_name, chat_manager=chat_manager
    )
    if not result.get("renamed"):
        reason = result.get("reason", "Bilinmeyen hata")
        if "zaten var" in reason:
            status = 409
        elif "adında koleksiyon yok" in reason:
            status = 404
        else:
            status = 400
        raise HTTPException(status_code=status, detail=reason)
    return {
        "renamed": result["new_name"],
        "chats_updated": result.get("chats_updated", 0),
    }


@app.post("/collections/{name}/activate")
def set_active_collection(name: str):
    """Aktif koleksiyonu değiştirir."""
    success = db.set_active_collection(name)
    if not success:
        raise HTTPException(
            status_code=404,
            detail=f"'{name}' adında koleksiyon bulunamadı.",
        )
    return {"active": db.active_collection}


class DeleteDocumentsRequest(BaseModel):
    """Silinecek doküman adlarının listesi."""

    file_names: list[str]


@app.delete("/documents")
def delete_documents(body: DeleteDocumentsRequest):
    """Aktif koleksiyondan bir veya birden çok dokümanı siler."""
    if not body.file_names:
        raise HTTPException(
            status_code=400,
            detail="Silinecek doküman listesi boş.",
        )

    result = db.delete_documents(body.file_names)

    # 'failed' boş değilse 207 dönmek REST geleneğinde "kısmi başarı"
    # anlamına gelir. Şu an basit tutuyoruz, sade 200 ile döndürüp
    # detayı body'de veriyoruz; frontend hem 'deleted' hem 'failed'
    # listesini görüp UI'da gösterebilir.
    return result


@app.post("/documents/estimate")
def estimate_documents(
    files: list[UploadFile] = File(...),
    use_vlm: bool = Form(True),
):
    """
    Yüklenecek dosyalar için tahmini ingest süresini hesaplar.

    Dosyaları geçici dizine yazıp PyMuPDF ile sayfa+görsel sayılar, sonra
    AppConfig.INGEST_* sabitleriyle toplam süreyi tahmin eder. Dosyalar
    işlem sonunda silinir; frontend gerçek upload için aynı dosyaları
    /documents'a tekrar gönderir.

    Bayrak (AppConfig.INGEST_ESTIMATE_ENABLED) False ise 404 döner;
    frontend bu durumda estimate adımını atlayıp direkt upload eder.
    """
    if not AppConfig.INGEST_ESTIMATE_ENABLED:
        raise HTTPException(
            status_code=404,
            detail="Ingest tahmini özelliği şu anda devre dışı.",
        )

    if not files:
        raise HTTPException(status_code=400, detail="Dosya gönderilmedi.")

    # Boyut limiti kontrolü — add_documents ile aynı kural; tahmin yapamadığımız
    # dosyalar 'rejected' listesinde bilgi olarak döner.
    tmp_dir = tempfile.mkdtemp(prefix="rag_estimate_")
    try:
        file_tuples: list[tuple[str, Path]] = []
        rejected: list[dict] = []

        for file in files:
            file.file.seek(0, 2)
            size_bytes = file.file.tell()
            file.file.seek(0)
            size_mb = size_bytes / (1024 * 1024)

            if size_mb > AppConfig.MAX_FILE_SIZE_MB:
                rejected.append({
                    "name": file.filename,
                    "reason": (
                        f"Boyut limiti aşıldı ({size_mb:.1f} MB > "
                        f"{AppConfig.MAX_FILE_SIZE_MB} MB)"
                    ),
                })
                continue

            dest = Path(tmp_dir) / file.filename
            with open(dest, "wb") as f:
                shutil.copyfileobj(file.file, f)
            file_tuples.append((file.filename, dest))

        if not file_tuples:
            return {
                "files": [],
                "total_pages": 0,
                "total_images": 0,
                "total_seconds": 0.0,
                "use_vlm": use_vlm,
                "rejected": rejected,
            }

        result = estimate_total_seconds(file_tuples, use_vlm=use_vlm)
        result["rejected"] = rejected
        return result
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@app.post("/documents")
def add_documents(
    files: list[UploadFile] = File(...),
    decisions: str = Form("{}"),
    use_vlm: bool = Form(True),
):
    """
    PDF yükler. Çakışan dosyalar için frontend, /documents/check'ten
    aldığı bilgiyle her dosyanın kararını söyler.

    decisions parametresi JSON string olarak gelir, örnek:
        {"test1.pdf": "overwrite", "test2.pdf": "skip"}
    Çakışmayan dosyaları decisions'a yazmaya gerek yok.

    use_vlm: False ise VLM hiç yüklenmez, görsel içerikler atlanır.
    Hızı önemli olduğundan veya VRAM tasarrufu için kullanıcı kapatabilir.
    """
    if not files:
        raise HTTPException(status_code=400, detail="Dosya gönderilmedi.")

    # Karar dict'ini JSON'dan parse et
    try:
        decisions_dict = json.loads(decisions)
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=400,
            detail="decisions parametresi geçerli JSON değil.",
        )

    # Boyut limiti kontrolü — terminal'deki gibi
    too_big: list[dict] = []
    accepted: list[UploadFile] = []
    for file in files:
        # UploadFile'ın .size özelliği var ama Starlette sürümüne göre
        # her zaman dolu olmayabilir. Güvenli yol: stream'in sonuna gidip pozisyonu ölç.
        file.file.seek(0, 2)  # 2 = dosyanın sonu
        size_bytes = file.file.tell()
        file.file.seek(0)  # başa geri sar, sonra okunacak

        size_mb = size_bytes / (1024 * 1024)
        if size_mb > AppConfig.MAX_FILE_SIZE_MB:
            too_big.append(
                {
                    "file_name": file.filename,
                    "reason": f"Boyut limiti aşıldı ({size_mb:.1f} MB > {AppConfig.MAX_FILE_SIZE_MB} MB)",
                }
            )
        else:
            accepted.append(file)

    if not accepted:
        return {"success": [], "failed": too_big, "skipped": []}

    # Kabul edilenleri geçici dizine yaz
    tmp_dir = tempfile.mkdtemp(prefix="rag_upload_")
    try:
        saved_paths = []
        for file in accepted:
            dest = os.path.join(tmp_dir, file.filename)
            with open(dest, "wb") as f:
                shutil.copyfileobj(file.file, f)
            saved_paths.append(dest)

            # Karar verilmemiş çakışmalar için güvenli default: skip
            # DBManager dict.get(file_name, "ask") yapıyor; dict varsayılanını
            # "skip" yapamayız ama önce string olarak "skip" gönderip dict ile
            # override etmek de mümkün değil. En temizi: çakışan ama kararı
            # olmayan dosyalar için dict'e elle "skip" yazmak.
            if file.filename not in decisions_dict and db.document_exists(
                file.filename
            ):
                decisions_dict[file.filename] = "skip"

        result = db.add_documents(
            file_paths=saved_paths,
            on_conflict=decisions_dict,
            use_vlm=use_vlm,
        )

        # Boyut yüzünden atlananları failed listesine ekle
        result["failed"].extend(too_big)
        return result
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


class CheckDocumentsRequest(BaseModel):
    """Yüklenmeden önce çakışma kontrolü için dosya adları."""

    file_names: list[str]


@app.post("/documents/check")
def check_documents(body: CheckDocumentsRequest):
    """
    Verilen dosya adlarından hangileri aktif koleksiyonda zaten var,
    hangileri yeni — onu söyler. Henüz hiçbir şey yüklenmez.
    """
    existing: list[str] = []
    new: list[str] = []
    for name in body.file_names:
        if db.document_exists(name):
            existing.append(name)
        else:
            new.append(name)
    return {"existing": existing, "new": new}


class RenameDocumentRequest(BaseModel):
    """Doküman yeniden adlandırma isteği."""

    old_name: str
    new_name: str


@app.post("/documents/rename")
def rename_document(body: RenameDocumentRequest):
    """
    Aktif koleksiyondaki bir dokümanı yeniden adlandırır.
    Chunk'lar ve embedding'ler yerinde kalır — sadece metadata güncellenir.
    """
    result = db.rename_document(body.old_name, body.new_name)
    if not result.get("renamed"):
        reason = result.get("reason", "Bilinmeyen hata")
        # Çakışma durumu için 409, diğerleri için 400
        status = 409 if "zaten var" in reason else 400
        raise HTTPException(status_code=status, detail=reason)
    return {"renamed": result["new_name"]}


class MoveDocumentRequest(BaseModel):
    """Doküman taşıma isteği: aktif koleksiyondan başka koleksiyona."""

    file_name: str
    target_collection: str


@app.post("/documents/move")
def move_document(body: MoveDocumentRequest):
    """
    Bir dokümanı aktif koleksiyondan hedef koleksiyona taşır.
    Embedding yeniden hesaplanmaz — chunks aynı vektörlerle target'a kopyalanıp
    source'tan silinir.
    """
    result = db.move_document(body.file_name, body.target_collection)
    if not result.get("moved"):
        reason = result.get("reason", "Bilinmeyen hata")
        if "zaten var" in reason:
            status = 409
        elif "içinde yok" in reason or "adında koleksiyon yok" in reason:
            status = 404
        else:
            status = 400
        raise HTTPException(status_code=status, detail=reason)
    return {
        "moved": result["file_name"],
        "target": result["target"],
        "chunks": result.get("chunks", 0),
    }


class ReorderDocumentsRequest(BaseModel):
    """Aktif koleksiyonun yeni doküman sırası."""

    file_names: list[str]


@app.post("/documents/reorder")
def reorder_documents(body: ReorderDocumentsRequest):
    """
    Aktif koleksiyonun doküman sırasını verilen listeye göre günceller.
    Sadece catalog dict sırası değişir; ChromaDB/sections'a dokunulmaz.
    """
    result = db.reorder_documents(body.file_names)
    if not result.get("reordered"):
        reason = result.get("reason", "Bilinmeyen hata")
        status = 400
        if "koleksiyon yok" in reason:
            status = 404
        raise HTTPException(status_code=status, detail=reason)
    return {"reordered": True, "count": result["count"]}


class QueryRequest(BaseModel):
    """Sorgu için gerekli bilgiler. file_name=None ise tüm koleksiyon kapsamı."""

    question: str
    file_name: str | None = None


@app.post("/query")
def query(body: QueryRequest):
    """
    Aktif koleksiyondaki belirli bir doküman üzerinde sorgu çalıştırır.
    Modeller yüklenip cevap üretildiği için uzun sürebilir.

    Eşzamanlılık: aynı anda tek bir sorgu çalışır (VRAM koruması).
    İkinci istek gelirse 409 Conflict döner.
    """
    # Boş soru kontrolü — Pydantic str doğrulamasını geçer ama anlamsız iş.
    if not body.question.strip():
        raise HTTPException(status_code=400, detail="Soru boş olamaz.")

    # Doküman adı verilmişse var olduğunu doğrula. file_name=None ise
    # tüm aktif koleksiyon kapsamında arama yapılacak — doğrulama gerekmez.
    if body.file_name is not None and not db.document_exists(body.file_name):
        raise HTTPException(
            status_code=404,
            detail=f"'{body.file_name}' aktif koleksiyonda bulunamadı.",
        )

    # Lock'u dene — şu an başka sorgu çalışıyorsa hemen 409 dön.
    if not query_lock.acquire(blocking=False):
        raise HTTPException(
            status_code=409,
            detail="Sunucu şu an başka bir sorgu işliyor. Lütfen birkaç saniye bekleyip tekrar deneyin.",
        )

    try:
        engine = QueryEngine(collection_name=db.active_collection)
        answer = engine.run(question=body.question, file_name=body.file_name)
        return {"answer": answer}
    finally:
        query_lock.release()
        log.info("Sorgu lock'u serbest bırakıldı (/query).")


@app.post("/query/stream")
def query_stream(body: QueryRequest):
    """
    Sorguyu çalıştırır, cevabı token token akıtır.
    Frontend ReadableStream ile parçaları okur ve ekrana yazar.

    Eşzamanlılık: aynı anda tek bir sorgu çalışır (VRAM koruması).
    İkinci istek gelirse 409 Conflict döner.
    """
    if not body.question.strip():
        raise HTTPException(status_code=400, detail="Soru boş olamaz.")

    # file_name=None → tüm koleksiyon kapsamı. Dolu ise doküman var mı bak.
    if body.file_name is not None and not db.document_exists(body.file_name):
        raise HTTPException(
            status_code=404,
            detail=f"'{body.file_name}' aktif koleksiyonda bulunamadı.",
        )

    # Lock'u dene — başka bir streaming devam ediyorsa hemen 409 dön.
    if not query_lock.acquire(blocking=False):
        raise HTTPException(
            status_code=409,
            detail="Sunucu şu an başka bir sorgu işliyor. Lütfen birkaç saniye bekleyip tekrar deneyin.",
        )

    def stream_with_lock():
        """
        Generator: stream yarıda kesilse de (GeneratorExit) finally ile
        lock garantili serbest kalır. with QueryEngine bloku da ayrıca
        modellerin unload edilmesini sağlar.
        """
        try:
            engine = QueryEngine(collection_name=db.active_collection)
            for chunk in engine.run_stream(
                question=body.question,
                file_name=body.file_name,
            ):
                yield chunk
        finally:
            query_lock.release()
            log.info("Sorgu lock'u serbest bırakıldı (/query/stream).")

    return StreamingResponse(
        stream_with_lock(),
        media_type="text/plain; charset=utf-8",
    )


# ── Sohbet endpoint'leri ─────────────────────────────────────────────────


@app.get("/chats")
def list_chats(collection: str | None = None):
    """
    Sohbet listesini döner (mesajsız özet, hızlı sidebar render için).
    collection verilirse o koleksiyondaki sohbetleri filtreler.
    """
    return {"chats": chat_manager.list_chats(collection=collection)}


@app.get("/chats/{chat_id}")
def get_chat(chat_id: str):
    """Tek bir sohbeti tüm mesajlarıyla döner."""
    chat = chat_manager.get_chat(chat_id)
    if chat is None:
        raise HTTPException(
            status_code=404,
            detail=f"Sohbet bulunamadı: {chat_id}",
        )
    return chat


class CreateChatRequest(BaseModel):
    """Yeni sohbet açılırken hangi koleksiyona bağlanacağı."""

    collection: str


@app.post("/chats")
def create_chat(body: CreateChatRequest):
    """Yeni boş sohbet açar, koleksiyona kilitler."""
    # Koleksiyonun var olduğunu doğrula — olmayan koleksiyona sohbet açmak,
    # sonra orphan sohbet oluşturmak demektir, hiç yaratılmasın.
    if body.collection not in db.list_collections():
        raise HTTPException(
            status_code=400,
            detail=f"'{body.collection}' adında koleksiyon yok.",
        )
    return chat_manager.create_chat(body.collection)


@app.delete("/chats/{chat_id}")
def delete_chat(chat_id: str):
    """Sohbeti kalıcı olarak siler."""
    success = chat_manager.delete_chat(chat_id)
    if not success:
        raise HTTPException(
            status_code=404,
            detail=f"Sohbet bulunamadı: {chat_id}",
        )
    return {"deleted": chat_id}


class UpdateChatTitleRequest(BaseModel):
    """Başlık güncelleme için."""

    title: str


@app.patch("/chats/{chat_id}")
def update_chat_title(chat_id: str, body: UpdateChatTitleRequest):
    """Sadece sohbet başlığını günceller."""
    chat = chat_manager.update_title(chat_id, body.title)
    if chat is None:
        raise HTTPException(
            status_code=404,
            detail=f"Sohbet bulunamadı: {chat_id}",
        )
    return chat


class MessageScope(BaseModel):
    """
    Bir user mesajının hangi kapsamda sorulduğunu temsil eder.
    type='document' ise file_name dolu olmalı.
    type='collection' ise tüm aktif koleksiyon kapsamı demek, file_name None.
    """

    type: str  # "document" | "collection"
    file_name: str | None = None


class AddMessageRequest(BaseModel):
    """Sohbete eklenecek tek bir mesaj."""

    role: str  # "user" | "assistant"
    content: str
    scope: MessageScope | None = None


@app.post("/chats/{chat_id}/messages")
def add_message(chat_id: str, body: AddMessageRequest):
    """
    Sohbete bir mesaj ekler. Frontend her user mesajı ve her assistant
    cevabı (stream bittikten sonra) için bunu çağırır.
    İlk user mesajında ChatManager başlığı otomatik üretir.
    """
    if body.role not in ("user", "assistant"):
        raise HTTPException(
            status_code=400,
            detail="role 'user' veya 'assistant' olmalı.",
        )

    message: dict = {"role": body.role, "content": body.content}
    if body.scope is not None:
        message["scope"] = body.scope.model_dump()

    chat = chat_manager.add_message(chat_id, message)
    if chat is None:
        raise HTTPException(
            status_code=404,
            detail=f"Sohbet bulunamadı: {chat_id}",
        )
    return chat
