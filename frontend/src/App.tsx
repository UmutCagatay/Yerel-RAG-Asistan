import { useState, useEffect, useRef } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import "./App.css";

const API = "http://localhost:8000";

type Doc = {
  file_name: string;
  chunk_count: number;
  section_count: number;
  added_at: string;
};

type MessageScope = {
  type: "document" | "collection";
  file_name?: string | null;
};

type Message = {
  role: "user" | "assistant";
  content: string;
  scope?: MessageScope;
};

type ChatSummary = {
  id: string;
  title: string;
  collection: string;
  created_at: string;
  updated_at: string;
  message_count: number;
};

type ToastMsg = {
  type: "success" | "error" | "info";
  message: string;
};

// <think>...</think> bloklarını metin parçalarından ayırır.
// Akış sırasında blok kapanmamış olabilir; o zaman 'complete: false' işaretlenir.
function parseThinkBlocks(content: string) {
  const regex = /<think>([\s\S]*?)(?:<\/think>|$)/g;
  const parts: Array<{
    type: "think" | "text";
    content: string;
    complete?: boolean;
  }> = [];
  let lastEnd = 0;
  let match: RegExpExecArray | null;
  while ((match = regex.exec(content)) !== null) {
    if (match.index > lastEnd) {
      parts.push({
        type: "text",
        content: content.slice(lastEnd, match.index),
      });
    }
    const fullMatch = content.slice(match.index, match.index + match[0].length);
    parts.push({
      type: "think",
      content: match[1],
      complete: fullMatch.endsWith("</think>"),
    });
    lastEnd = match.index + match[0].length;
  }
  if (lastEnd < content.length) {
    parts.push({ type: "text", content: content.slice(lastEnd) });
  }
  return parts;
}

// Asistan mesajını markdown olarak render eder; <think> bloklarını
// açılır-kapanır olarak gösterir.
function AssistantMessage({ content }: { content: string }) {
  const parts = parseThinkBlocks(content);
  return (
    <div className="space-y-2">
      {parts.map((part, i) => {
        if (part.type === "think") {
          return (
            <details key={i} className="text-xs text-slate-500">
              <summary className="cursor-pointer hover:text-slate-700 select-none">
                {part.complete ? "Düşünme sürecini gör" : "Düşünüyor..."}
              </summary>
              <div className="mt-1 pl-3 border-l-2 border-slate-300 whitespace-pre-wrap text-slate-600">
                {part.content.trim()}
              </div>
            </details>
          );
        }
        return (
          <div key={i} className="prose prose-sm max-w-none">
            <ReactMarkdown remarkPlugins={[remarkGfm]}>
              {part.content}
            </ReactMarkdown>
          </div>
        );
      })}
    </div>
  );
}

// Saniyeyi "X dk Y sn" / "Y sn" formatlar. Geçen süre gösteriminde kullanılır.
// 60 sn altında "Y sn", üstünde "X dk Y sn". Math.floor saniyeyi tam sayıya
// indirir, kullanıcı "57.4 sn" yerine "57 sn" görür.
function formatDuration(seconds: number): string {
  const s = Math.max(0, Math.floor(seconds));
  if (s < 60) return `${s} sn`;
  const m = Math.floor(s / 60);
  const rem = s % 60;
  return rem === 0 ? `${m} dk` : `${m} dk ${rem} sn`;
}

// User mesajının yanında "hangi kapsamda soruldu" rozeti.
// type=document → 📄 <file_name>, type=collection → 📁 koleksiyon
function ScopeBadge({ scope }: { scope?: MessageScope }) {
  if (!scope) return null;
  if (scope.type === "document" && scope.file_name) {
    return (
      <div className="text-xs text-slate-400 mt-1 text-right">
        📄 {scope.file_name}
      </div>
    );
  }
  if (scope.type === "collection") {
    return (
      <div className="text-xs text-slate-400 mt-1 text-right">
        📁 tüm koleksiyon
      </div>
    );
  }
  return null;
}

function App() {
  const [docs, setDocs] = useState<Doc[]>([]);
  const [collection, setCollection] = useState<string>("");
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedDoc, setSelectedDoc] = useState<string | null>(null);

  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState<string>("");
  const [isStreaming, setIsStreaming] = useState<boolean>(false);

  // Sohbet state'leri
  const [chats, setChats] = useState<ChatSummary[]>([]);
  const [activeChatId, setActiveChatId] = useState<string | null>(null);
  const [editingTitleId, setEditingTitleId] = useState<string | null>(null);
  const [editingTitleValue, setEditingTitleValue] = useState<string>("");

  const [isUploading, setIsUploading] = useState<boolean>(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const [pendingFiles, setPendingFiles] = useState<File[]>([]);
  const [conflicts, setConflicts] = useState<string[]>([]);
  const [decisions, setDecisions] = useState<
    Record<string, "overwrite" | "skip">
  >({});

  const [deletingDoc, setDeletingDoc] = useState<string | null>(null);
  const [editingDocName, setEditingDocName] = useState<string | null>(null);
  const [editingDocValue, setEditingDocValue] = useState<string>("");
  // VLM (görsel okuma) toggle — default açık.
  // Kapalıyken upload hızlı (~5-10sn VLM yükleme + görsel başı ~10sn atlanır).
  const [useVlm, setUseVlm] = useState<boolean>(true);

  // Koleksiyon yeniden adlandırma state'leri
  const [editingColName, setEditingColName] = useState<string | null>(null);
  const [editingColValue, setEditingColValue] = useState<string>("");

  const [allCollections, setAllCollections] = useState<string[]>([]);
  const [showCollectionMenu, setShowCollectionMenu] = useState<boolean>(false);
  const collectionMenuRef = useRef<HTMLDivElement>(null);

  const [maxFileSizeMb, setMaxFileSizeMb] = useState<number>(50);
  // İşlenmekte olan doküman sayısı — upload modalında "N doküman işleniyor"
  // yazısı için. Backend tek seferde tüm batch'i işleyip döndüğü için canlı
  // "x/N" ilerleme yok; toplam sayı + geçen süre gösterilir.
  const [uploadCount, setUploadCount] = useState<number>(0);
  const [uploadStartMs, setUploadStartMs] = useState<number | null>(null);
  const [uploadElapsedSec, setUploadElapsedSec] = useState<number>(0);
  const [toast, setToast] = useState<ToastMsg | null>(null);

  // Onay kutusu — native confirm() yerine kendi modal'ımız.
  // Promise tabanlı: askConfirm(...) çağrılır, kullanıcı butona basınca
  // resolve edilir. Böylece "localhost:1420 şunu diyor" öneki kalmaz.
  const [confirmState, setConfirmState] = useState<{
    message: string;
    confirmLabel: string;
    resolve: (ok: boolean) => void;
  } | null>(null);

  function askConfirm(message: string, confirmLabel = "Sil"): Promise<boolean> {
    return new Promise((resolve) => {
      setConfirmState({ message, confirmLabel, resolve });
    });
  }

  function resolveConfirm(ok: boolean) {
    if (confirmState) confirmState.resolve(ok);
    setConfirmState(null);
  }

  // Metin girişli onay kutusu — native prompt() yerine. Yine Promise tabanlı;
  // resolve(değer) ya da iptalde resolve(null).
  const [promptState, setPromptState] = useState<{
    message: string;
    value: string;
    submitLabel: string;
    resolve: (value: string | null) => void;
  } | null>(null);

  function askPrompt(
    message: string,
    defaultValue = "",
    submitLabel = "Tamam",
  ): Promise<string | null> {
    return new Promise((resolve) => {
      setPromptState({ message, value: defaultValue, submitLabel, resolve });
    });
  }

  function resolvePrompt(value: string | null) {
    if (promptState) promptState.resolve(value);
    setPromptState(null);
  }

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const messagesContainerRef = useRef<HTMLDivElement>(null);
  const [autoScroll, setAutoScroll] = useState<boolean>(true);

  // Aktif sohbet başlığı (header'da göstermek için)
  const activeChat = chats.find((c) => c.id === activeChatId);

  // ── Initial fetch ──────────────────────────────────────────────────────
  async function fetchInitial() {
    setLoading(true);
    setError(null);
    try {
      const [docsData, colData, healthData] = await Promise.all([
        fetch(`${API}/documents`).then((r) => r.json()),
        fetch(`${API}/collections`).then((r) => r.json()),
        fetch(`${API}/health`).then((r) => r.json()),
      ]);
      setDocs(docsData.documents);
      setCollection(docsData.collection);
      setAllCollections(colData.all);
      if (typeof healthData.max_file_size_mb === "number") {
        setMaxFileSizeMb(healthData.max_file_size_mb);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Bağlantı hatası");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    fetchInitial();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Aktif koleksiyon değişince sohbet listesi de yenilensin
  useEffect(() => {
    if (!collection) return;
    refreshChats();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [collection]);

  // Dropdown dışına tıklayınca kapansın
  useEffect(() => {
    if (!showCollectionMenu) return;
    function onMouseDown(e: MouseEvent) {
      if (
        collectionMenuRef.current &&
        !collectionMenuRef.current.contains(e.target as Node)
      ) {
        setShowCollectionMenu(false);
      }
    }
    document.addEventListener("mousedown", onMouseDown);
    return () => document.removeEventListener("mousedown", onMouseDown);
  }, [showCollectionMenu]);

  // Dropdown her kapandığında rename edit modunu da temizle.
  // Aksi takdirde kullanıcı edit moduna geçip dışarı tıklaç dropdown'ı
  // tekrar açtığında hala input ile karşılaşır (input DOM'dan kalktığı için
  // onBlur tetiklenmiyor).
  useEffect(() => {
    if (!showCollectionMenu) {
      setEditingColName(null);
      setEditingColValue("");
    }
  }, [showCollectionMenu]);

  // Toast'u belirli süre sonra otomatik kapat. Önemli mesajlar (error/info)
  // için daha uzun dur — örneğin failed dosya listesi okunmalı.
  useEffect(() => {
    if (!toast) return;
    const dur = toast.type === "success" ? 4000 : 8000;
    const t = setTimeout(() => setToast(null), dur);
    return () => clearTimeout(t);
  }, [toast]);

  // Upload modali açıkken geçen saniyeyi güncelle. setInterval 1s, başlangıç
  // anı uploadStartMs'te tutulur (Date.now() ile delta). isUploading false
  // olunca interval temizlenir; bir sonraki upload'da sıfırdan başlar.
  useEffect(() => {
    if (!isUploading || uploadStartMs === null) return;
    const t = setInterval(() => {
      setUploadElapsedSec(Math.floor((Date.now() - uploadStartMs) / 1000));
    }, 1000);
    return () => clearInterval(t);
  }, [isUploading, uploadStartMs]);

  // Mesaj geldikçe otomatik en alta kay — ama kullanıcı yukarı kaydırdıysa karma.
  useEffect(() => {
    if (autoScroll) {
      messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }
  }, [messages, autoScroll]);

  // Kullanıcı scroll edince autoScroll'u güncelle.
  function handleMessagesScroll() {
    const el = messagesContainerRef.current;
    if (!el) return;
    const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 30;
    setAutoScroll(atBottom);
  }

  // ── Sohbet işlemleri ────────────────────────────────────────────────────

  async function refreshChats() {
    try {
      const r = await fetch(
        `${API}/chats?collection=${encodeURIComponent(collection)}`,
      );
      if (!r.ok) return;
      const data = await r.json();
      setChats(data.chats ?? []);
    } catch (err) {
      console.error("Sohbet listesi yenilenemedi:", err);
    }
  }

  function handleNewChat() {
    // Lazy: hemen backend'e yaratmıyoruz, ilk mesajda yaratılacak.
    setActiveChatId(null);
    setMessages([]);
    setAutoScroll(true);
  }

  async function handleSelectChat(chatId: string) {
    if (chatId === activeChatId) return;
    try {
      const r = await fetch(`${API}/chats/${chatId}`);
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const chat = await r.json();
      setActiveChatId(chatId);
      setMessages(chat.messages ?? []);
      setAutoScroll(true);
    } catch (err) {
      setToast({ type: "error", message: "Sohbet yüklenemedi." });
      console.error(err);
    }
  }

  async function handleDeleteChat(chatId: string) {
    if (!(await askConfirm("Bu sohbet silinecek. Emin misin?"))) return;
    try {
      const r = await fetch(`${API}/chats/${chatId}`, { method: "DELETE" });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      setChats((prev) => prev.filter((c) => c.id !== chatId));
      if (activeChatId === chatId) {
        setActiveChatId(null);
        setMessages([]);
      }
    } catch (err) {
      setToast({ type: "error", message: "Sohbet silinemedi." });
      console.error(err);
    }
  }

  function handleStartRename(chat: ChatSummary) {
    setEditingTitleId(chat.id);
    setEditingTitleValue(chat.title);
  }

  async function handleSubmitRename() {
    if (!editingTitleId) return;
    const title = editingTitleValue.trim();
    const id = editingTitleId;
    setEditingTitleId(null);
    if (!title) return;
    try {
      const r = await fetch(`${API}/chats/${id}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ title }),
      });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      setChats((prev) =>
        prev.map((c) => (c.id === id ? { ...c, title } : c)),
      );
    } catch (err) {
      setToast({ type: "error", message: "Başlık güncellenemedi." });
      console.error(err);
    }
  }

  function handleCancelRename() {
    setEditingTitleId(null);
    setEditingTitleValue("");
  }

  // ── Mesaj gönderme ──────────────────────────────────────────────────────

  async function handleSend() {
    if (!input.trim() || isStreaming) return;

    const question = input.trim();
    // Bu mesajın kapsamı: doküman seçiliyse o doküman, değilse tüm koleksiyon
    const scope: MessageScope = selectedDoc
      ? { type: "document", file_name: selectedDoc }
      : { type: "collection" };

    setInput("");
    setIsStreaming(true);

    // Optimistic: UI'a hemen ekle
    setMessages((prev) => [
      ...prev,
      { role: "user", content: question, scope },
      { role: "assistant", content: "" },
    ]);

    let assistantContent = "";
    let shouldPersist = false;

    try {
      const response = await fetch(`${API}/query/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question,
          file_name: selectedDoc, // null gönderebilir, backend tüm koleksiyon arar
        }),
      });

      if (response.status === 409) {
        // Sunucu meşgul — UI'dan optimistic mesajları geri al, kaydetme.
        setMessages((prev) => prev.slice(0, -2));
        setToast({
          type: "info",
          message:
            "Sunucu şu an başka bir sorgu işliyor. Birkaç saniye bekleyip tekrar deneyin.",
        });
        return;
      }

      if (!response.ok || !response.body) {
        throw new Error(`HTTP ${response.status}`);
      }

      shouldPersist = true;
      const reader = response.body.getReader();
      const decoder = new TextDecoder();

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        const chunk = decoder.decode(value, { stream: true });
        assistantContent += chunk;
        setMessages((prev) => {
          const updated = [...prev];
          updated[updated.length - 1] = {
            ...updated[updated.length - 1],
            content: updated[updated.length - 1].content + chunk,
          };
          return updated;
        });
      }
    } catch (err) {
      // Network/HTTP hatası — UI'da hata mesajı, yine kaydet ki kullanıcı geriye dönünce görsün
      assistantContent = `[Hata: ${err instanceof Error ? err.message : "bilinmeyen"}]`;
      setMessages((prev) => {
        const updated = [...prev];
        updated[updated.length - 1] = {
          role: "assistant",
          content: assistantContent,
        };
        return updated;
      });
      shouldPersist = true;
    } finally {
      setIsStreaming(false);
    }

    // Backend'e kaydet (lazy chat creation + 2 mesaj)
    if (shouldPersist) {
      try {
        let chatId = activeChatId;
        if (chatId === null) {
          const r = await fetch(`${API}/chats`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ collection }),
          });
          if (!r.ok) throw new Error("Sohbet oluşturulamadı");
          const newChat = await r.json();
          chatId = newChat.id;
          setActiveChatId(chatId);
        }

        await fetch(`${API}/chats/${chatId}/messages`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ role: "user", content: question, scope }),
        });
        await fetch(`${API}/chats/${chatId}/messages`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            role: "assistant",
            content: assistantContent,
          }),
        });

        // Sıralamayı ve başlığı güncellemek için listeyi yenile
        refreshChats();
      } catch (err) {
        console.warn("Sohbet kaydedilemedi:", err);
      }
    }
  }

  // ── Doküman işlemleri ───────────────────────────────────────────────────

  async function refreshDocs() {
    try {
      const res = await fetch(`${API}/documents`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      setDocs(data.documents);
    } catch (err) {
      console.error("Liste yenilenemedi:", err);
    }
  }

  // Doküman yeniden adlandırma — sohbet rename ile aynı pattern.
  // Chunk'lar ve embedding'ler yerinde kalır, sadece metadata güncellenir.
  function handleStartRenameDoc(fileName: string) {
    setEditingDocName(fileName);
    setEditingDocValue(fileName);
  }

  async function handleSubmitRenameDoc() {
    if (!editingDocName) return;
    const oldName = editingDocName;
    const newName = editingDocValue.trim();
    setEditingDocName(null);
    setEditingDocValue("");

    if (!newName || newName === oldName) return;

    try {
      const res = await fetch(`${API}/documents/rename`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ old_name: oldName, new_name: newName }),
      });

      if (res.status === 409) {
        const err = await res.json().catch(() => null);
        setToast({
          type: "error",
          message: err?.detail || `'${newName}' zaten var.`,
        });
        return;
      }
      if (!res.ok) {
        const err = await res.json().catch(() => null);
        throw new Error(err?.detail || `HTTP ${res.status}`);
      }

      await refreshDocs();
      // Seçili doküman bu ise yeni isme güncelle ki seimme bozulmasın.
      // Eski mesajlardaki scope.file_name dokunulmaz (geriye dönük snapshot).
      if (selectedDoc === oldName) {
        setSelectedDoc(newName);
      }
      setToast({
        type: "success",
        message: `'${oldName}' -> '${newName}'`,
      });
    } catch (err) {
      setToast({
        type: "error",
        message: `Yeniden adlandırılamadı: ${err instanceof Error ? err.message : "bilinmeyen"}`,
      });
    }
  }

  function handleCancelRenameDoc() {
    setEditingDocName(null);
    setEditingDocValue("");
  }

  // Doküman sıralama — frontend liste swap'i + backend'e tüm sıra gönderim.
  // Optimistic update: önce UI'da swap, sonra backend'e gönder. Hata olursa
  // eski sıraya rollback.
  async function handleReorderDoc(index: number, direction: -1 | 1) {
    const newIndex = index + direction;
    if (newIndex < 0 || newIndex >= docs.length) return;

    const previousDocs = docs;
    const newDocs = [...docs];
    [newDocs[index], newDocs[newIndex]] = [newDocs[newIndex], newDocs[index]];
    setDocs(newDocs);

    try {
      const res = await fetch(`${API}/documents/reorder`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          file_names: newDocs.map((d) => d.file_name),
        }),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => null);
        throw new Error(err?.detail || `HTTP ${res.status}`);
      }
    } catch (err) {
      // Rollback eski sıraya
      setDocs(previousDocs);
      setToast({
        type: "error",
        message: `Sıralama güncellenemedi: ${err instanceof Error ? err.message : "bilinmeyen"}`,
      });
    }
  }

  async function handleDeleteDoc(fileName: string) {
    if (!(await askConfirm(`'${fileName}' silinecek. Emin misin?`))) return;

    setDeletingDoc(fileName);
    try {
      const res = await fetch(`${API}/documents`, {
        method: "DELETE",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ file_names: [fileName] }),
      });

      if (!res.ok) throw new Error(`HTTP ${res.status}`);

      const result = await res.json();
      if (result.failed && result.failed.length > 0) {
        setToast({
          type: "error",
          message: `Silinemedi: ${result.failed[0].reason}`,
        });
      } else {
        // Silinen dosya seçiliyse seçimi kaldır — ama aktif sohbetin
        // mesajlarına dokunma, onlar geriye dönük görüntüleme için kalır.
        if (selectedDoc === fileName) {
          setSelectedDoc(null);
        }
        await refreshDocs();
      }
    } catch (err) {
      setToast({
        type: "error",
        message: `Silme hatası: ${err instanceof Error ? err.message : "bilinmeyen"}`,
      });
    } finally {
      setDeletingDoc(null);
    }
  }

  // ── Koleksiyon işlemleri ────────────────────────────────────────────────

  async function handleSwitchCollection(name: string) {
    if (name === collection) {
      setShowCollectionMenu(false);
      return;
    }

    try {
      const res = await fetch(`${API}/collections/${name}/activate`, {
        method: "POST",
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);

      const docsRes = await fetch(`${API}/documents`);
      const docsData = await docsRes.json();

      setCollection(name);
      setDocs(docsData.documents);
      setSelectedDoc(null);
      // Sohbet bir koleksiyona kilitli — koleksiyon değişti, aktif sohbeti bırak
      setActiveChatId(null);
      setMessages([]);
      setShowCollectionMenu(false);
      // chats listesi useEffect ile collection değiştiği için otomatik yenilenir
    } catch (err) {
      setToast({
        type: "error",
        message: `Koleksiyon değiştirilemedi: ${err instanceof Error ? err.message : "bilinmeyen"}`,
      });
    }
  }

  async function handleCreateCollection() {
    const name = await askPrompt("Yeni koleksiyon adı:", "", "Oluştur");
    if (!name || !name.trim()) return;

    try {
      const res = await fetch(`${API}/collections`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: name.trim() }),
      });

      if (!res.ok) {
        const errBody = await res.json().catch(() => null);
        throw new Error(errBody?.detail || `HTTP ${res.status}`);
      }

      const data = await res.json();
      setAllCollections((prev) => [...prev, data.created]);
      setShowCollectionMenu(false);
    } catch (err) {
      setToast({
        type: "error",
        message: `Koleksiyon oluşturulamadı: ${err instanceof Error ? err.message : "bilinmeyen"}`,
      });
    }
  }

  async function handleDeleteCollection(name: string) {
    if (
      !(await askConfirm(
        `'${name}' koleksiyonu, içindeki tüm dokümanlar ve sohbetler silinecek. Emin misin?`,
      ))
    )
      return;

    try {
      const res = await fetch(`${API}/collections/${name}`, {
        method: "DELETE",
      });

      if (!res.ok) {
        const errBody = await res.json().catch(() => null);
        throw new Error(errBody?.detail || `HTTP ${res.status}`);
      }

      const wasActive = name === collection;
      setAllCollections((prev) => prev.filter((c) => c !== name));

      if (wasActive) {
        const docsRes = await fetch(`${API}/documents`);
        const docsData = await docsRes.json();
        setCollection(docsData.collection);
        setDocs(docsData.documents);
        setSelectedDoc(null);
        setActiveChatId(null);
        setMessages([]);
      }
    } catch (err) {
      setToast({
        type: "error",
        message: `Koleksiyon silinemedi: ${err instanceof Error ? err.message : "bilinmeyen"}`,
      });
    }
  }

  // ── Yükleme işlemleri ───────────────────────────────────────────────────

  // Koleksiyon yeniden adlandırma — dropdown li'lerinde ✏ butonuyla başlatılır.
  // Dropdown'ın dblclick ile çatışmasını önlemek için ayrı buton tercih edildi
  // (single click handleSwitchCollection'ı tetikler ve menüyü kapatır).
  // 'default' yeniden adlandırılamaz — backend de reddedecek ama UI'da
  // baştan buton gizli olmalı.
  function handleStartRenameCol(name: string) {
    if (name === "default") return;
    setEditingColName(name);
    setEditingColValue(name);
  }

  async function handleSubmitRenameCol() {
    if (!editingColName) return;
    const oldName = editingColName;
    const newName = editingColValue.trim();
    setEditingColName(null);
    setEditingColValue("");

    if (!newName || newName === oldName) return;

    try {
      const res = await fetch(
        `${API}/collections/${encodeURIComponent(oldName)}`,
        {
          method: "PATCH",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ new_name: newName }),
        },
      );

      if (res.status === 409) {
        const err = await res.json().catch(() => null);
        setToast({
          type: "error",
          message: err?.detail || `'${newName}' zaten var.`,
        });
        return;
      }
      if (!res.ok) {
        const err = await res.json().catch(() => null);
        throw new Error(err?.detail || `HTTP ${res.status}`);
      }

      // Listede yeni isime güncelle
      setAllCollections((prev) =>
        prev.map((c) => (c === oldName ? newName : c)),
      );
      // Aktif koleksiyon yeniden adlandırıldıysa state'i güncelle.
      // Backend zaten sections + chunks metadata + sohbetler hepsini
      // senkron güncelledi; collection state değişince useEffect chats'i
      // yenileyecek.
      if (collection === oldName) {
        setCollection(newName);
      }

      setToast({
        type: "success",
        message: `Koleksiyon: '${oldName}' -> '${newName}'`,
      });
    } catch (err) {
      setToast({
        type: "error",
        message: `Yeniden adlandırılamadı: ${err instanceof Error ? err.message : "bilinmeyen"}`,
      });
    }
  }

  function handleCancelRenameCol() {
    setEditingColName(null);
    setEditingColValue("");
  }

  async function handleUpload(files: FileList) {
    if (files.length === 0) return;

    // Boyut ön kontrolü — backend de yapacak ama kullanıcıyı erken uyaralım
    const allFiles = Array.from(files);
    const tooBig: string[] = [];
    const okFiles: File[] = [];
    for (const f of allFiles) {
      const sizeMb = f.size / (1024 * 1024);
      if (sizeMb > maxFileSizeMb) {
        tooBig.push(`${f.name} (${sizeMb.toFixed(1)} MB)`);
      } else {
        okFiles.push(f);
      }
    }

    if (tooBig.length > 0) {
      setToast({
        type: "error",
        message: `Boyut limiti aşıldı (${maxFileSizeMb} MB): ${tooBig.join(", ")}`,
      });
    }

    if (okFiles.length === 0) {
      if (fileInputRef.current) fileInputRef.current.value = "";
      return;
    }

    const fileArr = okFiles;

    try {
      const checkRes = await fetch(`${API}/documents/check`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ file_names: fileArr.map((f) => f.name) }),
      });

      if (!checkRes.ok) throw new Error(`HTTP ${checkRes.status}`);

      const checkData = await checkRes.json();

      if (checkData.existing.length === 0) {
        await doUpload(fileArr, {});
      } else {
        setPendingFiles(fileArr);
        setConflicts(checkData.existing);
        const initial: Record<string, "overwrite" | "skip"> = {};
        for (const name of checkData.existing) {
          initial[name] = "skip";
        }
        setDecisions(initial);
      }
    } catch (err) {
      setToast({
        type: "error",
        message: `Kontrol hatası: ${err instanceof Error ? err.message : "bilinmeyen"}`,
      });
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  }

  async function doUpload(
    files: File[],
    decisionsMap: Record<string, "overwrite" | "skip">,
  ) {
    setPendingFiles([]);
    setConflicts([]);
    setDecisions({});

    // Upload modalı için: kaç doküman işleniyor + geçen süre sayacı başlat.
    setUploadCount(files.length);
    setUploadStartMs(Date.now());
    setUploadElapsedSec(0);

    setIsUploading(true);
    try {
      const formData = new FormData();
      for (const file of files) {
        formData.append("files", file);
      }
      formData.append("decisions", JSON.stringify(decisionsMap));
      formData.append("use_vlm", String(useVlm));

      const response = await fetch(`${API}/documents`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) throw new Error(`HTTP ${response.status}`);

      const result = await response.json();
      console.log("Yükleme sonucu:", result);

      const successCount = result.success?.length ?? 0;
      const skippedCount = result.skipped?.length ?? 0;
      const failedCount = result.failed?.length ?? 0;

      // Özet satırı (1. satır)
      const summary: string[] = [];
      if (successCount > 0) summary.push(`${successCount} yüklendi`);
      if (skippedCount > 0) summary.push(`${skippedCount} atlandı`);
      if (failedCount > 0) summary.push(`${failedCount} başarısız`);

      const lines: string[] = [];
      lines.push(summary.join(" · ") || "Hiçbir dosya işlenmedi");

      // Başarısız dosyaların detayı — hangi dosya neden düştü
      if (failedCount > 0 && Array.isArray(result.failed)) {
        for (const f of result.failed) {
          lines.push(`✗ ${f.file_name}: ${f.reason}`);
        }
      }

      // VLM yüklenemediyse uyarı — görsel içerik atlanmış olur.
      // Sadece kullanıcı VLM'i açmış ama yüklenememişse uyarı göster.
      // Kullanıcı toggle'ı zaten kapatmışsa uyarı yanlış olur.
      if (useVlm && result.vlm_loaded === false && successCount > 0) {
        lines.push(
          "⚠ Görsel modeli yüklenemedi — tablo/şema içerikleri atlandı.",
        );
      }

      // Tip seçimi: failed varsa error, VLM eksikse info, diğer success
      const toastType: ToastMsg["type"] =
        failedCount > 0
          ? "error"
          : useVlm && result.vlm_loaded === false
            ? "info"
            : "success";

      setToast({
        type: toastType,
        message: lines.join("\n"),
      });

      await refreshDocs();
    } catch (err) {
      console.error("Yükleme hatası:", err);
      setToast({
        type: "error",
        message: `Yükleme hatası: ${err instanceof Error ? err.message : "bilinmeyen"}`,
      });
    } finally {
      setIsUploading(false);
      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }
    }
  }

  function confirmConflicts() {
    const files = pendingFiles;
    const decs = decisions;
    // Çakışma onay modalını kapat, doğrudan upload'a geç.
    setPendingFiles([]);
    setConflicts([]);
    setDecisions({});
    doUpload(files, decs);
  }

  function cancelConflicts() {
    setPendingFiles([]);
    setConflicts([]);
    setDecisions({});
    if (fileInputRef.current) fileInputRef.current.value = "";
  }

  function setAllDecisions(value: "overwrite" | "skip") {
    const next: Record<string, "overwrite" | "skip"> = {};
    for (const name of conflicts) {
      next[name] = value;
    }
    setDecisions(next);
  }

  // ── Render ──────────────────────────────────────────────────────────────

  return (
    <div className="flex flex-col h-screen">
      <header className="border-b border-slate-200 bg-white px-5 py-3 flex items-center gap-3 flex-shrink-0">
        <div className="flex items-center gap-2.5">
          <div className="w-7 h-7 rounded-lg bg-gradient-to-br from-indigo-500 to-violet-600 flex items-center justify-center shadow-sm">
            <svg className="w-4 h-4 text-white" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round">
              <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
              <line x1="7" y1="9" x2="17" y2="9" />
              <line x1="7" y1="13" x2="13" y2="13" />
            </svg>
          </div>
          <h1 className="text-base font-semibold text-slate-900 tracking-tight">
            Yerel RAG Asistanı
          </h1>
        </div>
        {selectedDoc && (
          <>
            <div className="h-4 w-px bg-slate-200" />
            <div className="flex items-center gap-1.5 text-xs">
              <span className="text-slate-500">Kapsam:</span>
              <span className="px-2 py-0.5 bg-indigo-50 text-indigo-700 rounded-md font-medium border border-indigo-100">
                📄 {selectedDoc}
              </span>
            </div>
          </>
        )}
        {!selectedDoc && activeChatId && (
          <>
            <div className="h-4 w-px bg-slate-200" />
            <span className="text-xs text-slate-500 italic">
              Kapsam: tüm koleksiyon
            </span>
          </>
        )}
      </header>

      {error && (
        <div className="bg-red-50 border-b border-red-200 px-4 py-2 flex items-center justify-between text-sm flex-shrink-0">
          <span className="text-red-900">
            Backend'e bağlanılamıyor — sunucu çalışıyor mu?{" "}
            <span className="text-red-700 text-xs">({error})</span>
          </span>
          <button
            onClick={fetchInitial}
            disabled={loading}
            className="text-xs px-3 py-1 border border-red-300 text-red-900 rounded-md hover:bg-red-100 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            {loading ? "Bağlanıyor..." : "Tekrar Dene"}
          </button>
        </div>
      )}

      <div className="flex flex-1 overflow-hidden">
        <aside className="w-72 bg-slate-100 border-r border-slate-200 flex flex-col overflow-hidden flex-shrink-0">
          {/* Koleksiyon picker — header'dan buraya taşındı */}
          <div className="p-3 border-b border-slate-200 relative" ref={collectionMenuRef}>
            <button
              onClick={() => setShowCollectionMenu((v) => !v)}
              disabled={isStreaming}
              className="w-full flex items-center justify-between gap-2 px-3 py-2 bg-white border border-slate-200 rounded-lg hover:border-indigo-300 hover:shadow-sm disabled:opacity-50 disabled:cursor-not-allowed transition-all"
              title={isStreaming ? "Sorgu sürüyor, bekleyin" : "Koleksiyon değiştir"}
            >
              <div className="flex items-center gap-2.5 min-w-0">
                <div className="w-7 h-7 rounded-md bg-indigo-50 border border-indigo-100 flex items-center justify-center text-indigo-600 text-sm flex-shrink-0">
                  📚
                </div>
                <div className="text-left min-w-0">
                  <div className="text-[10px] uppercase tracking-wider text-slate-500 font-semibold">
                    Koleksiyon
                  </div>
                  <div className="text-sm font-medium text-slate-900 truncate">
                    {collection || "—"}
                  </div>
                </div>
              </div>
              <svg
                className={`w-4 h-4 text-slate-400 flex-shrink-0 transition-transform ${showCollectionMenu ? "rotate-180" : ""}`}
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
              </svg>
            </button>

            {showCollectionMenu && (
              <div className="absolute left-3 right-3 top-full mt-1.5 bg-white border border-slate-200 rounded-lg shadow-lg z-50 overflow-hidden">
                <ul className="py-1 max-h-64 overflow-y-auto">
                  {allCollections.map((name) => (
                    <li
                      key={name}
                      className={`group flex items-center justify-between gap-2 px-3 py-1.5 text-sm hover:bg-slate-50 ${
                        editingColName === name ? "" : "cursor-pointer"
                      } ${
                        name === collection
                          ? "font-medium text-indigo-700 bg-indigo-50/60"
                          : "text-slate-700"
                      }`}
                      onClick={() => {
                        if (editingColName === name) return;
                        handleSwitchCollection(name);
                      }}
                    >
                      {editingColName === name ? (
                        <input
                          autoFocus
                          type="text"
                          value={editingColValue}
                          onChange={(e) => setEditingColValue(e.target.value)}
                          onClick={(e) => e.stopPropagation()}
                          onKeyDown={(e) => {
                            e.stopPropagation();
                            if (e.key === "Enter") handleSubmitRenameCol();
                            if (e.key === "Escape") handleCancelRenameCol();
                          }}
                          onBlur={handleSubmitRenameCol}
                          className="flex-1 border border-slate-300 rounded px-1.5 py-0.5 text-sm bg-white text-slate-900 focus:outline-none focus:border-indigo-400"
                        />
                      ) : (
                        <span className="truncate flex-1">{name}</span>
                      )}
                      {editingColName !== name && name !== "default" && (
                        <div className="flex items-center gap-1">
                          <button
                            onClick={(e) => {
                              e.stopPropagation();
                              handleStartRenameCol(name);
                            }}
                            className="opacity-0 group-hover:opacity-100 text-slate-400 hover:text-indigo-600 text-xs px-1 transition-colors"
                            title="Yeniden adlandır"
                          >
                            ✏
                          </button>
                          <button
                            onClick={(e) => {
                              e.stopPropagation();
                              handleDeleteCollection(name);
                            }}
                            className="opacity-0 group-hover:opacity-100 text-slate-400 hover:text-red-600 text-xs px-1 transition-colors"
                            title="Koleksiyonu sil"
                          >
                            ×
                          </button>
                        </div>
                      )}
                    </li>
                  ))}
                  <li
                    onClick={handleCreateCollection}
                    className="border-t border-slate-100 mt-1 px-3 py-1.5 text-sm cursor-pointer hover:bg-slate-50 text-indigo-600 font-medium"
                  >
                    + Yeni Koleksiyon
                  </li>
                </ul>
              </div>
            )}
          </div>

          {/* Eylemler: Doküman Ekle + görsel okuma toggle */}
          <div className="px-3 py-3 border-b border-slate-200 space-y-3">
            <button
              onClick={() => fileInputRef.current?.click()}
              disabled={isUploading || isStreaming}
              className="w-full flex items-center justify-center gap-2 px-3 py-2 bg-indigo-600 text-white text-sm font-medium rounded-lg hover:bg-indigo-700 active:bg-indigo-800 disabled:bg-slate-300 disabled:cursor-not-allowed transition-colors shadow-sm"
              title={isStreaming ? "Sorgu sürüyor, bekleyin" : "Doküman ekle"}
            >
              {isUploading ? (
                <>
                  <div className="w-3.5 h-3.5 border-2 border-white border-t-transparent rounded-full animate-spin" />
                  <span>Yükleniyor...</span>
                </>
              ) : (
                <>
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M12 4v16m8-8H4" />
                  </svg>
                  <span>Doküman Ekle</span>
                </>
              )}
            </button>
            <input
              ref={fileInputRef}
              type="file"
              accept=".pdf"
              multiple
              className="hidden"
              onChange={(e) => {
                if (e.target.files) handleUpload(e.target.files);
              }}
            />

            <label
              className="flex items-center justify-between px-0.5 cursor-pointer select-none group"
              title="Açık: tablo/şemalar da işlenir (yavaş). Kapalı: sadece metin (hızlı)."
            >
              <div className="flex items-center gap-2 text-sm text-slate-700 group-hover:text-slate-900">
                <svg className="w-4 h-4 text-slate-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                </svg>
                <span>Görsel okuma</span>
              </div>
              <div className="relative inline-flex items-center">
                <input
                  type="checkbox"
                  checked={useVlm}
                  onChange={(e) => setUseVlm(e.target.checked)}
                  disabled={isUploading}
                  className="sr-only peer"
                />
                <div className="w-9 h-5 bg-slate-300 rounded-full peer-checked:bg-indigo-600 peer-disabled:opacity-50 peer-disabled:cursor-not-allowed transition-colors" />
                <div className="absolute left-0.5 top-0.5 w-4 h-4 bg-white rounded-full transition-transform peer-checked:translate-x-4 shadow-sm pointer-events-none" />
              </div>
            </label>
          </div>

          {/* Dokümanlar + Sohbetler — flex tabanlı bölüşme */}
          <div className="flex flex-col flex-1 min-h-0">
            {/* Dokümanlar */}
            <div className="flex flex-col flex-1 min-h-0">
              <div className="px-3 pt-3 pb-1.5 flex items-center justify-between flex-shrink-0">
                <span className="text-xs uppercase tracking-wider text-slate-500 font-semibold">
                  Dokümanlar
                </span>
                {!loading && docs.length > 0 && (
                  <span className="text-[10px] text-slate-400 font-medium tabular-nums">
                    {docs.length}
                  </span>
                )}
              </div>
              <div className="flex-1 overflow-y-auto px-2 pb-2 min-h-0">
                {loading && (
                  <p className="text-xs text-slate-500 px-2 py-1">Yükleniyor...</p>
                )}
                {!loading && !error && docs.length === 0 && (
                  <div className="text-xs text-slate-400 px-2 py-3 text-center italic">
                    Henüz doküman yok.
                  </div>
                )}
                {!loading && !error && docs.length > 0 && (
                  <ul className="space-y-0.5 text-sm">
                    {docs.map((doc, index) => (
                      <li key={doc.file_name}>
                        <div
                          className={`group flex items-center justify-between gap-2 px-2 py-1.5 rounded-md border transition-colors ${
                            isStreaming
                              ? "opacity-60 cursor-not-allowed"
                              : "cursor-pointer"
                          } ${
                            selectedDoc === doc.file_name
                              ? "bg-white text-indigo-700 border-indigo-200 shadow-sm"
                              : !isStreaming
                                ? "text-slate-700 hover:bg-white/70 border-transparent"
                                : "text-slate-700 border-transparent"
                          }`}
                          onClick={() => {
                            if (isStreaming) return;
                            if (editingDocName === doc.file_name) return;
                            setSelectedDoc((prev) =>
                              prev === doc.file_name ? null : doc.file_name,
                            );
                          }}
                          onDoubleClick={(e) => {
                            if (isStreaming) return;
                            e.stopPropagation();
                            handleStartRenameDoc(doc.file_name);
                          }}
                          title={
                            isStreaming
                              ? "Sorgu sürüyor, bekleyin"
                              : "Çift tıklayarak yeniden adlandır"
                          }
                        >
                          {editingDocName === doc.file_name ? (
                            <input
                              autoFocus
                              type="text"
                              value={editingDocValue}
                              onChange={(e) => setEditingDocValue(e.target.value)}
                              onClick={(e) => e.stopPropagation()}
                              onKeyDown={(e) => {
                                if (e.key === "Enter") handleSubmitRenameDoc();
                                if (e.key === "Escape") handleCancelRenameDoc();
                              }}
                              onBlur={handleSubmitRenameDoc}
                              className="flex-1 border border-slate-300 rounded px-1.5 py-0 text-sm bg-white text-slate-900 focus:outline-none focus:border-indigo-400"
                            />
                          ) : (
                            <div className="flex items-center gap-1.5 min-w-0 flex-1">
                              <span className="text-slate-400 text-xs flex-shrink-0">📄</span>
                              <span className="truncate">{doc.file_name}</span>
                            </div>
                          )}
                          {editingDocName !== doc.file_name && (
                            <div className="flex items-center gap-0.5 flex-shrink-0">
                              <button
                                onClick={(e) => {
                                  e.stopPropagation();
                                  handleReorderDoc(index, -1);
                                }}
                                disabled={index === 0 || isStreaming}
                                className="opacity-0 group-hover:opacity-100 text-slate-400 hover:text-slate-700 text-xs px-1 disabled:opacity-0 transition-colors"
                                title="Yukarı taşı"
                              >
                                ↑
                              </button>
                              <button
                                onClick={(e) => {
                                  e.stopPropagation();
                                  handleReorderDoc(index, 1);
                                }}
                                disabled={
                                  index === docs.length - 1 || isStreaming
                                }
                                className="opacity-0 group-hover:opacity-100 text-slate-400 hover:text-slate-700 text-xs px-1 disabled:opacity-0 transition-colors"
                                title="Aşağı taşı"
                              >
                                ↓
                              </button>
                              <button
                                onClick={(e) => {
                                  e.stopPropagation();
                                  handleDeleteDoc(doc.file_name);
                                }}
                                disabled={
                                  deletingDoc === doc.file_name || isStreaming
                                }
                                className="opacity-0 group-hover:opacity-100 text-slate-400 hover:text-red-600 text-xs px-1 disabled:opacity-0 transition-colors"
                                title="Sil"
                              >
                                {deletingDoc === doc.file_name ? "..." : "×"}
                              </button>
                            </div>
                          )}
                        </div>
                      </li>
                    ))}
                  </ul>
                )}
              </div>
            </div>

            {/* Sohbetler */}
            <div className="flex flex-col flex-1 min-h-0 border-t border-slate-200">
              <div className="px-3 pt-3 pb-1.5 flex items-center justify-between flex-shrink-0">
                <span className="text-xs uppercase tracking-wider text-slate-500 font-semibold">
                  Sohbetler
                </span>
                <div className="flex items-center gap-2">
                  {chats.length > 0 && (
                    <span className="text-[10px] text-slate-400 font-medium tabular-nums">
                      {chats.length}
                    </span>
                  )}
                  <button
                    onClick={handleNewChat}
                    disabled={isStreaming}
                    className="text-slate-400 hover:text-indigo-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                    title={isStreaming ? "Sorgu sürüyor, bekleyin" : "Yeni sohbet"}
                  >
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M12 4v16m8-8H4" />
                    </svg>
                  </button>
                </div>
              </div>
              <div className="flex-1 overflow-y-auto px-2 pb-2 min-h-0">
                {chats.length === 0 && (
                  <div className="text-xs text-slate-400 px-2 py-3 text-center italic">
                    Henüz sohbet yok.
                  </div>
                )}
                {chats.length > 0 && (
                  <ul className="space-y-0.5 text-sm">
                    {chats.map((chat) => (
                      <li
                        key={chat.id}
                        className={`group flex items-center justify-between gap-2 px-2 py-1.5 rounded-md border transition-colors ${
                          isStreaming && chat.id !== activeChatId
                            ? "opacity-50 cursor-not-allowed"
                            : "cursor-pointer"
                        } ${
                          activeChatId === chat.id
                            ? "bg-white text-indigo-700 border-indigo-200 shadow-sm"
                            : !isStreaming
                              ? "text-slate-700 hover:bg-white/70 border-transparent"
                              : "text-slate-700 border-transparent"
                        }`}
                        onClick={() => {
                          if (isStreaming) return;
                          if (editingTitleId !== chat.id) handleSelectChat(chat.id);
                        }}
                        onDoubleClick={(e) => {
                          if (isStreaming) return;
                          e.stopPropagation();
                          handleStartRename(chat);
                        }}
                        title={
                          isStreaming
                            ? "Sorgu sürüyor, bekleyin"
                            : "Çift tıklayarak başlığı düzenle"
                        }
                      >
                        {editingTitleId === chat.id ? (
                          <input
                            autoFocus
                            type="text"
                            value={editingTitleValue}
                            onChange={(e) => setEditingTitleValue(e.target.value)}
                            onClick={(e) => e.stopPropagation()}
                            onKeyDown={(e) => {
                              if (e.key === "Enter") handleSubmitRename();
                              if (e.key === "Escape") handleCancelRename();
                            }}
                            onBlur={handleSubmitRename}
                            className="flex-1 border border-slate-300 rounded px-1.5 py-0 text-sm bg-white text-slate-900 focus:outline-none focus:border-indigo-400"
                          />
                        ) : (
                          <div className="flex items-center gap-1.5 min-w-0 flex-1">
                            <span className="text-slate-400 text-xs flex-shrink-0">💬</span>
                            <span className="truncate">{chat.title}</span>
                          </div>
                        )}
                        {editingTitleId !== chat.id && (
                          <button
                            onClick={(e) => {
                              e.stopPropagation();
                              handleDeleteChat(chat.id);
                            }}
                            disabled={isStreaming}
                            className="opacity-0 group-hover:opacity-100 text-slate-400 hover:text-red-600 text-xs px-1 disabled:opacity-0 transition-colors"
                            title="Sohbeti sil"
                          >
                            ×
                          </button>
                        )}
                      </li>
                    ))}
                  </ul>
                )}
              </div>
            </div>
          </div>
        </aside>

        <main className="flex-1 flex flex-col bg-white">
          {(activeChat || messages.length > 0) && (
            <div className="border-b border-slate-200 bg-white px-4 py-1.5 flex items-center justify-between flex-shrink-0">
              <span className="text-sm text-slate-700 truncate">
                {activeChat ? activeChat.title : "Yeni Sohbet"}
              </span>
              {messages.length > 0 && (
                <button
                  onClick={handleNewChat}
                  disabled={isStreaming}
                  className="text-xs text-slate-600 hover:text-indigo-600 px-2 py-1 rounded hover:bg-slate-50 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                  title={isStreaming ? "Sorgu sürüyor, bekleyin" : "Yeni sohbet"}
                >
                  + Yeni Sohbet
                </button>
              )}
            </div>
          )}
          <div
            ref={messagesContainerRef}
            onScroll={handleMessagesScroll}
            className="flex-1 overflow-y-auto p-4 space-y-3"
          >
            {messages.length === 0 && (
              <div className="flex flex-col items-center justify-center mt-16 text-center px-6">
                <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-indigo-500 to-violet-600 flex items-center justify-center text-white text-xl mb-3 shadow-md">
                  💬
                </div>
                <p className="text-sm text-slate-700 font-medium">
                  {selectedDoc
                    ? `'${selectedDoc}' içinde soru sorabilirsin.`
                    : "Sohbete başla."}
                </p>
                <p className="text-xs text-slate-500 mt-1">
                  {selectedDoc
                    ? "Doküman seçimini kaldırarak tüm koleksiyonda da arayabilirsin."
                    : "Bir doküman seç ya da seçmeden tüm koleksiyonda sor."}
                </p>
              </div>
            )}
            {messages.map((msg, i) => {
              const isLastStreaming = isStreaming && i === messages.length - 1;
              if (msg.role === "user") {
                return (
                  <div key={i} className="max-w-xl ml-auto">
                    <div className="bg-gradient-to-br from-indigo-600 to-violet-700 text-white px-3.5 py-2 rounded-2xl rounded-tr-md whitespace-pre-wrap shadow-md shadow-indigo-900/20">
                      {msg.content}
                    </div>
                    <ScopeBadge scope={msg.scope} />
                  </div>
                );
              }
              return (
                <div
                  key={i}
                  className="max-w-xl bg-slate-50 border border-slate-200 text-slate-800 px-3.5 py-2 rounded-2xl rounded-tl-md"
                >
                  {msg.content ? (
                    <AssistantMessage content={msg.content} />
                  ) : isLastStreaming ? (
                    <span className="text-slate-400 flex items-center gap-1">
                      <span className="w-1.5 h-1.5 bg-indigo-500 rounded-full animate-pulse" />
                      <span className="w-1.5 h-1.5 bg-indigo-500 rounded-full animate-pulse" style={{ animationDelay: "150ms" }} />
                      <span className="w-1.5 h-1.5 bg-indigo-500 rounded-full animate-pulse" style={{ animationDelay: "300ms" }} />
                    </span>
                  ) : null}
                </div>
              );
            })}
            <div ref={messagesEndRef} />
          </div>

          <div className="border-t border-slate-200 bg-white p-3 flex gap-2 flex-shrink-0">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  handleSend();
                }
              }}
              disabled={isStreaming}
              placeholder={
                selectedDoc
                  ? `'${selectedDoc}' içinde sor...`
                  : "Tüm koleksiyonda sor..."
              }
              className="flex-1 bg-white border border-slate-300 rounded-lg px-3.5 py-2 text-sm text-slate-900 placeholder:text-slate-400 focus:outline-none focus:border-indigo-400 focus:ring-2 focus:ring-indigo-100 disabled:bg-slate-50 disabled:cursor-not-allowed transition-all"
            />
            <button
              onClick={handleSend}
              disabled={!input.trim() || isStreaming}
              className="bg-indigo-600 text-white px-4 py-2 rounded-lg text-sm font-medium hover:bg-indigo-700 active:bg-indigo-800 disabled:bg-slate-300 disabled:cursor-not-allowed transition-colors shadow-sm"
            >
              {isStreaming ? "..." : "Gönder"}
            </button>
          </div>
        </main>
      </div>

      {conflicts.length > 0 && (
        <div className="fixed inset-0 bg-slate-900/50 backdrop-blur-sm flex items-center justify-center z-50">
          <div className="bg-white rounded-xl shadow-2xl max-w-lg w-full mx-4 p-6 border border-slate-200">
            <h3 className="text-lg font-semibold text-slate-900 mb-2">Çakışan Dosyalar</h3>
            <p className="text-sm text-slate-600 mb-4">
              Aşağıdaki dosyalar koleksiyonda zaten var. Her biri için ne
              yapılacağını seç.
            </p>

            <div className="flex gap-2 mb-3 text-xs">
              <button
                onClick={() => setAllDecisions("overwrite")}
                className="px-2.5 py-1 border border-slate-300 rounded-md hover:bg-slate-50 hover:border-indigo-300 hover:text-indigo-700 transition-colors"
              >
                Tümüne üzerine yaz
              </button>
              <button
                onClick={() => setAllDecisions("skip")}
                className="px-2.5 py-1 border border-slate-300 rounded-md hover:bg-slate-50 hover:border-indigo-300 hover:text-indigo-700 transition-colors"
              >
                Tümünü atla
              </button>
            </div>

            <ul className="space-y-2 max-h-64 overflow-y-auto mb-4">
              {conflicts.map((name) => (
                <li
                  key={name}
                  className="flex items-center justify-between gap-2"
                >
                  <span className="text-sm text-slate-700 truncate flex-1">{name}</span>
                  <div className="flex gap-1">
                    <button
                      onClick={() =>
                        setDecisions((prev) => ({
                          ...prev,
                          [name]: "overwrite",
                        }))
                      }
                      className={`text-xs px-2 py-1 rounded-md transition-colors ${
                        decisions[name] === "overwrite"
                          ? "bg-indigo-600 text-white"
                          : "border border-slate-300 hover:bg-slate-50"
                      }`}
                    >
                      Üzerine yaz
                    </button>
                    <button
                      onClick={() =>
                        setDecisions((prev) => ({ ...prev, [name]: "skip" }))
                      }
                      className={`text-xs px-2 py-1 rounded-md transition-colors ${
                        decisions[name] === "skip"
                          ? "bg-indigo-600 text-white"
                          : "border border-slate-300 hover:bg-slate-50"
                      }`}
                    >
                      Atla
                    </button>
                  </div>
                </li>
              ))}
            </ul>

            <div className="flex justify-end gap-2">
              <button
                onClick={confirmConflicts}
                className="px-3 py-1.5 bg-indigo-600 text-white text-sm font-medium rounded-md hover:bg-indigo-700 transition-colors shadow-sm"
              >
                Devam Et
              </button>
              <button
                onClick={cancelConflicts}
                className="px-3 py-1.5 border border-slate-300 rounded-md text-sm hover:bg-slate-50 transition-colors"
              >
                İptal
              </button>
            </div>
          </div>
        </div>
      )}

      {isUploading && (
        <div className="fixed inset-0 bg-slate-900/50 backdrop-blur-sm flex items-center justify-center z-50">
          <div className="bg-white rounded-xl shadow-2xl px-8 py-6 flex items-center gap-4 border border-slate-200">
            <div className="w-6 h-6 border-4 border-indigo-600 border-t-transparent rounded-full animate-spin" />
            <div>
              <p className="font-medium text-slate-900">
                {uploadCount} doküman işleniyor…
              </p>
              <p className="text-xs text-slate-500 mt-1 tabular-nums">
                Geçen süre: {formatDuration(uploadElapsedSec)}
              </p>
            </div>
          </div>
        </div>
      )}

      {confirmState && (
        <div className="fixed inset-0 bg-slate-900/50 backdrop-blur-sm flex items-center justify-center z-50">
          <div className="bg-white rounded-xl shadow-2xl max-w-sm w-full mx-4 p-6 border border-slate-200">
            <p className="text-sm text-slate-700 whitespace-pre-line mb-5">
              {confirmState.message}
            </p>
            <div className="flex justify-end gap-2">
              <button
                onClick={() => resolveConfirm(true)}
                className="px-3 py-1.5 bg-red-600 text-white text-sm font-medium rounded-md hover:bg-red-700 transition-colors shadow-sm"
              >
                {confirmState.confirmLabel}
              </button>
              <button
                onClick={() => resolveConfirm(false)}
                className="px-3 py-1.5 border border-slate-300 rounded-md text-sm hover:bg-slate-50 transition-colors"
              >
                Vazgeç
              </button>
            </div>
          </div>
        </div>
      )}

      {promptState && (
        <div className="fixed inset-0 bg-slate-900/50 backdrop-blur-sm flex items-center justify-center z-50">
          <div className="bg-white rounded-xl shadow-2xl max-w-sm w-full mx-4 p-6 border border-slate-200">
            <p className="text-sm text-slate-700 mb-3">{promptState.message}</p>
            <input
              autoFocus
              type="text"
              value={promptState.value}
              onChange={(e) =>
                setPromptState((prev) =>
                  prev ? { ...prev, value: e.target.value } : prev,
                )
              }
              onKeyDown={(e) => {
                if (e.key === "Enter") resolvePrompt(promptState.value);
                if (e.key === "Escape") resolvePrompt(null);
              }}
              className="w-full border border-slate-300 rounded-md px-3 py-1.5 text-sm bg-white text-slate-900 focus:outline-none focus:border-indigo-400 mb-5"
            />
            <div className="flex justify-end gap-2">
              <button
                onClick={() => resolvePrompt(promptState.value)}
                className="px-3 py-1.5 bg-indigo-600 text-white text-sm font-medium rounded-md hover:bg-indigo-700 transition-colors shadow-sm"
              >
                {promptState.submitLabel}
              </button>
              <button
                onClick={() => resolvePrompt(null)}
                className="px-3 py-1.5 border border-slate-300 rounded-md text-sm hover:bg-slate-50 transition-colors"
              >
                Vazgeç
              </button>
            </div>
          </div>
        </div>
      )}

      {toast && (
        <div className="fixed bottom-4 right-4 z-50">
          <div
            className={`px-4 py-3 rounded-lg shadow-lg max-w-md text-sm flex items-start gap-3 border ${
              toast.type === "success"
                ? "bg-green-50 border-green-200 text-green-900"
                : toast.type === "error"
                  ? "bg-red-50 border-red-200 text-red-900"
                  : "bg-indigo-50 border-indigo-200 text-indigo-900"
            }`}
          >
            <span className="flex-1 whitespace-pre-line">{toast.message}</span>
            <button
              onClick={() => setToast(null)}
              className="text-slate-400 hover:text-slate-700 text-xs transition-colors"
            >
              ×
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
