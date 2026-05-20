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
            <details key={i} className="text-xs text-gray-500">
              <summary className="cursor-pointer hover:text-gray-700 select-none">
                {part.complete ? "Düşünme sürecini gör" : "Düşünüyor..."}
              </summary>
              <div className="mt-1 pl-3 border-l-2 border-gray-300 whitespace-pre-wrap text-gray-600">
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

// User mesajının yanında "hangi kapsamda soruldu" rozeti.
// type=document → 📄 <file_name>, type=collection → 📁 koleksiyon
function ScopeBadge({ scope }: { scope?: MessageScope }) {
  if (!scope) return null;
  if (scope.type === "document" && scope.file_name) {
    return (
      <div className="text-xs text-gray-400 mt-1 text-right">
        📄 {scope.file_name}
      </div>
    );
  }
  if (scope.type === "collection") {
    return (
      <div className="text-xs text-gray-400 mt-1 text-right">
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
  // Taşıma menüsü açık olan doküman — li'nin altına hedef listesi açılır
  const [movingDoc, setMovingDoc] = useState<string | null>(null);
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
  const [toast, setToast] = useState<ToastMsg | null>(null);

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
    if (!confirm("Bu sohbet silinecek. Emin misin?")) return;
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

  // Doküman taşıma — inline mini menü ile hedef koleksiyon seçtirir.
  // Embedding yeniden hesaplanmaz, ChromaDB chunks aynı vektörlerle
  // target koleksiyona kopyalanıp source'tan silinir.
  function handleStartMoveDoc(fileName: string) {
    setMovingDoc(fileName);
  }

  function handleCancelMoveDoc() {
    setMovingDoc(null);
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

  async function handleMoveDoc(fileName: string, target: string) {
    setMovingDoc(null);
    try {
      const res = await fetch(`${API}/documents/move`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          file_name: fileName,
          target_collection: target,
        }),
      });

      if (res.status === 409) {
        const err = await res.json().catch(() => null);
        setToast({
          type: "error",
          message: err?.detail || `'${fileName}' hedefte zaten var.`,
        });
        return;
      }
      if (!res.ok) {
        const err = await res.json().catch(() => null);
        throw new Error(err?.detail || `HTTP ${res.status}`);
      }

      const result = await res.json();
      // Taşınan doküman aktif koleksiyondan kalktı, listeyi yenile
      await refreshDocs();
      // Seçili doküman taşınmışsa seçimi kaldır — artık bu koleksiyonda yok
      if (selectedDoc === fileName) {
        setSelectedDoc(null);
      }
      setToast({
        type: "success",
        message: `'${fileName}' -> '${target}' (${result.chunks ?? 0} chunk)`,
      });
    } catch (err) {
      setToast({
        type: "error",
        message: `Taşınamadı: ${err instanceof Error ? err.message : "bilinmeyen"}`,
      });
    }
  }

  async function handleDeleteDoc(fileName: string) {
    if (!confirm(`'${fileName}' silinecek. Emin misin?`)) return;

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
        alert(`Silinemedi: ${result.failed[0].reason}`);
      } else {
        // Silinen dosya seçiliyse seçimi kaldır — ama aktif sohbetin
        // mesajlarına dokunma, onlar geriye dönük görüntüleme için kalır.
        if (selectedDoc === fileName) {
          setSelectedDoc(null);
        }
        await refreshDocs();
      }
    } catch (err) {
      alert(
        `Silme hatası: ${err instanceof Error ? err.message : "bilinmeyen"}`,
      );
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
      alert(
        `Koleksiyon değiştirilemedi: ${err instanceof Error ? err.message : "bilinmeyen"}`,
      );
    }
  }

  async function handleCreateCollection() {
    const name = prompt("Yeni koleksiyon adı:");
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
      alert(
        `Koleksiyon oluşturulamadı: ${err instanceof Error ? err.message : "bilinmeyen"}`,
      );
    }
  }

  async function handleDeleteCollection(name: string) {
    if (
      !confirm(
        `'${name}' koleksiyonu ve içindeki tüm dokümanlar silinecek. Emin misin?`,
      )
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
      alert(
        `Koleksiyon silinemedi: ${err instanceof Error ? err.message : "bilinmeyen"}`,
      );
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
      alert(
        `Kontrol hatası: ${err instanceof Error ? err.message : "bilinmeyen"}`,
      );
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
    doUpload(pendingFiles, decisions);
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
      <header className="border-b px-4 py-2 flex items-center gap-4 relative">
        <h1 className="text-lg font-semibold">Yerel RAG Asistanı</h1>
        {selectedDoc && (
          <span className="text-sm text-gray-700">
            Seçili: <span className="font-medium">{selectedDoc}</span>
          </span>
        )}
        {!selectedDoc && activeChatId && (
          <span className="text-sm text-gray-500 italic">
            Kapsam: tüm koleksiyon
          </span>
        )}
        <div className="ml-auto relative" ref={collectionMenuRef}>
          <button
            onClick={() => setShowCollectionMenu((v) => !v)}
            disabled={isStreaming}
            className="text-sm text-gray-700 hover:bg-gray-100 px-3 py-1 rounded flex items-center gap-1 disabled:opacity-50 disabled:cursor-not-allowed"
            title={isStreaming ? "Sorgu sürüyor, bekleyin" : undefined}
          >
            Koleksiyon: <span className="font-medium">{collection || "—"}</span>
            <span className="text-xs">▼</span>
          </button>

          {showCollectionMenu && (
            <div className="absolute right-0 top-full mt-1 bg-white border rounded shadow-lg min-w-50 z-50">
              <ul className="py-1">
                {allCollections.map((name) => (
                  <li
                    key={name}
                    className={`group flex items-center justify-between gap-2 px-3 py-1 text-sm hover:bg-gray-100 ${
                      editingColName === name ? "" : "cursor-pointer"
                    } ${
                      name === collection ? "font-medium text-blue-600" : ""
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
                        className="flex-1 border rounded px-1 py-0 text-sm bg-white text-gray-900"
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
                          className="opacity-0 group-hover:opacity-100 text-gray-500 hover:text-blue-600 text-xs px-1"
                          title="Yeniden adlandır"
                        >
                          ✏
                        </button>
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            handleDeleteCollection(name);
                          }}
                          className="opacity-0 group-hover:opacity-100 text-gray-500 hover:text-red-600 text-xs px-1"
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
                  className="border-t mt-1 px-3 py-1 text-sm cursor-pointer hover:bg-gray-100 text-blue-600"
                >
                  + Yeni Koleksiyon
                </li>
              </ul>
            </div>
          )}
        </div>
      </header>

      {error && (
        <div className="bg-red-50 border-b border-red-200 px-4 py-2 flex items-center justify-between text-sm">
          <span className="text-red-900">
            Backend'e bağlanılamıyor — sunucu çalışıyor mu?{" "}
            <span className="text-red-700 text-xs">({error})</span>
          </span>
          <button
            onClick={fetchInitial}
            disabled={loading}
            className="text-xs px-3 py-1 border border-red-300 text-red-900 rounded hover:bg-red-100 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {loading ? "Bağlanıyor..." : "Tekrar Dene"}
          </button>
        </div>
      )}

      <div className="flex flex-1 overflow-hidden">
        <aside className="w-64 border-r flex flex-col overflow-hidden">
          {/* Dokümanlar */}
          <div className="p-4 border-b overflow-y-auto" style={{ maxHeight: "50%" }}>
            <div className="flex items-center justify-between mb-2">
              <h2 className="font-semibold">Dokümanlar</h2>
              <button
                onClick={() => fileInputRef.current?.click()}
                disabled={isUploading || isStreaming}
                className="text-xs bg-blue-500 text-white px-2 py-1 rounded hover:bg-blue-600 disabled:bg-gray-300 disabled:cursor-not-allowed"
                title={
                  isStreaming ? "Sorgu sürüyor, bekleyin" : "Doküman ekle"
                }
              >
                {isUploading ? "..." : "+ Ekle"}
              </button>
            </div>
            <label
              className="flex items-center gap-1.5 text-xs text-gray-600 mb-2 cursor-pointer hover:text-gray-800 select-none"
              title="Açık: tablo/şemalar da işlenir (yavaş). Kapalı: sadece metin (hızlı)."
            >
              <input
                type="checkbox"
                checked={useVlm}
                onChange={(e) => setUseVlm(e.target.checked)}
                disabled={isUploading}
                className="cursor-pointer disabled:cursor-not-allowed"
              />
              <span>Görsel okuma</span>
            </label>
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

            {loading && <p className="text-sm text-gray-500">Yükleniyor...</p>}
            {!loading && !error && docs.length === 0 && (
              <p className="text-sm text-gray-500">Doküman yok.</p>
            )}
            {!loading && !error && docs.length > 0 && (
              <ul className="space-y-1 text-sm">
                {docs.map((doc, index) => (
                  <li key={doc.file_name} className="rounded">
                    <div
                      className={`group flex items-center justify-between gap-2 px-2 py-1 rounded ${
                        isStreaming
                          ? "opacity-60 cursor-not-allowed"
                          : "cursor-pointer"
                      } ${
                        selectedDoc === doc.file_name
                          ? "bg-blue-100 text-blue-900"
                          : !isStreaming
                            ? "hover:bg-gray-100"
                            : ""
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
                          className="flex-1 border rounded px-1 py-0 text-sm bg-white text-gray-900"
                        />
                      ) : (
                        <span className="truncate">{doc.file_name}</span>
                      )}
                      {editingDocName !== doc.file_name && (
                        <div className="flex items-center gap-1 flex-shrink-0">
                          <button
                            onClick={(e) => {
                              e.stopPropagation();
                              handleReorderDoc(index, -1);
                            }}
                            disabled={index === 0 || isStreaming}
                            className="opacity-0 group-hover:opacity-100 text-gray-500 hover:text-gray-800 text-xs px-1 disabled:opacity-0"
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
                            className="opacity-0 group-hover:opacity-100 text-gray-500 hover:text-gray-800 text-xs px-1 disabled:opacity-0"
                            title="Aşağı taşı"
                          >
                            ↓
                          </button>
                          <button
                            onClick={(e) => {
                              e.stopPropagation();
                              handleStartMoveDoc(doc.file_name);
                            }}
                            disabled={isStreaming}
                            className="opacity-0 group-hover:opacity-100 text-gray-500 hover:text-blue-600 text-xs px-1 disabled:opacity-0"
                            title="Başka koleksiyona taşı"
                          >
                            →
                          </button>
                          <button
                            onClick={(e) => {
                              e.stopPropagation();
                              handleDeleteDoc(doc.file_name);
                            }}
                            disabled={
                              deletingDoc === doc.file_name || isStreaming
                            }
                            className="opacity-0 group-hover:opacity-100 text-gray-500 hover:text-red-600 text-xs px-1 disabled:opacity-0"
                            title="Sil"
                          >
                            {deletingDoc === doc.file_name ? "..." : "×"}
                          </button>
                        </div>
                      )}
                    </div>
                    {movingDoc === doc.file_name && (
                      <div className="mt-1 mx-2 mb-1 px-2 py-1 bg-gray-50 border border-gray-200 rounded text-xs">
                        <div className="text-gray-600 mb-1">Şuraya taşı:</div>
                        {allCollections.filter((c) => c !== collection)
                          .length === 0 ? (
                          <div className="text-gray-500 italic">
                            Başka koleksiyon yok.
                          </div>
                        ) : (
                          <div className="flex flex-wrap gap-1">
                            {allCollections
                              .filter((c) => c !== collection)
                              .map((c) => (
                                <button
                                  key={c}
                                  onClick={(e) => {
                                    e.stopPropagation();
                                    handleMoveDoc(doc.file_name, c);
                                  }}
                                  className="px-2 py-0.5 border border-gray-300 rounded bg-white hover:bg-blue-50 hover:border-blue-300"
                                >
                                  {c}
                                </button>
                              ))}
                          </div>
                        )}
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            handleCancelMoveDoc();
                          }}
                          className="mt-1 text-gray-500 hover:text-gray-700"
                        >
                          İptal
                        </button>
                      </div>
                    )}
                  </li>
                ))}
              </ul>
            )}
          </div>

          {/* Sohbetler */}
          <div className="p-4 flex-1 overflow-y-auto">
            <div className="flex items-center justify-between mb-2">
              <h2 className="font-semibold">Sohbetler</h2>
              <button
                onClick={handleNewChat}
                disabled={isStreaming}
                className="text-xs bg-blue-500 text-white px-2 py-1 rounded hover:bg-blue-600 disabled:bg-gray-300 disabled:cursor-not-allowed"
                title={isStreaming ? "Sorgu sürüyor, bekleyin" : "Yeni sohbet aç"}
              >
                + Yeni
              </button>
            </div>
            {chats.length === 0 && (
              <p className="text-sm text-gray-500">Henüz sohbet yok.</p>
            )}
            {chats.length > 0 && (
              <ul className="space-y-1 text-sm">
                {chats.map((chat) => (
                  <li
                    key={chat.id}
                    className={`group flex items-center justify-between gap-2 px-2 py-1 rounded ${
                      isStreaming && chat.id !== activeChatId
                        ? "opacity-50 cursor-not-allowed"
                        : "cursor-pointer"
                    } ${
                      activeChatId === chat.id
                        ? "bg-blue-100 text-blue-900"
                        : !isStreaming
                          ? "hover:bg-gray-100"
                          : ""
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
                        className="flex-1 border rounded px-1 py-0 text-sm bg-white text-gray-900"
                      />
                    ) : (
                      <span className="truncate flex-1">{chat.title}</span>
                    )}
                    {editingTitleId !== chat.id && (
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          handleDeleteChat(chat.id);
                        }}
                        disabled={isStreaming}
                        className="opacity-0 group-hover:opacity-100 text-gray-500 hover:text-red-600 text-xs px-1 disabled:opacity-0"
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
        </aside>

        <main className="flex-1 flex flex-col">
          {(activeChat || messages.length > 0) && (
            <div className="border-b px-4 py-1 flex items-center justify-between">
              <span className="text-sm text-gray-600 truncate">
                {activeChat ? activeChat.title : "Yeni Sohbet"}
              </span>
              {messages.length > 0 && (
                <button
                  onClick={handleNewChat}
                  disabled={isStreaming}
                  className="text-xs text-gray-600 hover:text-gray-900 px-2 py-1 rounded hover:bg-gray-100 disabled:opacity-50 disabled:cursor-not-allowed"
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
              <p className="text-sm text-gray-400 text-center mt-8">
                {selectedDoc
                  ? "Soru yazıp gönderebilirsin."
                  : "Bir doküman seç ya da seçmeden tüm koleksiyonda sor."}
              </p>
            )}
            {messages.map((msg, i) => {
              const isLastStreaming = isStreaming && i === messages.length - 1;
              if (msg.role === "user") {
                return (
                  <div key={i} className="max-w-xl ml-auto">
                    <div className="bg-blue-500 text-white px-3 py-2 rounded-lg whitespace-pre-wrap">
                      {msg.content}
                    </div>
                    <ScopeBadge scope={msg.scope} />
                  </div>
                );
              }
              return (
                <div
                  key={i}
                  className="max-w-xl bg-gray-100 px-3 py-2 rounded-lg"
                >
                  {msg.content ? (
                    <AssistantMessage content={msg.content} />
                  ) : isLastStreaming ? (
                    <span className="text-gray-500">...</span>
                  ) : null}
                </div>
              );
            })}
            <div ref={messagesEndRef} />
          </div>

          <div className="border-t p-4 flex gap-2">
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
              className="flex-1 border rounded px-3 py-2 disabled:bg-gray-100"
            />
            <button
              onClick={handleSend}
              disabled={!input.trim() || isStreaming}
              className="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600 disabled:bg-gray-300 disabled:cursor-not-allowed"
            >
              {isStreaming ? "..." : "Gönder"}
            </button>
          </div>
        </main>
      </div>

      {conflicts.length > 0 && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg shadow-xl max-w-lg w-full mx-4 p-6">
            <h3 className="text-lg font-semibold mb-2">Çakışan Dosyalar</h3>
            <p className="text-sm text-gray-600 mb-4">
              Aşağıdaki dosyalar koleksiyonda zaten var. Her biri için ne
              yapılacağını seç.
            </p>

            <div className="flex gap-2 mb-3 text-xs">
              <button
                onClick={() => setAllDecisions("overwrite")}
                className="px-2 py-1 border rounded hover:bg-gray-50"
              >
                Tümüne üzerine yaz
              </button>
              <button
                onClick={() => setAllDecisions("skip")}
                className="px-2 py-1 border rounded hover:bg-gray-50"
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
                  <span className="text-sm truncate flex-1">{name}</span>
                  <div className="flex gap-1">
                    <button
                      onClick={() =>
                        setDecisions((prev) => ({
                          ...prev,
                          [name]: "overwrite",
                        }))
                      }
                      className={`text-xs px-2 py-1 rounded ${
                        decisions[name] === "overwrite"
                          ? "bg-blue-500 text-white"
                          : "border hover:bg-gray-50"
                      }`}
                    >
                      Üzerine yaz
                    </button>
                    <button
                      onClick={() =>
                        setDecisions((prev) => ({ ...prev, [name]: "skip" }))
                      }
                      className={`text-xs px-2 py-1 rounded ${
                        decisions[name] === "skip"
                          ? "bg-blue-500 text-white"
                          : "border hover:bg-gray-50"
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
                onClick={cancelConflicts}
                className="px-3 py-1 border rounded hover:bg-gray-50"
              >
                İptal
              </button>
              <button
                onClick={confirmConflicts}
                className="px-3 py-1 bg-blue-500 text-white rounded hover:bg-blue-600"
              >
                Devam Et
              </button>
            </div>
          </div>
        </div>
      )}

      {isUploading && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg shadow-xl px-8 py-6 flex items-center gap-4">
            <div className="w-6 h-6 border-4 border-blue-500 border-t-transparent rounded-full animate-spin" />
            <div>
              <p className="font-medium">Dokümanlar işleniyor...</p>
              <p className="text-xs text-gray-500 mt-1">
                PDF parse + VLM analizi + embedding. Bu dakikalar sürebilir.
              </p>
            </div>
          </div>
        </div>
      )}

      {toast && (
        <div className="fixed bottom-4 right-4 z-50">
          <div
            className={`px-4 py-3 rounded-lg shadow-lg max-w-md text-sm flex items-start gap-3 ${
              toast.type === "success"
                ? "bg-green-50 border border-green-200 text-green-900"
                : toast.type === "error"
                  ? "bg-red-50 border border-red-200 text-red-900"
                  : "bg-blue-50 border border-blue-200 text-blue-900"
            }`}
          >
            <span className="flex-1 whitespace-pre-line">{toast.message}</span>
            <button
              onClick={() => setToast(null)}
              className="text-gray-500 hover:text-gray-700 text-xs"
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
