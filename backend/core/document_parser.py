import hashlib
import logging
import os
import re
import shutil
import time
import uuid as _uuid
from collections import Counter

import numpy as np
import pymupdf4llm
from core.config import AppConfig
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode  # ← YENİ
from PIL import Image

log = logging.getLogger(__name__)

# ── Sabitler ──────────────────────────────────────────────────────────────────

# VLM bloğunu baştan sona eşleyen kalıp.
# re.split() ile kullanıldığında capturing group sayesinde
# ayraçların kendisi de sonuç listesine dahil olur.
VLM_BLOCK_PATTERN = re.compile(r"(<VLM_START\b[^>]*>.*?<VLM_END>)", re.DOTALL)


class DocumentParser:
    """
    PDF ayrıştırıcı.

    pymupdf4llm 1.27+ sürümünden itibaren pymupdf_layout paketi otomatik
    olarak kurulur ve import sırasında aktifleşir. Ekstra bir ayar gerekmez.
    Layout modu; başlık tespiti, tablo algılama ve okuma sırası yeniden
    yapılandırması için AI tabanlı bir Graph Neural Network kullanır.

    Sorumluluklar:
        pymupdf4llm  →  Metin, başlık, madde işaretleri, paragraflar ve
                        görsel referansları (![]()). dpi=300 ile görseller
                        yüksek çözünürlükte diske yazılır.

        PIL          →  Görsel filtresi. Dekoratif şeritler, logolar ve
                        küçük öğeleri VLM'e göndermeden eleyerek gereksiz
                        VLM çağrılarını önler.

        VLM          →  Yalnızca filtreden geçen gerçek içerik görselleri
                        için çalışır.
    """

    def __init__(
        self,
        hf_threshold: float = 0.6,
    ):
        # DEBUG: Parser her ingestion için bir kez oluşur; ingestion_engine
        # zaten üst seviyede "Faz 1" + "Parse ediliyor" mesajlarını basıyor,
        # bu satırı INFO yapmak terminalde tekrar etkisi yaratır.
        log.debug("DocumentParser başlatılıyor.")
        self.chunk_size = AppConfig.CHUNK_SIZE
        self.chunk_overlap = AppConfig.CHUNK_OVERLAP
        self.hf_threshold = hf_threshold

        self.parser = SentenceSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )

    # ──────────────────────────────────────────────────────────────────────────
    # BÖLÜM 1 — Tekrar Eden Üstbilgi / Altbilgi Temizleyici
    # ──────────────────────────────────────────────────────────────────────────

    def _remove_frequent_headers_footers(self, pages: list) -> str:
        """
        Sayfa listesindeki her sayfanın ilk ve son birkaç satırını inceler.
        Belirli bir eşiğin üzerinde tekrar eden satırları (sayfa numarası,
        üniversite adı, belge başlığı vb.) tüm sayfalardan siler ve
        temizlenmiş metni tek bir string olarak döndürür.
        """
        total_pages = len(pages)
        if total_pages <= 2:
            # Tekrar tespiti için en az 3 sayfa gerekli. Kısa belgelerde atla.
            log.debug(
                f"Header/footer tespiti atlandı: {total_pages} sayfa "
                "(en az 3 sayfa gerekli)."
            )
            return "\n\n".join([p["text"] for p in pages])

        candidate_lines = []
        for page in pages:
            lines = page["text"].split("\n")
            lines = [line.strip() for line in lines if line.strip()]
            if not lines:
                continue
            candidate_lines.extend(lines[:3])
            candidate_lines.extend(lines[-3:])

        line_counts = Counter(candidate_lines)
        stop_lines = set()
        for line, count in line_counts.items():
            if (count / total_pages) >= self.hf_threshold:
                stop_lines.add(line)

        if stop_lines:
            log.debug(
                f"Tekrar eden header/footer tespit edildi: {len(stop_lines)} satır "
                f"(eşik={self.hf_threshold:.2f}, sayfa={total_pages}). Silinenler: {stop_lines}"
            )
        else:
            log.debug(
                f"Tekrar eden header/footer bulunamadı "
                f"(eşik={self.hf_threshold:.2f}, sayfa={total_pages})."
            )

        cleaned_pages = []
        for page in pages:
            lines = page["text"].split("\n")
            kept_lines = [line for line in lines if line.strip() not in stop_lines]
            cleaned_pages.append("\n".join(kept_lines))

        return "\n\n".join(cleaned_pages)

    # ──────────────────────────────────────────────────────────────────────────
    # BÖLÜM 2 — Markdown Temizleyici
    # ──────────────────────────────────────────────────────────────────────────

    def _clean_markdown(self, text: str) -> str:
        """
        pymupdf4llm çıktısındaki gürültüleri temizler.
        """
        text = re.sub(
            r"\*\*----- Start of picture text -----\*\*.*?\*\*----- End of picture text -----\*\*<br>",
            "",
            text,
            flags=re.DOTALL,
        )
        text = re.sub(
            r"\*\*==> picture \[.*?\] intentionally omitted <==\*\*", "", text
        )
        text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)
        text = re.sub(r"(?:\n[ \t\x0b\f\r\xa0]*){3,}", "\n\n", text)

        # Tablo hücresi içindeki <br> etiketlerini boşlukla değiştir
        # Örnek: |**Kelime**<br>**devamı**| → |**Kelime devamı**|
        # text = re.sub(r"<br\s*/?>", " ", text)

        return text.strip()

    # ──────────────────────────────────────────────────────────────────────────
    # BÖLÜM 3 — Dekoratif Görsel Filtresi (PIL)
    # ──────────────────────────────────────────────────────────────────────────

    def _is_decorative_image(self, image_path: str) -> bool:
        """
        Bir görselin dekoratif/gürültü mü yoksa gerçek içerik mi olduğunu
        üç bağımsız kuralla belirler.

        Kural 1 — Boyut:   Toplam piksel < 4000 veya en kısa kenar < 20px
        Kural 2 — En-boy:  Oran > 20 veya < 0.05
        Kural 3 — Renk:    Std sapma < 18 → neredeyse tek renkli
        """
        try:
            img = Image.open(image_path).convert("RGB")
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
            # Bozuk veya açılamayan görsel. Davranış değişmiyor (True dönüyor
            # ve metinden silinecek), ama olay sessiz kalmasın — debug'a yaz.
            log.debug(
                f"Görsel okunamadı, dekoratif sayıldı: {image_path}",
                exc_info=True,
            )
            return True

    # ──────────────────────────────────────────────────────────────────────────
    # BÖLÜM 4 — VLM Enjeksiyonu (Filtreli)
    # ──────────────────────────────────────────────────────────────────────────

    def _inject_vlm_analysis(self, text: str, vlm_engine) -> str:
        """
        Metin içindeki ![]() görsel referanslarını bulur.
        Her görsel önce _is_decorative_image filtresinden geçer:
            → Dekoratif ise: referans metinden silinir, VLM çağrılmaz.
            → Gerçek içerik ise: VLM'e gönderilir, analiz sonucu görselin
               orijinal konumuna yerleştirilir.
        """
        if vlm_engine is None:
            # ingestion_engine zaten VLM yüklenemediğinde WARNING basıyor;
            # burada tekrar WARNING çift loglama olur. INFO ile yetin.
            log.info("VLM motoru yok, görseller metne dahil edilmeyecek.")
            return text

        pattern = r"!\[.*?\]\((.*?)\)"
        matches = list(re.finditer(pattern, text))

        if not matches:
            log.info("Dokümanda analiz edilecek görsel bulunamadı.")
            return text

        decorative_paths = {
            m.group(1) for m in matches if self._is_decorative_image(m.group(1))
        }
        meaningful_count = len(matches) - len(decorative_paths)

        log.info(
            f"{len(matches)} görsel bulundu → "
            f"{len(decorative_paths)} dekoratif filtrelendi → "
            f"{meaningful_count} görsel VLM'e gönderilecek."
        )

        if meaningful_count == 0:
            for match in matches:
                text = text.replace(match.group(0), "")
            log.info("Tüm görseller dekoratif, VLM çalıştırılmadı.")
            return text

        # VLM çağrılan ama boş içerik dönen görselleri say. İçerik kaybı
        # ingestion başarılı görünse bile oluşabilir — sonda WARNING ile özetle.
        # Dekoratif olduğu için zaten atlananlar bu sayaca DAHİL DEĞİL,
        # onlar yukarıdaki info mesajında ayrı raporlandı.
        failed_count = 0
        vlm_called = 0

        start_time = time.time()
        for match in matches:
            original_tag = match.group(0)
            image_path = match.group(1)

            if image_path in decorative_paths:
                text = text.replace(original_tag, "")
                continue

            vlm_called += 1
            vlm_result = vlm_engine.extract_text(image_path)

            if vlm_result:
                img_id = os.path.basename(image_path)
                replacement = f"\n<VLM_START id='{img_id}'>\n{vlm_result}\n<VLM_END>\n"
                text = text.replace(original_tag, replacement)
            else:
                # Tag silinmeye devam, ama olayı say.
                text = text.replace(original_tag, "")
                failed_count += 1

        log.debug(f"Toplam VLM işlem süresi: {time.time() - start_time:.2f} sn")

        if failed_count > 0:
            log.warning(
                f"VLM {vlm_called} görsel için çağrıldı, "
                f"{failed_count} tanesi boş içerik döndürdü — içerik kaybı oluştu."
            )

        log.info("VLM enjeksiyonu tamamlandı.")
        return text

    # ──────────────────────────────────────────────────────────────────────────
    # BÖLÜM 5 — VLM Farkındalıklı Hibrit Chunker  ← YENİ
    # ──────────────────────────────────────────────────────────────────────────

    def _collect_slide_titles(self, text: str) -> set:
        """
        Metinde 2+ kez tekrar eden ## başlıklarını bulur.

        Slayt PDF'lerinde aynı başlık birden çok slaytta tekrar eder
        ("Tablo Yapım Kuralları" 3 kez gibi). Bunlar yeni bölüm sınırı değil,
        aynı konunun farklı sayfalarıdır. Bu metotla tekrar eden başlıkları
        toplayıp, _split_on_internal_headings'te split noktası olarak
        saymayacağız. Sonuç: ilgili tüm slaytlar tek section'da kalır,
        başlıkla içerik kopmaz.

        Karşılaştırma normalize edilir (case-fold + markdown temizliği) ki
        "**Tablolar**" ile "tablolar" aynı sayılsın.
        """
        from collections import Counter

        counts = Counter()
        for m in re.finditer(r"^#{1,3}\s+(.+)$", text, flags=re.MULTILINE):
            body = m.group(1).strip()
            # Markdown süslerini at, casefold ile karşılaştırılabilir hale getir
            normalized = re.sub(r"[*_`]+", "", body).strip().casefold()
            if normalized:
                counts[normalized] += 1

        return {title for title, n in counts.items() if n >= 2}

    def _is_real_heading(self, line: str) -> bool:
        """
        '#' ile başlayan bir satırın gerçek başlık mı yoksa pymupdf4llm'in
        font-tabanlı yanlış pozitifi mi olduğuna karar verir.

        Yanlış pozitif kaynakları:
            - Büyük puntolu bullet:    "# * Verilerin özetlenmesi..."
            - Büyük puntolu cümle:     "# Sayısal verilerin... grafik denir."
            - Formül/sonuç satırı:     "## χ² = 0,864 P= 0,50"
            - Tek karakter / sembol:   "## ?"
            - Paragraf uzunluğu:       "## ... 200 karakterlik anlatı"

        Heuristikler her birini ayrı eler. Hepsinden geçen satır gerçek başlıktır.

        Tablo/Şekil/Grafik captionları cümle gibi bitse bile başlık sayılır;
        bunlar gerçek bölüm sınırlarıdır.
        """
        stripped = line.strip()
        m = re.match(r"^(#{1,3})\s+(.+)$", stripped)
        if not m:
            return False

        body = m.group(2).strip()
        body_clean = re.sub(r"[*_`]+", "", body).strip()

        # H1 — Bullet ile başlıyor → gerçekte bullet
        if re.match(r"^[\*•\u2022\-]\s", body):
            return False

        # H2 — Boyut filtresi: çok kısa veya paragraf uzunluğunda
        if len(body_clean) < 3 or len(body_clean) > 120:
            return False

        # H3 — Cümle gibi bitiyor (. ! ?) ve yeterince uzun
        #      Caption (Tablo X. / Şekil X.) istisna — bunlar gerçek başlık
        if len(body_clean) > 35 and body_clean[-1] in ".!?":
            if not re.match(r"^(Tablo|Şekil|Grafik|Figure|Table)\s+\d+", body_clean):
                return False

        # H4 — Formül baskın: kısa metin + eşitlik/karşılaştırma + sayı
        if len(body_clean) < 50 and re.search(r"[=<>]\s*\d", body_clean):
            return False

        return True

    def _split_on_internal_headings(
        self, text: str, slide_titles: set = None
    ) -> list[str]:
        """
        Bir metin bloğunu GERÇEK '##' başlıklarından parçalar; sahte başlıkları
        bölme noktası olarak kullanmaz. Bölme sonrası MIN_SEGMENT_LEN altındaki
        parçaları komşusuyla birleştirir — başlık-only orphan bırakmaz.

        Birleştirme kuralı:
            Önce ileri merge: küçük parça → bir sonrakinin başına yapışır.
            Bu yön tercih edilir çünkü heading + içerik bağıntısının doğru yönü budur:
            '## Daire/Pay Grafikler' başlığı kendi altındaki içeriğe yapışır.

            İleri yön mümkün değilse (parça en sondaysa) geri merge yapılır.

        slide_titles parametresi: belgede 2+ kez tekrar eden başlıkların
        normalize edilmiş set'i. Bu set'teki başlıklar split noktası olarak
        sayılmaz; aynı konunun farklı slaytları bütün halinde kalır.
        """
        slide_titles = slide_titles or set()

        # Adım 1 — Aday split noktalarını bul
        text_with_lead = "\n" + text
        candidates = list(re.finditer(r"\n(#{1,2})\s+([^\n]+)", text_with_lead))

        # Sadece gerçek başlıkları split noktası say.
        # Slayt başlığı: belgede 2+ kez tekrar eden başlık. İLK geçişi
        # gerçek bir konu sınırıdır → split olur. Sonraki geçişler aynı
        # konunun farklı slaytlarıdır → split olmaz, içerikte kalır.
        # Bu sayede iki ayrı slide_title grubu aynı chunk'ta birleşmez.
        real_split_offsets = []
        seen_slide_titles: set = set()
        for m in candidates:
            line = m.group(0).lstrip()
            if not self._is_real_heading(line):
                continue
            heading_body = re.sub(r"^#+\s+", "", line).strip()
            heading_norm = re.sub(r"[*_`]+", "", heading_body).strip().casefold()
            if heading_norm in slide_titles:
                if heading_norm in seen_slide_titles:
                    continue  # 2., 3., ... geçiş — split etme, içerikte kalsın
                seen_slide_titles.add(heading_norm)
                # İlk geçiş → normal split akışına devam
            real_split_offsets.append(m.start())

        if not real_split_offsets:
            return [text] if text.strip() else []

        # Adım 2 — Offsetlerden parçaları çıkar
        parts: list[str] = []
        prev = 0
        for off in real_split_offsets:
            if off > prev:
                chunk = text_with_lead[prev:off].strip()
                if chunk:
                    parts.append(chunk)
            prev = off
        last = text_with_lead[prev:].strip()
        if last:
            parts.append(last)

        # Adım 3 — Slide_title kaydırma: parça sonunda asılı kalan tekrar
        # eden başlıkları sonraki parçanın başına taşı. Böylece ana başlık
        # ait olduğu içeriğin üstünde kalır.
        parts = self._shift_trailing_slide_titles(parts, slide_titles)

        # Adım 4 — Küçük parçaları komşusuyla birleştir
        return self._merge_small_parts(parts, AppConfig.MIN_SEGMENT_LEN)

    def _shift_trailing_slide_titles(
        self, parts: list[str], slide_titles: set
    ) -> list[str]:
        """
        Bir parçanın sonunda asılı kalan slide_title satırlarını sonraki
        parçanın başına taşır.

        Neden gerekli: slide_title (belgede 2+ kez tekrar eden başlık),
        aslında SONRAKİ içeriğin ana başlığıdır. Örnek:

            [parça i  ] ## Tablolar + içerik + ## Değişken Türleri
            [parça i+1] ## NİCELİKSEL + içerik

        Burada "Değişken Türleri" Slayt N'nin sonunda DEĞİL, Slayt N+1'in
        BAŞINDA olması gereken ana başlık. Slide_title olduğu için split
        noktası yapılmamış ama önceki parçanın kuyruğunda kalmış.

        Bu metot sondaki slide_title'ları kopartıp sonraki parçaya yapıştırır.
        Son parça için kaydırma yapılmaz (gönderecek bir sonraki parça yok).
        """
        if not slide_titles or len(parts) < 2:
            return parts

        for i in range(len(parts) - 1):
            lines = parts[i].rstrip().split("\n")
            to_shift: list[str] = []

            # Sondan başa doğru: peş peşe slide_title satırlarını topla.
            # Aralarda boş satır olabilir, onları da götürürüz.
            while lines:
                last_line = lines[-1].strip()
                if not last_line:
                    lines.pop()
                    continue

                m = re.match(r"^#{1,2}\s+(.+)$", last_line)
                if not m:
                    break  # heading olmayan içerik satırı geldi, dur

                heading_body = m.group(1).strip()
                heading_norm = re.sub(r"[*_`]+", "", heading_body).strip().casefold()
                if heading_norm not in slide_titles:
                    break  # slide_title değil (gerçek heading), dur

                to_shift.insert(0, lines.pop())

            if to_shift:
                parts[i] = "\n".join(lines).rstrip()
                parts[i + 1] = "\n".join(to_shift) + "\n\n" + parts[i + 1]

        # Kaydırma sonucu boşalan parçaları temizle
        return [p for p in parts if p.strip()]

    def _merge_small_parts(self, parts: list[str], min_len: int) -> list[str]:
        """
        MIN altındaki parçaları ileri-öncelikli birleştirir.

        İleri merge → küçük parça (genelde sadece heading) bir sonrakinin başına
        yapışır. Heading'in altındaki içerikle bütünleşir.

        Geri merge fallback → parça zaten son sıradaysa ve önceki varsa,
        önceki parçanın sonuna eklenir.

        Tek parça varsa (komşu yok) olduğu gibi kabul edilir; bu durumda parçanın
        küçük olması parser'ın değil, asıl belgenin sorunudur.
        """
        if not parts:
            return parts

        out: list[str] = []
        i = 0
        while i < len(parts):
            p = parts[i]
            if len(p) < min_len:
                # İleri merge dene
                if i + 1 < len(parts):
                    parts[i + 1] = p + "\n\n" + parts[i + 1]
                    i += 1
                    continue
                # Geri merge dene
                if out:
                    out[-1] = out[-1] + "\n\n" + p
                    i += 1
                    continue
                # Komşu yok, olduğu gibi al
            out.append(p)
            i += 1

        return out

    def _chunking_with_vlm_awareness(self, enriched_text: str, file_path: str) -> list:
        """
        VLM bloklarını asla bölmeyen, başlık bağlamını koruyan hibrit chunk sistemi.

        Üç aşamalı çalışır:

        Aşama 1 — Tip etiketleme:
            Metin VLM_BLOCK_PATTERN ile parçalanır.
            Her parça "vlm" veya "text" olarak etiketlenir.

        Aşama 2 — Küçük segment birleştirme:
            MIN_SEGMENT_LEN altındaki metin segmentleri (başlıklar, tek
            satırlar) komşularıyla birleştirilir.
            Birleştirme önceliği:
                1. Sonraki segment metin ise → ona önek olarak ekle
                2. Önceki segment metin ise → ona sonek olarak ekle
                3. Her iki taraf da VLM ise → node olarak kalır,
                   başlık yine de son_baslik değişkenine kaydedilir

        Aşama 3 — Node üretimi:
            - Metin segmentleri → SentenceSplitter (overlap korunur)
            - VLM segmentleri → Atomik TextNode (asla bölünmez)
              Her VLM node'una [Bölüm: <başlık>] satırı eklenir.
              Embedding modeli görselin hangi başlık altında olduğunu görür.

        node_index:
            Her node'a sıralı bir tam sayı atanır.
            Retriever bu indeksi kullanarak komşu node'ları getirebilir.
        """
        # ── Aşama 0: Slayt başlıklarını topla ─────────────────────────────────
        # Belgede 2+ kez tekrar eden ## başlıkları slayt başlığıdır;
        # split noktası olarak sayılmazlar. Bir kez toplanır, sonra her
        # text segment için _split_on_internal_headings'e parametre geçilir.
        slide_titles = self._collect_slide_titles(enriched_text)
        if slide_titles:
            log.debug(
                f"Slayt başlığı tespit edildi: {len(slide_titles)} tekrar eden başlık."
            )

        # ── Aşama 1: Tip etiketleme ───────────────────────────────────────────
        raw_segments = VLM_BLOCK_PATTERN.split(enriched_text)
        typed: list[dict] = []
        for seg in raw_segments:
            if not seg.strip():
                continue
            is_vlm = bool(VLM_BLOCK_PATTERN.match(seg.strip()))
            typed.append({"type": "vlm" if is_vlm else "text", "content": seg.strip()})

        # ── Aşama 1.5: Fake heading temizliği ─────────────────────────────────
        # pymupdf4llm font büyüklüğü yüzünden bazı CÜMLELERİ ## ile işaretler
        # (örn. "## Bu tür grafikler her zaman artış gösterir."). _is_real_heading
        # bunları zaten split noktası saymıyor ama markdown ## işareti chunk
        # içinde kalıyor → embedding ekstra token olarak görür.
        # Burada ## prefix'i silinip cümle sıradan paragrafa indirgenir.
        # VLM bloklarına DOKUNULMAZ (içlerinde markdown table header olabilir).
        for seg in typed:
            if seg["type"] != "text":
                continue
            new_lines = []
            for line in seg["content"].split("\n"):
                if line.lstrip().startswith("#"):
                    if not self._is_real_heading(line):
                        line = re.sub(r"^\s*#{1,3}\s+", "", line)
                new_lines.append(line)
            seg["content"] = "\n".join(new_lines)

        # ── Aşama 1.7: Fazla boşluk yutma ─────────────────────────────────────
        # PDF'de bullet maddeler için kullanılan dekoratif görseller filtrelenince
        # yerlerinde boş satır birikir (örn. başlık + 12 ardışık \n + içerik).
        # 3+ ardışık satır sonunu tek paragraf molasına (\n\n) indir.
        # _clean_markdown bunu metin ilk işlendiğinde yapıyor ama VLM enjeksiyonu
        # ondan sonra çalışıp yeni boşluklar yaratıyor — burada tekrar süpürülür.
        # VLM bloklarına DOKUNULMAZ (markdown table iç boşlukları korunmalı).
        for seg in typed:
            if seg["type"] != "text":
                continue
            seg["content"] = re.sub(
                r"(?:\n[ \t\x0b\f\r\xa0]*){3,}", "\n\n", seg["content"]
            )

        # ── Aşama 2: Küçük segment birleştirme ───────────────────────────────
        merged: list[dict] = []
        i = 0
        while i < len(typed):
            seg = typed[i]

            if (
                seg["type"] == "text"
                and len(seg["content"]) < AppConfig.MIN_SEGMENT_LEN
            ):
                # Önce sonraki metin segmentiyle birleştirmeyi dene
                if i + 1 < len(typed) and typed[i + 1]["type"] == "text":
                    typed[i + 1]["content"] = (
                        seg["content"] + "\n\n" + typed[i + 1]["content"]
                    )
                    i += 1
                    continue
                # Sonraki VLM veya yok; önceki metin varsa ona ekle
                elif merged and merged[-1]["type"] == "text":
                    merged[-1]["content"] += "\n\n" + seg["content"]
                    i += 1
                    continue
                # İki tarafı da VLM: node olarak kalır (başlık bilgisi kaybolmasın)

            merged.append(dict(seg))
            i += 1

        # ── Aşama 3: Node üretimi ─────────────────────────────────────────────
        base_metadata = {"file_name": os.path.basename(file_path)}
        nodes: list = []
        last_heading = ""  # Son görülen başlık metni

        for seg in merged:
            if seg["type"] == "vlm":
                # Başlık bağlamını VLM metnine göm.
                # Embedding modeli artık VLM içeriğiyle birlikte başlığı da görür.
                prefix = f"[Bölüm: {last_heading}]\n" if last_heading else ""

                node = TextNode(
                    text=prefix + seg["content"],
                    metadata={
                        **base_metadata,
                        "node_type": "vlm",
                        "node_index": len(nodes),
                    },
                )
                nodes.append(node)

            else:  # text
                # YENİ: Segmenti önce iç ## başlıklarından parçala.
                # Her parça SentenceSplitter'a ayrı ayrı girer.
                # Bu sayede bir segment içindeki farklı konular ayrı node'lara düşer
                # ve _create_section_parents onları section sınırı olarak tanıyabilir.
                sub_segs = self._split_on_internal_headings(
                    seg["content"], slide_titles
                )

                for sub_seg in sub_segs:
                    # Bu parçanın başlığını güncelle — VLM prefix için gerekli
                    sub_heading_matches = re.findall(
                        r"^#{1,2}\s+(.+?)$", sub_seg, re.MULTILINE
                    )
                    if sub_heading_matches:
                        last_heading = (
                            sub_heading_matches[0]
                            .strip()
                            .replace("**", "")
                            .replace("*", "")
                            .strip()
                        )

                    # YENİ: Pure-heading-only kontrolü.
                    # Eğer sub_seg sadece ## başlık satırlarından oluşuyor ve
                    # gerçek içerik (cümle, bullet, tablo) yoksa, node yapma.
                    # last_heading güncellendi; sonraki VLM zaten bunu prefix
                    # olarak alacak. "Başlık + 1-2 cümle" durumunda cümleler
                    # içerik sayılır, node KALIR — kullanıcının nüansı korunur.
                    non_heading_body = re.sub(
                        r"^#{1,3}\s+.+$", "", sub_seg, flags=re.MULTILINE
                    ).strip()
                    if not non_heading_body:
                        # Yalnızca başlık(lar) var. Bilgi last_heading'e taşındı.
                        continue

                    doc = Document(
                        text=sub_seg, metadata={**base_metadata, "node_type": "text"}
                    )
                    text_nodes = self.parser.get_nodes_from_documents([doc])
                    for tn in text_nodes:
                        tn.metadata["node_index"] = len(nodes)
                        nodes.append(tn)

        vlm_count = sum(1 for n in nodes if n.metadata.get("node_type") == "vlm")
        text_count = len(nodes) - vlm_count
        log.debug(
            f"Hibrit chunker tamamlandı: "
            f"{text_count} metin node + {vlm_count} VLM node = {len(nodes)} toplam."
        )
        return nodes

    def _create_section_parents(self, nodes: list) -> list:
        """
        Child node listesini ## başlıklarına göre section'lara gruplar,
        her section için bir parent TextNode üretir.

        Aşamalar:
            Aşama 1 — Heading sınırı tespiti (section_has_content kuralı)
            Aşama 2 — MIN birleştirme: çok kısa section'lar sonrakiyle birleştirilir
            Aşama 2.5 — MAX bölme: çok büyük section'lar node sınırından bölünür.
                         Boyut-tabanlı kesmelerde konu sürekliliği için önceki
                         parçanın sonu bir sonrakinin başına "örtüşme" olarak eklenir.
                         Heading-tabanlı sınırlarda örtüşme EKLENMEZ (farklı konular).
            Aşama 3 — Parent TextNode üretimi, child'lara section metadata eklenmesi
        """

        # Config'den lokal alias (method gövdesinde okunaklılık için)
        MIN_SECTION_CHARS = AppConfig.MIN_SECTION_CHARS
        MAX_SECTION_CHARS = AppConfig.MAX_SECTION_CHARS
        SECTION_OVERLAP_CHARS = AppConfig.SECTION_OVERLAP_CHARS

        child_nodes = [n for n in nodes if n.metadata.get("node_type") != "section"]

        # ── Aşama 1: Heading sınırı tespiti ──────────────────────────────────────
        sections: list[list] = []
        current: list = []
        section_has_content = False

        for node in child_nodes:
            text = node.text or ""
            ntype = node.metadata.get("node_type", "text")
            is_heading = ntype == "text" and bool(re.match(r"^#{1,2}\s", text.strip()))

            if is_heading:
                if section_has_content and current:
                    sections.append(current)
                    current = [node]
                    section_has_content = False
                else:
                    current.append(node)
            else:
                current.append(node)
                if text.strip():
                    section_has_content = True

        if current:
            sections.append(current)

        # ── Aşama 2: MIN birleştirme ──────────────────────────────────────────────
        merged_sections: list[list] = []
        i = 0
        while i < len(sections):
            sec = sections[i]
            total = sum(len(n.text or "") for n in sec)
            if total < MIN_SECTION_CHARS and i + 1 < len(sections):
                sections[i + 1] = sec + sections[i + 1]
            else:
                merged_sections.append(sec)
            i += 1

        # ── Aşama 2.5: MAX bölme — heading sınırlarını tercih eder ──────────────

        #
        # TEMEL PRENSİP: Bir başlık ve altındaki içerik (metin + VLM'ler) bütün
        # kalır. MAX_SECTION_CHARS bir "yumuşak sınır"dır; bir başlığın altındaki
        # tüm içerik bu sınırı aşıyorsa section o kadar büyük kalır, yine de
        # bölünmez. Böylece aynı konuya ait VLM'ler asla farklı section'lara düşmez.
        #
        # Heading bazlı bölmede overlap EKLENMEz (farklı konular, bağlantı gereksiz).
        # Yalnızca 0-1 heading varsa node sınırından bölünür ve overlap eklenir.

        final_sections: list[tuple[list, str]] = []  # (node_listesi, overlap_prefix)

        for sec in merged_sections:
            total = sum(len(n.text or "") for n in sec)

            if total <= MAX_SECTION_CHARS:
                final_sections.append((sec, ""))
                continue

            # Section MAX'ı aşıyor. Kaç tane heading var?
            heading_indices = [
                i
                for i, n in enumerate(sec)
                if n.metadata.get("node_type") == "text"
                and re.match(r"^#{1,2}\s", (n.text or "").strip())
            ]

            if len(heading_indices) > 1:
                # ── Heading bazlı bölme ────────────────────────────────────────
                # Kümülatif boyut MIN_SECTION_CHARS'ı geçtikten sonra bir sonraki
                # heading'de kes. Bu sayede çok kısa section'lar oluşmaz ve
                # her heading bloğunun içeriği (VLM'ler dahil) bütün kalır.
                current_start = 0
                current_len = 0
                temp_splits = []  # (başlangıç, bitiş) index çiftleri

                for i, node in enumerate(sec):
                    is_heading = i in heading_indices
                    if (
                        is_heading
                        and current_len >= MIN_SECTION_CHARS
                        and i > current_start
                    ):
                        # Yeterince dolu, bu heading noktasında kes
                        temp_splits.append((current_start, i))
                        current_start = i
                        current_len = len(node.text or "")
                    else:
                        current_len += len(node.text or "")

                temp_splits.append((current_start, len(sec)))  # Son parça

                for start, end in temp_splits:
                    chunk = sec[start:end]
                    if chunk:
                        # Heading bazlı kesmede overlap yok — konular zaten farklı
                        final_sections.append((chunk, ""))

                log.debug(
                    f"Section MAX aşıyor ({total} char), heading bazlı bölündü: "
                    f"{len(temp_splits)} parça."
                )

            else:
                # ── Node sınırı bazlı bölme (heading yok veya tek heading) ───
                # Başlık yapısı olmayan belgeler (düz makale, rapor vb.) için.
                # Konu sürekliliği için önceki parçanın sonu sonrakine taşınır.
                before = len(final_sections)
                overlap_tail = ""
                current_chunk: list = []
                current_len = 0

                for node in sec:
                    node_len = len(node.text or "")
                    if current_len + node_len > MAX_SECTION_CHARS and current_chunk:
                        final_sections.append((current_chunk, overlap_tail))
                        full_text = "\n".join(n.text or "" for n in current_chunk)
                        overlap_tail = (
                            full_text[-SECTION_OVERLAP_CHARS:]
                            if len(full_text) > SECTION_OVERLAP_CHARS
                            else full_text
                        )
                        current_chunk = [node]
                        current_len = node_len
                    else:
                        current_chunk.append(node)
                        current_len += node_len

                if current_chunk:
                    final_sections.append((current_chunk, overlap_tail))

                created = len(final_sections) - before
                log.debug(
                    f"Section MAX aşıyor ({total} char), node sınırı bazlı bölündü: "
                    f"{created} parça, ~{SECTION_OVERLAP_CHARS} char overlap."
                )

        # ── Aşama 3: Parent node'ları üret ───────────────────────────────────────
        result_nodes: list = []

        for sec_nodes, overlap_prefix in final_sections:
            section_id = str(_uuid.uuid4())

            indices = [
                n.metadata["node_index"]
                for n in sec_nodes
                if n.metadata.get("node_index") is not None
            ]
            start_idx = min(indices) if indices else 0
            end_idx = max(indices) if indices else 0

            core_text = "\n\n".join(
                (n.text or "").strip() for n in sec_nodes if (n.text or "").strip()
            )

            # Boyut-tabanlı kesme varsa önceki konu bağlamını öne ekle
            if overlap_prefix:
                parent_text = (
                    f"[Önceki Bölüm Bağlamı: ...{overlap_prefix}]\n\n{core_text}"
                )
            else:
                parent_text = core_text

            # Section'ın baskın başlığını çek: core_text'in ilk ## satırı.
            # Retriever debug çıktısı bunu okuyor; olmazsa 'sec=[?-?]' gibi
            # anlamsız etiketler görünüyor.
            section_heading = ""
            for line in core_text.split("\n"):
                line = line.strip()
                hm = re.match(r"^#{1,2}\s+(.+)$", line)
                if hm:
                    section_heading = re.sub(r"[*_`]+", "", hm.group(1)).strip()
                    break

            from llama_index.core.schema import TextNode as _TextNode

            parent_node = _TextNode(
                id_=section_id,
                text=parent_text,
                metadata={
                    "file_name": sec_nodes[0].metadata.get("file_name", ""),
                    "node_type": "section",
                    "section_id": section_id,
                    "section_heading": section_heading,
                    "node_index": start_idx,
                    "section_start_index": start_idx,
                    "section_end_index": end_idx,
                },
            )

            for child in sec_nodes:
                child.metadata["section_id"] = section_id
                child.metadata["section_start_index"] = start_idx
                child.metadata["section_end_index"] = end_idx
                result_nodes.append(child)

            result_nodes.append(parent_node)

        log.debug(f"Section parent'ları oluşturuldu: {len(final_sections)} section.")
        return result_nodes

    # ──────────────────────────────────────────────────────────────────────────
    # BÖLÜM 6 — Yol Yardımcısı
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _slugify_for_path(name: str) -> str:
        """
        Klasör adını ASCII-safe hale getirir.

        Neden: PyMuPDF'in alt katmanındaki MuPDF (C) kütüphanesi Windows'ta
        PNG kaydederken yolu sistem encoding'i (cp1254) ile açmaya çalışıyor;
        'ö', 'ğ', 'ş' gibi karakterler içeren bir klasör verildiğinde
        FzErrorSystem (code=2) hatası alınıyor. Bu nedenle temp_images altında
        oluşturulan alt klasörün adını ASCII'ye düşürüyoruz. PDF dosyasının
        kendi adı (file_path) DEĞİŞMEZ — sadece geçici görsel klasörü.
        """
        tr_map = str.maketrans({
            "ç": "c", "Ç": "C",
            "ğ": "g", "Ğ": "G",
            "ı": "i", "İ": "I",
            "ö": "o", "Ö": "O",
            "ş": "s", "Ş": "S",
            "ü": "u", "Ü": "U",
        })
        s = name.translate(tr_map)
        # Geride kalan non-ASCII karakterleri ve boşluk/özel karakterleri at
        s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
        # Çift alt çizgileri tekleştir, baş/son alt çizgileri kırp
        s = re.sub(r"_+", "_", s).strip("_.")
        return s or "_unnamed"

    # ──────────────────────────────────────────────────────────────────────────
    # ANA METOD
    # ──────────────────────────────────────────────────────────────────────────

    def parse(self, file_path: str, vlm_engine=None):
        """
        Ayrıştırma akışı:

            Adım 1 — pymupdf4llm:  Metin + görsel referansları çıkar
            Adım 2 — Üstbilgi/altbilgi temizliği
            Adım 3 — Markdown temizliği
            Adım 4 — VLM enjeksiyonu
            Adım 5 — VLM farkındalıklı hibrit chunking
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"HATA: Ayrıştırılacak belge bulunamadı → {file_path}"
            )

        parse_start = time.time()
        base_name = os.path.basename(file_path)
        log.info(f"Parse başlatıldı: {base_name}")

        name_without_ext = os.path.splitext(base_name)[0]
        # MuPDF Windows'ta non-ASCII yolları açamadığı için temp klasör adı
        # ASCII'ye indirilir. PDF dosya yolu (file_path) olduğu gibi kalır.
        safe_name = self._slugify_for_path(name_without_ext)
        # Çakışma koruması: orijinal isimden 8-karakter deterministik hash.
        # "Çalışma.pdf" ve "Calisma.pdf" gibi slug'ı eşleşen dosyalar farklı
        # klasöre yazar; aynı PDF tekrar parse edilirse aynı klasöre düşer
        # (eski görseller üzerine yazılır, orphan klasör üretmez).
        hash_suffix = hashlib.md5(base_name.encode("utf-8")).hexdigest()[:8]
        unique_dir = f"{safe_name}_{hash_suffix}"
        temp_img_dir = str(AppConfig.TEMP_IMAGES_DIR / unique_dir)
        os.makedirs(temp_img_dir, exist_ok=True)

        # Görseller bu noktadan sonra Adım 4'te (VLM) tüketiliyor; parse
        # bittiğinde node'lar görsel yolu değil VLM metnini taşır. Bu yüzden
        # try/finally ile sarıp, parse hata verse bile bu dosyanın geçici
        # görsel klasörünü siliyoruz — temp_images sınırsız büyümesin.
        try:
            # ── Adım 1 ───────────────────────────────────────────────────────────
            log.debug("Adım 1: pymupdf4llm (Layout modu) çalıştırılıyor.")
            step_t = time.time()
            md_pages = pymupdf4llm.to_markdown(
                doc=file_path,
                write_images=True,
                image_path=temp_img_dir,
                dpi=250,
                page_chunks=True,
            )
            log.debug(
                f"Adım 1 bitti: {len(md_pages)} sayfa, {time.time() - step_t:.2f} sn."
            )

            # ── Adım 2 ───────────────────────────────────────────────────────────
            joined_text = self._remove_frequent_headers_footers(md_pages)

            # ── Adım 3 ───────────────────────────────────────────────────────────
            clean_text = self._clean_markdown(joined_text)

            # ── Adım 4 ───────────────────────────────────────────────────────────
            enriched_text = self._inject_vlm_analysis(clean_text, vlm_engine)

            # ── Adım 5 ───────────────────────────────────────────────────────────
            nodes = self._chunking_with_vlm_awareness(enriched_text, file_path)

            # parse() içinde, _chunking_with_vlm_awareness'tan sonra
            nodes = self._create_section_parents(nodes)  # ← YENİ: parent-child katmanı

            log.info(
                f"Parse tamamlandı: {base_name} — "
                f"{len(md_pages)} sayfa, {len(nodes)} node, "
                f"{time.time() - parse_start:.2f} sn."
            )
            return nodes
        finally:
            # Geçici görsel klasörünü temizle. ignore_errors: Windows'ta dosya
            # kilidi vb. olsa bile parse'ı patlatmasın.
            shutil.rmtree(temp_img_dir, ignore_errors=True)
            log.debug(f"Geçici görsel klasörü silindi: {temp_img_dir}")
