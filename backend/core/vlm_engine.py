"""
VLM Engine — ZwZ-4B (Qwen3VL tabanlı)
Kategorisiz, kural tabanlı evrensel prompt.
"""

import base64
import io
import logging
import os
import time

from core.config import AppConfig
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Qwen3VLChatHandler
from PIL import Image

log = logging.getLogger(__name__)

_PATCH_SIZE: int = 32
# Ölçümle bulunan değer. Eski 1_310_720 fazla düşüktü: yoğun görselleri
# (tablo/şema) zorlayıp modele fazla token ürettiriyor, net YAVAŞLATIYORdu.
# 3M'de yoğun görseller belirgin hızlandı, zaten okunan görsellerde fark yok.
# Daha yükseğe çıkmanın bu korpusta kazancı yok (görseller 3M'i geçmiyor),
# sınırsız ise OOM/kuyruk riski açar — 3M dengeli üst sınır.
_VLM_MAX_PIXELS: int = 3_000_000


class VLMEngine:
    """
    Görsel → metin çıkaran VLM sarmalayıcısı (ZwZ-4B, Qwen3-VL tabanlı).

    Yaşam döngüsü: __init__'te model + mmproj (CLIP) bir kez VRAM'e yüklenir,
    her görsel için extract_text() çağrılır, iş bitince unload() ile VRAM
    boşaltılır. Ingestion'da Faz 1 boyunca tek instance tüm PDF'lerin
    görsellerini işler (model başına tekrar yükleme yok).

    Çıktı Markdown: tablo → Markdown tablosu, şema/grafik → yapısal açıklama.
    Bu metin sonra parser tarafından <VLM_START>...<VLM_END> etiketleriyle
    doküman metnine gömülür.
    """

    def __init__(self) -> None:
        self.model_path = str(AppConfig.VLM_MODEL_PATH)
        self.mmproj_path = str(AppConfig.VLM_MMPROJ_PATH)

        if not os.path.exists(self.model_path) or not os.path.exists(self.mmproj_path):
            raise FileNotFoundError(
                "[HATA] VLM veya mmproj model dosyaları bulunamadı!\n"
                f"  model : {self.model_path}\n"
                f"  mmproj: {self.mmproj_path}"
            )

        self.chat_handler = Qwen3VLChatHandler(clip_model_path=self.mmproj_path)

        load_start = time.time()
        self.llm = Llama(
            model_path=self.model_path,
            chat_handler=self.chat_handler,
            n_ctx=AppConfig.VLM_N_CTX,
            n_gpu_layers=-1,
            n_batch=2048,
            flash_attn_type=1,
            swa_full=True,
            type_k=8,
            type_v=8,
            verbose=False,
        )

        log.info(f"VLM Motoru (ZwZ-4B) yüklendi ({time.time() - load_start:.2f} sn).")

    def _prepare_image(self, file_path: str) -> str:
        # Görseli RGB'ye çevir, çok büyükse _VLM_MAX_PIXELS'e ölçekle (vision
        # token sayısını ve OOM riskini sınırlar), PNG → base64 data URI döndür.
        with Image.open(file_path) as img:
            img = img.convert("RGB")
            w, h = img.size

            if w * h > _VLM_MAX_PIXELS:
                scale = (_VLM_MAX_PIXELS / (w * h)) ** 0.5
                new_w = max(_PATCH_SIZE, (int(w * scale) // _PATCH_SIZE) * _PATCH_SIZE)
                new_h = max(_PATCH_SIZE, (int(h * scale) // _PATCH_SIZE) * _PATCH_SIZE)
                img = img.resize((new_w, new_h), Image.LANCZOS)
                old_tok = (w * h) // (_PATCH_SIZE**2)
                new_tok = (new_w * new_h) // (_PATCH_SIZE**2)
                log.debug(
                    f"      Boyutlandırıldı: {w}×{h} → {new_w}×{new_h} "
                    f"(~{old_tok} → ~{new_tok} vision token)"
                )
            else:
                tok = (w * h) // (_PATCH_SIZE**2)
                log.debug(f"      Boyut: {w}×{h} (~{tok} vision token)")

            buf = io.BytesIO()
            img.save(buf, format="PNG", optimize=True)
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        return f"data:image/png;base64,{b64}"

    @staticmethod
    def _build_prompt() -> str:
        return (
            "Bu görseli analiz et.\n\n"
            "Görselin içeriğine göre:\n"
            "- Metin varsa tamamını Markdown olarak eksiksiz çıkar.\n"
            "- Tablo varsa Markdown tablosu yap. Başlık TEK satır olsun, çok "
            "satırlı başlık yapma: üst başlığı kapsadığı her alt sütuna dağıtıp "
            "üst+alt'ı tek isimde birleştir. Her satır — başlık dahil — aynı sayıda "
            "sütun içersin; boş göz varsa hücreyi boş bırak, kaydırma. Tablo "
            "içine not yazma.\n"
            "- Şema veya diyagram varsa yapısını, bileşenlerini ve akışını anlat.\n"
            "- Grafik veya chart varsa eksen etiketlerini, değerleri oku ve trendi yorumla.\n"
            "- Fotoğraf veya görsel sahne varsa içeriği detaylıca betimle.\n"
            "- Kategori dışı kalıyorsa görselden kısaca bahset.\n\n"
            "Tüm metin, sayı, etiket ve formülleri eksiksiz aktar.\n"
            "Teknik terimler ve yabancı etiketler orijinal dilinde kalsın.\n"
            "Emin olmadığın yerlerde 'muhtemelen' de, uydurma.\n"
            "Sözde kod yazma.\n\n"
            "'[ANALİZ_BİTTİ]' ile bitir."
        )

    def extract_text(self, image_path: str) -> str:
        """
        Tek bir görseli analiz edip Markdown metin döndürür.

        Görsel okunamaz, hazırlanamaz veya model boş/hatalı cevap dönerse boş
        string döner — çağıran (parser) bunu "içerik yok" olarak ele alıp ilgili
        görsel referansını metinden düşürür. Hata fırlatmaz, ingestion'ı bölmez.
        """
        if not os.path.exists(image_path):
            log.warning(f"Görsel bulunamadı: {image_path}")
            return ""

        img_name = os.path.basename(image_path)
        log.info(f"Görsel analiz ediliyor: {img_name}")

        prep_start = time.time()
        try:
            data_uri = self._prepare_image(image_path)
        except Exception as e:
            log.error(f"Görsel hazırlanamadı ({img_name}): {e}", exc_info=True)
            return ""
        log.debug(f"      Hazırlama: {time.time() - prep_start:.2f}s")

        prompt = self._build_prompt()

        try:
            inf_start = time.time()
            response = self.llm.create_chat_completion(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Sen bir görsel analiz asistanısın. "
                            "Sadece isteneni yap, ekstra yorum ekleme. "
                            "Gereksiz bilgi tekrarı yapma."
                        ),
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": data_uri}},
                            {"type": "text", "text": prompt},
                        ],
                    },
                ],
                max_tokens=AppConfig.VLM_MAX_TOKENS,
                # Qwen3-VL (ZwZ baz modeli) GREEDY'de sonsuz tekrara düşer; model
                # kartı ve Qwen resmi rehberi greedy'i yasaklıyor. Çözüm: sampling'i
                # açıp top_k/top_p ile odaklı tutmak — tekrar döngüsünü kaynağında
                # kırar. Önceki repeat_penalty + DRY yamaları kaldırıldı: bunlar
                # greedy'i zorlamanın yan etkisiydi ve modeli doğru token'dan
                # uzaklaştırıp tabloları bozuyordu. top_k=20 / top_p=0.8 instruct
                # reçetesi; net tablo hücrelerinde doğru değer baskın kaldığı için
                # sadakat korunur.
                temperature=AppConfig.VLM_TEMPERATURE,
                top_k=20,
                top_p=0.8,
                min_p=0.0,
                # present_penalty: Qwen'in instruct/VL benchmark ayarında 1.5;
                # düşük çözünürlüklü/okunaksız görsellerde tekrar için 1.8'e çekildi
                # (presence penalty). Tekrarı bastırmanın resmi yolu; yazarlar OCR
                # skorlarını bununla aldı. NOT: JamePeng fork'unda parametre adı
                # "present_penalty" (upstream'deki "presence_penalty" değil).
                # Çok yüksek olursa dil karışımı yapabilir, o noktada 1.0'a çekilir.
                present_penalty=1.8,
                # DRY (dizi-tekrar cezası): present_penalty token bazında çalışır;
                # DRY peş peşe tekrarlayan DİZİLERİ hedefler — okunaksız görsellerdeki
                # "aynı satırı tavana kadar tekrarla" döngüsü tam bu. Önceden greedy'de
                # ters tepmişti; artık düzgün sampling'de (temp 0.7) tasarlandığı gibi
                # çalışır. dry_penalty_last_n=-1 ŞART: fork'ta varsayılanı 0 ve 0 = DRY
                # kapalı. Seq breaker varsayılanı "\n" içerir, tablo satırlarını korur.
                dry_multiplier=0.8,
                dry_base=1.75,
                dry_allowed_length=2,
                dry_penalty_last_n=-1,
                stop=["[ANALİZ_BİTTİ]"],
            )
            inf_duration = time.time() - inf_start

            content = response["choices"][0]["message"]["content"].strip()
            content = content.replace("[ANALİZ_BİTTİ]", "").strip()

            # Çıktı uzunluğu ölçümü (geçici teşhis): decode süresinin asıl
            # maliyeti üretilen token sayısı. usage llama.cpp'den gelir;
            # completion_tokens = decode edilen token. Süre + token + tok/s
            # birlikte loglanınca 20 sn'lik görsellerin gerçek dolu içerik mi
            # yoksa şişkin/tekrarlı üretim mi olduğu ayırt edilebiliyor.
            usage = response.get("usage") or {}
            out_tok = usage.get("completion_tokens")
            in_tok = usage.get("prompt_tokens")
            tok_per_s = (out_tok / inf_duration) if out_tok and inf_duration else None
            log.debug(
                f"      Inference: {inf_duration:.2f}s | "
                f"prompt_tok: {in_tok} | output_tok: {out_tok} | "
                f"output_chars: {len(content)}"
                + (f" | {tok_per_s:.1f} tok/s" if tok_per_s else "")
            )

            if not content:
                log.warning(f"Model boş içerik döndürdü ({img_name})")
                return ""

            return content

        except Exception as e:
            log.error(f"Inference başarısız ({img_name}): {e}", exc_info=True)
            return ""

    def unload(self) -> None:
        """
        VLM'i VRAM'den serbest bırakır.

        llama-cpp-python C++ tabanlı, PyTorch kullanmıyor. del → Python
        refcount düşer → C destructor llama_free() ve clip_free() çağırır →
        GGML kendi CUDA buffer'larını boşaltır. torch.cuda.empty_cache()
        burada işe yaramaz; PyTorch'un memory pool'u zaten boş.
        """
        import gc

        log.info("VLM Motoru VRAM'den tahliye ediliyor...")
        # Sıra önemli: önce Llama (chat_handler'a referansı var), sonra handler
        if hasattr(self, "llm"):
            del self.llm
        if hasattr(self, "chat_handler"):
            del self.chat_handler
        gc.collect()
        log.info("VLM belleği temizlendi.")

    # Context manager protokolü — 'with VLMEngine() as vlm:' kullanımı için.
    # __exit__ exception olsa bile çalışır, VRAM sızıntısını önler.
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.unload()
        return False  # Exception'ı yutmasın, yukarı propagate olsun
