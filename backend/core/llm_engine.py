import logging
import os
import time

from core.config import AppConfig
from langchain_community.llms import LlamaCpp
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

log = logging.getLogger(__name__)


class LLMEngine:
    """
    Cevap üreten LLM sarmalayıcısı (Turkish-Gemma-9B, GGUF/llama.cpp, GPU).

    Bağlam + soruyu Gemma sohbet şablonuna yerleştirip cevap üretir; hem
    tek seferlik (generate_answer) hem token-token (generate_answer_stream)
    çalışır. Prompt, modeli sadece verilen bağlama dayanmaya zorlar
    (uydurma yok). 'with' bloğuyla kullanılıp çıkışta unload ile VRAM
    boşaltılır.
    """

    def __init__(self):
        self.model_path = str(AppConfig.LLM_MODEL_PATH)

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model dosyası bulunamadı: {self.model_path}")

        load_start = time.time()

        self.llm = LlamaCpp(
            model_path=self.model_path,
            temperature=AppConfig.LLM_TEMPERATURE,
            max_tokens=AppConfig.LLM_MAX_TOKENS,
            n_ctx=AppConfig.LLM_N_CTX,
            n_gpu_layers=-1,  # -1 = tüm katmanlar GPU'da (8GB'a Q4 9B sığıyor)
            n_batch=512,      # prompt'u 512'lik gruplar halinde işle (hız/bellek dengesi)
            repeat_penalty=1.1,
            verbose=False,
            # type_k / type_v LangChain'in bildiği parametreler değil.
            # model_kwargs içine koyunca LangChain uyarı vermeden
            # doğrudan llama.cpp'ye iletir. 8 = GGML_TYPE_Q8_0.
            # f16_kv=True yerine bu yöntem: KV cache belleği ~%50 düşer.
            model_kwargs={"type_k": 8, "type_v": 8},
        )

        prompt_text = """<start_of_turn>user
Sen verilen bağlama dayanarak cevap üreten bir asistansın.

TEMEL KURAL:
- Sadece bağlamdaki bilgileri kullan, dış bilgi (önceden eğitildiğin bilgiler) ekleme.
- Bağlam içinde sentez ve çıkarım yapabilirsin: farklı parçaları birleştirebilir, bilgiyi yeniden ifade edebilir, eldeki bilgilerden mantıksal sonuçlar çıkarabilirsin.
- Soruda geçen kelimelerin bağlamda birebir geçmesi gerekmez. Soruyla konu olarak yakın bilgi bağlamda varsa, ondan cevap üret.

NE ZAMAN CEVAP ÜRETME:
- Bağlam soruyla tamamen alakasızsa "Bu bilgiye sahip değilim." de.
- Bağlamda olmayan detayları ekleme, uydurma yapma.

GÖRSELLERDEN GELEN BİLGİLER:
- Bağlamda <VLM_START ...>...<VLM_END> etiketleri arasında gördüğün içerik, dokümandaki görsellerden (tablo, şema, grafik vb.) bir görsel modeli tarafından çıkarılmış metindir. Doğrudan dokümanın yazılı kısmı değildir.
- Bağlam (hem VLM blokları hem doğrudan dokümandan okunan metin/tablolar) yapısal hata içerebilir: bir sayı yanlış okunmuş, bir etiket atlanmış, sütun başlıkları veri hücreleriyle yanlış eşleşmiş ya da satır/sütun hizalaması kaymış olabilir. Bir tablo veya yapıdaki tutarsızlık fark edersen, içerikteki mantıksal ilişkilere bakarak hangi değerin hangi sütuna/kategoriye ait olduğunu çıkarsa.
- Bir VLM bloğunun ne anlattığı net değilse, aynı bağlam parçasındaki çevresindeki metne (başlık, üst/alt paragraflar) bakarak görselin orada neyi temsil ettiğini çıkarsamayı dene.

Bağlam:
{context}

Soru: {question}<end_of_turn>
<start_of_turn>model
"""

        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"], template=prompt_text
        )

        self.chain = self.prompt_template | self.llm | StrOutputParser()

        log.info(f"LLM Motoru (Gemma) yüklendi ({time.time() - load_start:.2f} sn).")

    def _fit_context_to_window(self, context: str, question: str) -> str:
        """
        Bağlamı, render edilen prompt + cevap payı LLM penceresine (n_ctx)
        sığacak şekilde kırpar. Taşma yoksa bağlamı aynen döndürür.

        Neden gerekli: section genişletme bazen (yoğun VLM/tablo bölümlerinde)
        çok büyük bağlam üretebiliyor; bu bağlam doğrudan LLM'e gidince
        'Requested tokens exceed context window' ile üretim çöküyordu. Burada
        evrensel bir güvenlik ağı kuruyoruz: sebebi ne olursa olsun bağlam
        pencereye sığar.

        Kırpma sondan yapılır (reranker en alakalı pasajları üste koyduğu için
        sonda kalan en düşük öncelikli kısımdır). Token bazlı; tokenizer
        erişilemezse ihtiyatlı bir karakter bütçesine düşülür.
        """
        budget = AppConfig.LLM_N_CTX - AppConfig.LLM_MAX_TOKENS - 256
        if budget <= 0:
            return context  # anlamsız config; dokunma

        marker = "\n\n[... bağlam, model penceresine sığması için kırpıldı ...]"
        client = getattr(self.llm, "client", None)

        # ── Token bazlı (tercih edilen) ──
        if client is not None:
            try:
                full_prompt = self.prompt_template.format(
                    context=context, question=question
                )
                total = len(
                    client.tokenize(
                        full_prompt.encode("utf-8"), add_bos=True, special=True
                    )
                )
                if total <= budget:
                    return context

                overflow = total - budget
                ctx_tokens = client.tokenize(
                    context.encode("utf-8"), add_bos=False, special=False
                )
                # overflow kadar + küçük tampon token'ı sondan at
                keep = max(0, len(ctx_tokens) - overflow - 32)
                trimmed = client.detokenize(ctx_tokens[:keep]).decode(
                    "utf-8", errors="ignore"
                )
                log.warning(
                    f"Bağlam pencereyi aşıyordu ({total} token > {budget} bütçe); "
                    f"sondan ~{overflow} token kırpıldı."
                )
                return trimmed + marker
            except Exception as e:
                log.error(
                    f"Token bazlı kırpma başarısız, karakter tabanlına düşülüyor: {e}",
                    exc_info=True,
                )

        # ── Karakter bazlı yedek (tokenizer erişilemezse) ──
        # Token-yoğun içerik (tablo/sayı) için ihtiyatlı ~2 karakter/token.
        char_budget = budget * 2
        if len(context) > char_budget:
            log.warning(
                f"Bağlam karakter bütçesini aşıyordu "
                f"({len(context)} > {char_budget}); kırpıldı."
            )
            return context[:char_budget] + marker
        return context

    def generate_answer(self, context: str, question: str) -> str:
        context = self._fit_context_to_window(context, question)
        try:
            return self.chain.invoke({"context": context, "question": question})
        except Exception as e:
            # ERROR seviyesi + exc_info: traceback dosyaya zengin biçimde gider,
            # kullanıcı terminalde kısa mesaj görür, hata kaybolmaz.
            log.error(f"LLM yanıt üretirken hata oluştu: {e}", exc_info=True)
            return f"LLM Yanıt Üretirken Hata Oluştu: {str(e)}"

    def generate_answer_stream(self, context: str, question: str):
        """
        Cevabı token token üreten generator versiyon.

        LangChain'in chain.stream() metodu generator döndürür; her bir
        parça (genelde 1-2 token) ortaya çıkar çıkmaz yield ile dışarı
        aktarılır. Çağıran taraf for döngüsü ile parçaları toplar veya
        HTTP stream'e yazar.
        """
        context = self._fit_context_to_window(context, question)
        try:
            for chunk in self.chain.stream({"context": context, "question": question}):
                # chunk her zaman string — StrOutputParser ile çıktı parse edildi.
                # Boş chunk olabilir, frontend'e yollamak anlamsız.
                if chunk:
                    yield chunk
        except Exception as e:
            log.error(f"LLM stream sırasında hata: {e}", exc_info=True)
            yield f"\n\n[HATA] LLM yanıt üretirken sorun oluştu: {e}"

    def unload(self):
        """
        LLM'i VRAM'den serbest bırakır.

        LangChain LlamaCpp arkada llama-cpp-python kullanıyor — C++ tabanlı,
        PyTorch değil. del + gc.collect() yeterli.
        """
        import gc

        log.info("LLM bellekten tahliye ediliyor...")
        if hasattr(self, "chain"):
            del self.chain
        if hasattr(self, "llm"):
            del self.llm
        gc.collect()
        log.info("LLM belleği temizlendi.")

    # Context manager protokolü — 'with LLMEngine() as llm:' kullanımı için.
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.unload()
        return False
