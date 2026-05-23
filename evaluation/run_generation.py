"""
3a — Cevap üretimi (yerel, GPU, internet YOK).

Gerçek sorgu hattını birebir çalıştırır:
    bağlam = retriever.get_relevant_context(soru, top_n, file_names=None)
    cevap  = llm.generate_answer(bağlam, soru)

QueryEngine.run() ile aynı akış; tek fark, verimlilik için iki FAZ:
    Faz A — retriever bir kez yüklenir, 42 sorunun bağlamı toplanır, unload
    Faz B — LLM bir kez yüklenir, 42 cevap üretilir, unload
Böylece LLM 42 kez değil 1 kez yüklenir. Retriever (CPU) ve LLM (GPU)
aynı anda bellekte durmaz — 8GB VRAM kısıtına uygun.

Çekirdeğe DOKUNMAZ; sadece import eder.

Çıktı: evaluation/results/generation_answers.json
       (cevaplar + bağlam + süreler — 3b hakem bunu tüketecek)

Kullanım (proje venv'i aktifken):
    cd C:\\Yerel_RAG_Asistan\\evaluation
    python run_generation.py
    python run_generation.py <koleksiyon_adi>

NOT: 42 cevap üretimi GPU'da birkaç dakika sürer (model her soruda
değil, bir kez yüklenir).
"""

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = PROJECT_ROOT / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from core.config import AppConfig  # noqa: E402
from core.llm_engine import LLMEngine  # noqa: E402
from core.retriever import RetrieverEngine  # noqa: E402

DATASET_PATH = Path(__file__).resolve().parent / "degerlendirme_seti.json"
OUTPUT_DIR = Path(__file__).resolve().parent / "results"

# QueryEngine bağlam boş kalınca bu mesajı döndürüyor — birebir taklit.
EMPTY_CONTEXT_MSG = "Bu doküman için sorguya uygun bir bağlam bulunamadı."


class AnswerGenerator:
    def __init__(self, collection: str = "default"):
        self.collection = collection
        self.top_n = AppConfig.RERANKER_TOP_N
        with open(DATASET_PATH, "r", encoding="utf-8") as f:
            self.dataset = json.load(f)

    def run(self) -> None:
        OUTPUT_DIR.mkdir(exist_ok=True)
        n = len(self.dataset)
        print(f"Koleksiyon: {self.collection} | {n} soru\n")

        # ── FAZ A: Bağlam toplama (retriever, CPU) ──
        print("[FAZ A] Retriever yükleniyor, bağlamlar toplanıyor...")
        contexts: dict[int, str] = {}
        retr_ms: dict[int, float] = {}
        with RetrieverEngine(collection_name=self.collection) as ret:
            for item in self.dataset:
                qid = item["id"]
                t = time.time()
                ctx = ret.get_relevant_context(
                    query=item["soru"],
                    top_n=self.top_n,
                    file_names=None,  # tüm koleksiyon
                )
                retr_ms[qid] = (time.time() - t) * 1000
                contexts[qid] = ctx
                durum = f"{len(ctx)} karakter" if ctx else "BOŞ"
                print(f"  S{qid:>2} bağlam: {durum}")
        # retriever burada otomatik unload
        print("[FAZ A] Bitti. Retriever bellekten indi.\n")

        # ── FAZ B: Cevap üretimi (LLM, GPU) ──
        print("[FAZ B] LLM yükleniyor, cevaplar üretiliyor (sürebilir)...")
        records: list[dict] = []
        with LLMEngine() as llm:
            for item in self.dataset:
                qid = item["id"]
                ctx = contexts[qid]

                if not ctx:
                    answer = EMPTY_CONTEXT_MSG
                    gen_ms = 0.0
                else:
                    t = time.time()
                    answer = llm.generate_answer(context=ctx, question=item["soru"])
                    gen_ms = (time.time() - t) * 1000

                records.append(
                    {
                        "id": qid,
                        "soru_tipi": item["soru_tipi"],
                        "soru": item["soru"],
                        "ideal_cevap": item["ideal_cevap"],
                        "kaynak_dosya": item.get("kaynak_dosya", ""),
                        "context": ctx,  # 3b faithfulness için gerekli
                        "system_answer": answer,
                        "timings": {
                            "retrieval_ms": round(retr_ms[qid], 1),
                            "generation_ms": round(gen_ms, 1),
                        },
                    }
                )

                onizleme = answer.replace("\n", " ")[:90]
                print(
                    f"  S{qid:>2} [{item['soru_tipi']:<11}] "
                    f"{gen_ms:>6.0f}ms | {onizleme}..."
                )
        # LLM burada otomatik unload
        print("[FAZ B] Bitti. LLM bellekten indi.\n")

        # ── Kayıt ──
        out_path = OUTPUT_DIR / "generation_answers.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"records": records}, f, ensure_ascii=False, indent=2)

        avg_gen = sum(
            r["timings"]["generation_ms"]
            for r in records
            if r["timings"]["generation_ms"] > 0
        ) / max(1, sum(1 for r in records if r["timings"]["generation_ms"] > 0))

        print("=" * 60)
        print(f"  {len(records)} cevap üretildi.")
        print(f"  Ortalama üretim süresi: {avg_gen:.0f} ms")
        print(f"  Kayıt: {out_path}")
        print("  Sıradaki: cevapları gözle kontrol et, sonra 3b (hakem).")
        print("=" * 60)


def main() -> int:
    collection = sys.argv[1] if len(sys.argv) > 1 else "default"
    AnswerGenerator(collection=collection).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
