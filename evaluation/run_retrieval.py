"""
Retrieval ölçümü (A+): reranker sonrası (asıl metrik) + ham vektör araması
(teşhis) birlikte ölçülür.

Çekirdeğe DOKUNMAZ. Projenin gerçek RetrieverEngine'ini import edip senin
arama+reranker hattını birebir çalıştırır. get_relevant_context yalnızca
metin döndürdüğü (dosya adı vermediği) için, ölçüm tarafında arama+rerank
adımını AYNI şekilde tekrarlayıp her sonucun file_name'ini okuruz.

Ground truth dosya seviyesinde (set chunk değil 'kaynak_dosya' veriyor),
dolayısıyla metrik de "doğru DOSYADAN bir parça geldi mi, kaçıncı sırada"
mantığında. Tek doğru dosya olduğu için recall@k = hit@k'dir.

Ölçülenler:
  • reranked (asıl)  : Hit@1, Hit@3, MRR  → LLM'e giden top-3 buradan
  • ham vektör (teşhis): doğru dosya top-10'da mıydı, kaçıncı sırada
  • reranker etkisi    : doğru dosyayı top-3'e taşıdı mı / düşürdü mü
  • tip kırılımı       : metin / gorsel / cok_dokuman ayrı ayrı
  • cevapsiz (5)       : skorlanmaz; reranker skorları bilgi olarak kaydedilir

Kullanım (proje venv'i aktifken):
    cd C:\\Yerel_RAG_Asistan\\evaluation
    python run_retrieval.py
    python run_retrieval.py <koleksiyon_adi>

Çıktılar evaluation/results/ altına yazılır.
"""

import json
import sys
import time
from pathlib import Path

# Çekirdeği import edebilmek için backend/ dizinini path'e ekle.
# (backend kodu 'from core.x import ...' biçiminde import ediyor.)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = PROJECT_ROOT / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from core.config import AppConfig          # noqa: E402
from core.retriever import RetrieverEngine  # noqa: E402

DATASET_PATH = Path(__file__).resolve().parent / "degerlendirme_seti.json"
OUTPUT_DIR = Path(__file__).resolve().parent / "results"

# Skorlanan tipler (kaynak_dosya dolu). cevapsiz hariç.
SCORED_TYPES = {"metin", "gorsel", "cok_dokuman"}


class RetrievalEvaluator:
    def __init__(self, collection: str = "default", top_n: int | None = None):
        self.collection = collection
        self.top_n = top_n if top_n is not None else AppConfig.RERANKER_TOP_N
        self.dataset = self._load_dataset()

    def _load_dataset(self) -> list[dict]:
        with open(DATASET_PATH, "r", encoding="utf-8") as f:
            return json.load(f)

    # ── Tek sorgu: arama + rerank, çekirdekteki akışın birebir aynısı ──
    def _retrieve(self, ret: RetrieverEngine, query: str) -> dict:
        """
        Tüm koleksiyon kapsamında (filtre yok) ham getirme + reranker.
        get_relevant_context'in 1-2. adımlarıyla aynı kod yolu.
        """
        t0 = time.time()
        raw_docs = ret.base_retriever.invoke(query)
        search_ms = (time.time() - t0) * 1000

        if not raw_docs:
            return {
                "raw_files": [],
                "reranked": [],
                "search_ms": search_ms,
                "reranker_ms": 0.0,
            }

        # Ham (vektör) sıralama — retriever'ın döndürdüğü sıra
        raw_files = [d.metadata.get("file_name") for d in raw_docs]

        # Reranker — çekirdekteki ile aynı: key=score (Document'ı karşılaştırmaz)
        t1 = time.time()
        scores = ret.reranker.rank(query, [d.page_content for d in raw_docs])
        reranker_ms = (time.time() - t1) * 1000

        scored = sorted(zip(scores, raw_docs), key=lambda x: x[0], reverse=True)
        reranked = [
            {"file": d.metadata.get("file_name"), "score": float(s)}
            for s, d in scored
        ]
        return {
            "raw_files": raw_files,
            "reranked": reranked,
            "search_ms": search_ms,
            "reranker_ms": reranker_ms,
        }

    @staticmethod
    def _rank_of(ordered_files: list, target: str) -> int | None:
        """target dosyasının listede ilk geldiği 1-tabanlı sıra; yoksa None."""
        if not target:
            return None
        for i, f in enumerate(ordered_files, start=1):
            if f == target:
                return i
        return None

    def run(self) -> None:
        OUTPUT_DIR.mkdir(exist_ok=True)
        records: list[dict] = []

        print(f"Koleksiyon: {self.collection} | top_n: {self.top_n}")
        print(f"{len(self.dataset)} soru işlenecek. Retriever yükleniyor...\n")

        # Context manager → çıkışta otomatik unload
        with RetrieverEngine(collection_name=self.collection) as ret:
            for item in self.dataset:
                qid = item["id"]
                soru = item["soru"]
                tip = item["soru_tipi"]
                target = item.get("kaynak_dosya", "")

                r = self._retrieve(ret, soru)
                reranked_files = [x["file"] for x in r["reranked"]]

                raw_rank = self._rank_of(r["raw_files"], target)
                rer_rank = self._rank_of(reranked_files, target)

                rec = {
                    "id": qid,
                    "soru_tipi": tip,
                    "soru": soru,
                    "kaynak_dosya": target,
                    "raw": {
                        "ordered_files": r["raw_files"],
                        "rank": raw_rank,  # doğru dosya ham aramada kaçıncı
                    },
                    "reranked": {
                        "ordered": r["reranked"],  # dosya + skor, top-10 sıralı
                        "rank": rer_rank,
                        "hit@1": rer_rank == 1,
                        "hit@3": rer_rank is not None and rer_rank <= self.top_n,
                    },
                    "timings": {
                        "search_ms": round(r["search_ms"], 1),
                        "reranker_ms": round(r["reranker_ms"], 1),
                    },
                }
                records.append(rec)

                # Kısa ilerleme satırı
                if tip in SCORED_TYPES:
                    mark = "✓" if rec["reranked"]["hit@3"] else "✗"
                    print(f"  S{qid:>2} [{tip:<11}] {mark} "
                          f"ham#{raw_rank} → rerank#{rer_rank}")
                else:
                    top = r["reranked"][0] if r["reranked"] else None
                    skor = f"{top['score']:+.3f}" if top else "—"
                    print(f"  S{qid:>2} [{tip:<11}] (skorlanmaz) "
                          f"top skor={skor}")

        metrics = self._compute_metrics(records)
        self._save(records, metrics)
        self._print_summary(metrics)

    # ── Metrik hesabı ──
    def _compute_metrics(self, records: list[dict]) -> dict:
        scored = [r for r in records if r["soru_tipi"] in SCORED_TYPES]

        def agg(subset: list[dict]) -> dict:
            n = len(subset)
            if n == 0:
                return {"n": 0}
            hit1 = sum(r["reranked"]["hit@1"] for r in subset)
            hit3 = sum(r["reranked"]["hit@3"] for r in subset)
            mrr = sum(
                (1.0 / r["reranked"]["rank"]) if r["reranked"]["rank"] else 0.0
                for r in subset
            )
            # Ham vektör aramasının recall'ı: doğru dosya top-10'da mıydı
            raw_hit = sum(1 for r in subset if r["raw"]["rank"] is not None)
            raw_mrr = sum(
                (1.0 / r["raw"]["rank"]) if r["raw"]["rank"] else 0.0
                for r in subset
            )
            return {
                "n": n,
                "hit@1": round(hit1 / n, 3),
                "hit@3": round(hit3 / n, 3),
                "mrr": round(mrr / n, 3),
                "ham_recall@10": round(raw_hit / n, 3),
                "ham_mrr": round(raw_mrr / n, 3),
            }

        # Reranker etkisi: doğru dosya ham top-10'da VAR ama reranked top-3'te YOK
        # (reranker düşürdü) — teşhis için kritik sayı.
        demoted = [
            r["id"] for r in scored
            if r["raw"]["rank"] is not None
            and not r["reranked"]["hit@3"]
        ]
        # Ham aramada hiç gelmeyen (vektör araması kaçırdı) — düzeltmesi farklı yer
        missed_by_search = [
            r["id"] for r in scored if r["raw"]["rank"] is None
        ]

        by_type = {
            t: agg([r for r in scored if r["soru_tipi"] == t])
            for t in sorted(SCORED_TYPES)
        }

        # Zamanlama ortalamaları (tüm sorular)
        avg_search = sum(r["timings"]["search_ms"] for r in records) / len(records)
        avg_rerank = sum(r["timings"]["reranker_ms"] for r in records) / len(records)

        # cevapsiz: skorlanmaz, top skorları bilgi olarak topla
        cevapsiz = [
            {"id": r["id"],
             "top": r["reranked"]["ordered"][0] if r["reranked"]["ordered"] else None}
            for r in records if r["soru_tipi"] == "cevapsiz"
        ]

        return {
            "genel": agg(scored),
            "tip_bazli": by_type,
            "reranker_dusurdu": demoted,
            "vektor_kacirdi": missed_by_search,
            "ortalama_search_ms": round(avg_search, 1),
            "ortalama_reranker_ms": round(avg_rerank, 1),
            "cevapsiz_bilgi": cevapsiz,
        }

    # ── Kayıt ──
    def _save(self, records: list[dict], metrics: dict) -> None:
        raw_path = OUTPUT_DIR / "retrieval_raw.json"
        with open(raw_path, "w", encoding="utf-8") as f:
            json.dump({"records": records, "metrics": metrics}, f,
                      ensure_ascii=False, indent=2)
        print(f"\nHam veri: {raw_path}")

    def _print_summary(self, m: dict) -> None:
        g = m["genel"]
        print("\n" + "=" * 64)
        print("  RETRIEVAL ÖZETİ (reranked = asıl, ham = teşhis)")
        print("=" * 64)
        print(f"  Skorlanan soru     : {g['n']}")
        print(f"  Hit@1              : {g['hit@1']}")
        print(f"  Hit@3 (LLM'e giden): {g['hit@3']}")
        print(f"  MRR                : {g['mrr']}")
        print(f"  Ham recall@10      : {g['ham_recall@10']}  (vektör araması)")
        print(f"  Ham MRR            : {g['ham_mrr']}")
        print("  " + "-" * 60)
        print("  Tip bazlı (hit@3 / mrr):")
        for t, s in m["tip_bazli"].items():
            if s["n"]:
                print(f"    {t:<12} n={s['n']:<2} "
                      f"hit@3={s['hit@3']} mrr={s['mrr']} "
                      f"(ham recall@10={s['ham_recall@10']})")
        print("  " + "-" * 60)
        print(f"  Reranker düşürdü (ham'da var, top-3'te yok): "
              f"{m['reranker_dusurdu'] or 'yok'}")
        print(f"  Vektör araması kaçırdı (ham'da hiç yok)   : "
              f"{m['vektor_kacirdi'] or 'yok'}")
        print("  " + "-" * 60)
        print(f"  Ort. arama   : {m['ortalama_search_ms']} ms")
        print(f"  Ort. reranker: {m['ortalama_reranker_ms']} ms")
        print("=" * 64)


def main() -> int:
    collection = sys.argv[1] if len(sys.argv) > 1 else "default"
    RetrievalEvaluator(collection=collection).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
