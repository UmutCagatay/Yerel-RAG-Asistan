"""
Sistem / verim metrikleri.

Üç bölüm:
  1) OFFLINE — app_base.log'dan ingestion süreleri + mevcut JSON'lardan
     sorgu gecikmeleri derlenir (yeniden çalıştırma YOK).
  2) CANLI BELLEK — arka planda nvidia-smi (VRAM) + psutil (RAM) örneklenirken:
       a) örnek bir PDF GEÇİCİ bir DB dizinine ingest edilir (VLM+embed tepe
          belleği), sonra geçici dizin silinir. Gerçek/dondurulmuş DB'ye
          DOKUNULMAZ.
       b) gerçek DB'de birkaç sorgu çalıştırılır (salt-okuma) → sorgu tepe
          belleği (retriever + LLM).
  3) TOKEN/SN — LLM bir kez yüklenip cevapların token sayısı ölçülür, üretim
     sürelerine bölünerek gerçek üretim hızı hesaplanır.

Çekirdeğe DOKUNMAZ; sadece import eder ve geçici dizin kullanır.

Kullanım (proje venv'i aktifken):
    cd C:\\Yerel_RAG_Asistan\\evaluation
    python run_system_metrics.py

Not: 2. ve 3. bölüm GPU kullanır ve birkaç dakika sürer. nvidia-smi şart;
psutil yoksa RAM atlanır (pip install psutil ile eklenebilir).
"""

import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from statistics import mean, median

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = PROJECT_ROOT / "backend"
sys.path.insert(0, str(BACKEND_DIR))

EVAL_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EVAL_DIR / "results"
LOG_PATH = BACKEND_DIR / "data" / "logs" / "app_base.log"
RETRIEVAL_JSON = RESULTS_DIR / "retrieval_raw.json"
GENERATION_JSON = RESULTS_DIR / "generation_answers.json"
DATASET_JSON = EVAL_DIR / "degerlendirme_seti.json"

TMP_DB_DIR = EVAL_DIR / "_metrics_tmp_db"      # geçici, sonra silinir
SAMPLE_PDF = PROJECT_ROOT / "test_dokumanlari" / "test1.pdf"  # ingest örneği (görselli, orta boy)
QUERY_SAMPLE_IDS = [1, 19, 10]                 # metin / gorsel / cok_dokuman

try:
    import psutil
    _PROC = psutil.Process(os.getpid())
    HAVE_PSUTIL = True
except Exception:
    HAVE_PSUTIL = False


# ──────────────────────────────────────────────────────────────────────────
# Bellek örnekleyici (arka plan thread)
# ──────────────────────────────────────────────────────────────────────────
class MemorySampler:
    """nvidia-smi (VRAM, MB) ve psutil RSS (RAM, MB) periyodik örnekler, tepe tutar."""

    def __init__(self, interval: float = 0.5):
        self.interval = interval
        self._stop = threading.Event()
        self._thread = None
        self.vram_samples: list[int] = []
        self.ram_samples: list[float] = []

    @staticmethod
    def _vram_mb() -> int | None:
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            )
            return int(out.stdout.strip().splitlines()[0])
        except Exception:
            return None

    def _ram_mb(self) -> float | None:
        if not HAVE_PSUTIL:
            return None
        try:
            return _PROC.memory_info().rss / (1024 * 1024)
        except Exception:
            return None

    def _loop(self):
        while not self._stop.is_set():
            v = self._vram_mb()
            r = self._ram_mb()
            if v is not None:
                self.vram_samples.append(v)
            if r is not None:
                self.ram_samples.append(r)
            self._stop.wait(self.interval)

    def start(self):
        self._stop.clear()
        self.vram_samples.clear()
        self.ram_samples.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        time.sleep(self.interval * 2)  # baseline için birkaç örnek

    def stop(self) -> dict:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)
        return {
            "vram_peak_mb": max(self.vram_samples) if self.vram_samples else None,
            "vram_baseline_mb": self.vram_samples[0] if self.vram_samples else None,
            "ram_peak_mb": round(max(self.ram_samples), 1) if self.ram_samples else None,
            "ram_baseline_mb": round(self.ram_samples[0], 1) if self.ram_samples else None,
        }


# ──────────────────────────────────────────────────────────────────────────
# 1) OFFLINE derleme
# ──────────────────────────────────────────────────────────────────────────
def parse_ingestion_log(path: Path) -> dict:
    if not path.exists():
        return {"error": f"log bulunamadı: {path}"}
    text = path.read_text(encoding="utf-8", errors="ignore")

    # Parse tamamlandı: <ad> — <sayfa> sayfa, <node> node, <sn> sn.
    parse_re = re.compile(
        r"Parse tamamlandı: (.+?) — (\d+) sayfa, (\d+) node, ([\d.]+) sn"
    )
    vlm_re = re.compile(r"Toplam VLM işlem süresi: ([\d.]+) sn")
    write_re = re.compile(
        r"(.+?) yazıldı: (\d+) child, (\d+) section, ([\d.]+) sn"
    )
    total_re = re.compile(r"Ingestion tamamlandı \(([\d.]+) sn\)")
    vlm_load_re = re.compile(r"VLM Motoru .*? yüklendi \(([\d.]+) sn\)")
    jina_load_re = re.compile(r"Jina V5 Nano ONNX yüklendi \(([\d.]+) sn\)")

    docs = []
    for m in parse_re.finditer(text):
        docs.append({
            "dosya": m.group(1).strip(),
            "sayfa": int(m.group(2)),
            "node": int(m.group(3)),
            "parse_sn": float(m.group(4)),
        })
    vlm_times = [float(x) for x in vlm_re.findall(text)]
    writes = [float(m.group(4)) for m in write_re.finditer(text)]
    total = [float(x) for x in total_re.findall(text)]

    parse_sn = [d["parse_sn"] for d in docs]
    return {
        "dokuman_sayisi": len(docs),
        "toplam_ingestion_sn": total[-1] if total else None,
        "toplam_parse_sn": round(sum(parse_sn), 1),
        "toplam_vlm_sn": round(sum(vlm_times), 1),
        "toplam_yazma_sn": round(sum(writes), 2),
        "parse_sn_ortalama": round(mean(parse_sn), 1) if parse_sn else None,
        "parse_sn_medyan": round(median(parse_sn), 1) if parse_sn else None,
        "parse_sn_min": round(min(parse_sn), 1) if parse_sn else None,
        "parse_sn_max": round(max(parse_sn), 1) if parse_sn else None,
        "vlm_yukleme_sn": float(vlm_load_re.findall(text)[0]) if vlm_load_re.findall(text) else None,
        "jina_yukleme_sn": float(jina_load_re.findall(text)[0]) if jina_load_re.findall(text) else None,
        "dokumanlar": docs,
    }


def _stats(vals: list[float]) -> dict:
    vals = [v for v in vals if v is not None]
    if not vals:
        return {}
    return {
        "ort": round(mean(vals), 1),
        "medyan": round(median(vals), 1),
        "min": round(min(vals), 1),
        "max": round(max(vals), 1),
        "n": len(vals),
    }


def parse_query_latency() -> dict:
    out = {}
    if RETRIEVAL_JSON.exists():
        recs = json.loads(RETRIEVAL_JSON.read_text(encoding="utf-8"))["records"]
        out["arama_ms"] = _stats([r["timings"]["search_ms"] for r in recs])
        out["reranker_ms"] = _stats([r["timings"]["reranker_ms"] for r in recs])
    if GENERATION_JSON.exists():
        recs = json.loads(GENERATION_JSON.read_text(encoding="utf-8"))["records"]
        out["uretim_ms"] = _stats(
            [r["timings"]["generation_ms"] for r in recs
             if r["timings"]["generation_ms"] > 0]
        )
    return out


# ──────────────────────────────────────────────────────────────────────────
# 2a) Canlı ingest belleği (geçici DB)
# ──────────────────────────────────────────────────────────────────────────
def measure_ingestion_memory() -> dict:
    from core.ingestion_engine import IngestionEngine

    if not SAMPLE_PDF.exists():
        return {"error": f"örnek PDF yok: {SAMPLE_PDF}"}

    # Geçici dizini temizle (önceki kalıntı varsa)
    if TMP_DB_DIR.exists():
        shutil.rmtree(TMP_DB_DIR, ignore_errors=True)
    TMP_DB_DIR.mkdir(parents=True, exist_ok=True)

    sampler = MemorySampler()
    sampler.start()
    t = time.time()
    try:
        ie = IngestionEngine(persist_dir=str(TMP_DB_DIR))
        ie.run([str(SAMPLE_PDF)], collection_name="metrics_tmp", use_vlm=True)
    finally:
        dur = time.time() - t
        mem = sampler.stop()
        # Geçici DB'yi sil — gerçek veriye dokunmaz
        shutil.rmtree(TMP_DB_DIR, ignore_errors=True)

    mem["sure_sn"] = round(dur, 1)
    mem["ornek_pdf"] = SAMPLE_PDF.name
    return mem


# ──────────────────────────────────────────────────────────────────────────
# 2b) Canlı sorgu belleği (gerçek DB, salt-okuma)
# ──────────────────────────────────────────────────────────────────────────
def measure_query_memory() -> dict:
    from core.query_engine import QueryEngine

    ds = {d["id"]: d for d in json.loads(DATASET_JSON.read_text(encoding="utf-8"))}
    sorular = [ds[i]["soru"] for i in QUERY_SAMPLE_IDS if i in ds]

    sampler = MemorySampler()
    sampler.start()
    wall = []
    try:
        qe = QueryEngine(collection_name="default")
        for s in sorular:
            t = time.time()
            qe.run(s, file_names=None)
            wall.append(round(time.time() - t, 1))
    finally:
        mem = sampler.stop()

    mem["sorgu_sayisi"] = len(sorular)
    mem["sorgu_toplam_sure_sn"] = wall  # her sorgu uçtan uca (model yükleme dahil)
    return mem


# ──────────────────────────────────────────────────────────────────────────
# 3) Token/saniye (LLM bir kez yüklenir, cevaplar tokenize edilir)
# ──────────────────────────────────────────────────────────────────────────
def measure_token_per_sec() -> dict:
    if not GENERATION_JSON.exists():
        return {"error": "generation_answers.json yok"}
    recs = json.loads(GENERATION_JSON.read_text(encoding="utf-8"))["records"]

    from core.llm_engine import LLMEngine
    rates = []
    sampler = MemorySampler()
    sampler.start()
    try:
        llm = LLMEngine()
        # LLMEngine icindeki LangChain nesnesi llm.llm; llama_cpp ise llm.llm.client
        client = getattr(getattr(llm, "llm", None), "client", None)
        for r in recs:
            gen_ms = r["timings"]["generation_ms"]
            ans = r["system_answer"]
            if gen_ms <= 0 or not ans or client is None:
                continue
            try:
                n_tok = len(client.tokenize(ans.encode("utf-8"),
                                            add_bos=False, special=False))
            except Exception:
                continue
            rates.append(n_tok / (gen_ms / 1000.0))
        llm.unload()
    finally:
        mem = sampler.stop()

    return {
        "token_per_sec": _stats(rates),
        "llm_vram_peak_mb": mem["vram_peak_mb"],
        "llm_ram_peak_mb": mem["ram_peak_mb"],
    }


# ──────────────────────────────────────────────────────────────────────────
def main() -> int:
    RESULTS_DIR.mkdir(exist_ok=True)
    out = {}

    print("[1/4] Offline derleme (log + JSON)...")
    out["ingestion_sureleri"] = parse_ingestion_log(LOG_PATH)
    out["sorgu_gecikmeleri"] = parse_query_latency()

    print("[2/4] İngest tepe belleği (geçici DB)...")
    out["ingest_bellek"] = measure_ingestion_memory()

    print("[3/4] Sorgu tepe belleği (gerçek DB, salt-okuma)...")
    out["sorgu_bellek"] = measure_query_memory()

    print("[4/4] Token/saniye + LLM belleği...")
    out["uretim_hizi"] = measure_token_per_sec()

    out_path = RESULTS_DIR / "system_metrics.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    # ── Özet ──
    ing = out["ingestion_sureleri"]
    lat = out["sorgu_gecikmeleri"]
    print("\n" + "=" * 60)
    print("  SİSTEM / VERİM METRİKLERİ ÖZETİ")
    print("=" * 60)
    print(f"  İNGESTION ({ing.get('dokuman_sayisi')} doküman):")
    print(f"    Toplam süre        : {ing.get('toplam_ingestion_sn')} sn")
    print(f"    Bunun VLM payı     : {ing.get('toplam_vlm_sn')} sn")
    print(f"    Parse/doküman      : ort {ing.get('parse_sn_ortalama')} / "
          f"medyan {ing.get('parse_sn_medyan')} / max {ing.get('parse_sn_max')} sn")
    ib = out["ingest_bellek"]
    print(f"    Tepe VRAM          : {ib.get('vram_peak_mb')} MB "
          f"(baz {ib.get('vram_baseline_mb')} MB)")
    print(f"    Tepe RAM           : {ib.get('ram_peak_mb')} MB")
    print("  " + "-" * 56)
    print("  SORGU:")
    if "arama_ms" in lat:
        print(f"    Arama       : ort {lat['arama_ms'].get('ort')} ms")
        print(f"    Reranker    : ort {lat['reranker_ms'].get('ort')} ms")
    if "uretim_ms" in lat:
        print(f"    Üretim      : ort {lat['uretim_ms'].get('ort')} ms "
              f"(medyan {lat['uretim_ms'].get('medyan')})")
    sb = out["sorgu_bellek"]
    print(f"    Tepe VRAM   : {sb.get('vram_peak_mb')} MB "
          f"(baz {sb.get('vram_baseline_mb')} MB)")
    print(f"    Tepe RAM    : {sb.get('ram_peak_mb')} MB")
    uh = out["uretim_hizi"]
    if "token_per_sec" in uh and uh["token_per_sec"]:
        print(f"    Üretim hızı : ort {uh['token_per_sec'].get('ort')} token/sn "
              f"(medyan {uh['token_per_sec'].get('medyan')})")
    print("  " + "-" * 56)
    print(f"  8GB VRAM sınırı; tepe kullanım yukarıda. psutil: "
          f"{'var' if HAVE_PSUTIL else 'YOK (RAM atlandı, pip install psutil)'}")
    print(f"\n  Ham sonuç: {out_path}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
