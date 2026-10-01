"""Usage: run_bench.py ours|strands|typesafe  -> results_<system>.json"""
import json, statistics as st, sys, time
sys.path.insert(0, ".")
from cases import *

system = sys.argv[1]
if system == "ours":
    sys.path.insert(0, "..")
    from src.truetype.engine import EngineConfig, GemmaLetterEngine
    from src.truetype.service import TypeSafeReplica
    t0 = time.time(); eng = GemmaLetterEngine(EngineConfig(top_k=5)); eng.load(); svc = TypeSafeReplica(eng)
    load_s = time.time() - t0
    def ask(state, questions):
        return svc.system_one(state=state, questions=questions).answers
elif system == "typesafe":  # hosted Jev API; latency includes network round trip
    import os
    import urllib.request
    key = os.environ["TYPESAFE_API_KEY"]
    load_s = 0.0
    def ask(state, questions):  # no `temperature` field: the hosted API rejects it (HTTP 400)
        body = json.dumps({"state": state, "model": "jev-latest", "questions": questions}).encode()
        req = urllib.request.Request("https://api.typesafe.ai/v1/systemone", data=body, headers={
            "Authorization": f"Bearer {key}", "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.load(r)["answers"]
else:
    import torch
    from strands_decider.infer import load_engine
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    t0 = time.time(); eng = load_engine("StrandsAgents/strands-decider-2B-hobson-v19", device=dev); load_s = time.time() - t0
    from strands_decider.schema import SystemOneRequest
    def ask(state, questions):
        r = eng.evaluate(SystemOneRequest(state=state, questions=questions))
        return {k: v.model_dump() for k, v in r.answers.items()}

def timed(state, qs):
    t = time.perf_counter(); a = ask(state, qs); return a, (time.perf_counter() - t) * 1000

def sync():
    import torch
    if torch.backends.mps.is_available(): torch.mps.synchronize()

ask("warmup", {"q": noul_q("Is this a warmup?")}); ask("warmup", {"c": choice_q()})  # warm
res = {"system": system, "load_s": load_s, "noul": [], "choice": [], "score": []}
lat = []
for s, q, exp in NOUL:
    a, ms = timed(s, {"q": noul_q(q)}); p = a["q"]["noul"]; lat.append(ms)
    res["noul"].append({"p": p, "ok": (p >= .5) == exp, "brier": (p - exp) ** 2})
for s, exp in CHOICE:
    a, ms = timed(s, {"q": choice_q()}); x = a["q"]; lat.append(ms)
    res["choice"].append({"pred": x["choice"], "exp": exp, "ok": x["choice"] == exp, "conf": x["confidence"], "p_exp": x["probabilities"].get(exp, 0)})
for s, exp in SCORE:
    a, ms = timed(s, {"q": score_q()}); x = a["q"]; lat.append(ms)
    sc = x["score"]
    res["score"].append({"score": sc, "exp": exp, "ok": round(sc) == exp, "abs_err": abs(sc - exp), "conf": x["confidence"]})
res["single_q_latency_ms"] = {"p50": st.median(lat), "p95": sorted(lat)[int(len(lat) * .95) - 1], "mean": st.mean(lat), "n": len(lat)}
# 3-question batched request
s = "I was charged twice for order A-104 and this blocks all payments."
bl = []
for _ in range(10):
    _, ms = timed(s, {"a": noul_q("Does the text report a billing problem?"), "b": choice_q(), "c": score_q()}); bl.append(ms)
res["three_q_latency_ms"] = {"p50": st.median(bl), "max": max(bl)}
json.dump(res, open(f"results_{system}.json", "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k not in ("noul", "choice", "score")}, indent=1))
for k in ("noul", "choice", "score"):
    print(k, "acc", sum(r["ok"] for r in res[k]), "/", len(res[k]))
