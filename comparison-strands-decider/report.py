import json, itertools
def pairwise(r):  # fraction of (lower,higher)-level pairs ordered correctly
    s=r["score"]; ok=n=0
    for a,b in itertools.combinations(s,2):
        if a["exp"]!=b["exp"]:
            n+=1; ok+= (a["score"]<b["score"])==(a["exp"]<b["exp"])
    return ok/n
def mae(r): return sum(x["abs_err"] for x in r["score"])/len(r["score"])
rows=[]
import os
SYS=[s for s in ("ours","strands","typesafe") if os.path.exists(f"results_{s}.json")]
for s in SYS:
    r=json.load(open(f"results_{s}.json"))
    rows.append((s,{
     "noul acc (/15)":sum(x["ok"] for x in r["noul"]),
     "noul Brier (lower better)":round(sum(x["brier"] for x in r["noul"])/15,5),
     "choice acc (/12)":sum(x["ok"] for x in r["choice"]),
     "choice mean P(correct)":round(sum(x["p_exp"] for x in r["choice"])/12,3),
     "score round-acc (/12)":sum(x["ok"] for x in r["score"]),
     "score pairwise ordering":round(pairwise(r),3),
     "score MAE (levels)":round(mae(r),3),
     "load s":round(r["load_s"],1),
     "1-q p50 ms":round(r["single_q_latency_ms"]["p50"]),
     "1-q p95 ms":round(r["single_q_latency_ms"]["p95"]),
     "3-q p50 ms":round(r["three_q_latency_ms"]["p50"])}))
NAMES={"ours":"ours (Gemma-4 12B, local)","strands":"Strands Decider 2B v19 (local)","typesafe":"TypeSafe Jev API (hosted)"}
keys=list(rows[0][1]); print("| metric | "+" | ".join(NAMES[s] for s,_ in rows)+" |\n|"+"---|"*(len(rows)+1))
for k in keys: print(f"| {k} | "+" | ".join(str(d[k]) for _,d in rows)+" |")
