"""对比 plugin（单轮，×3 rejudge）vs server-tool（agentic，×3）的分数提升。"""
import glob
import json
import os

NATIVE = [
    "google/gemini-3.1-pro-preview",
    "openai/gpt-5.5",
    "anthropic/claude-opus-4.8",
    "x-ai/grok-4.3",
]


def load_plugin(model):
    d = "output/sweep/" + model.replace("/", "_")
    p = os.path.join(d, "summary_rejudged.json")
    return json.load(open(p)) if os.path.exists(p) else None


def load_st(model):
    d = "output/sweep/" + model.replace("/", "_") + "__st"
    fs = glob.glob(os.path.join(d, "summary_*.json"))
    fs = [f for f in fs if "rejudged" not in f]
    return json.load(open(fs[0])) if fs else None


print(f"\n{'model':30s} {'plugin':>8s} {'server-tool':>12s} {'Δ':>7s}  {'scored(p/st)':>12s}")
print("-" * 78)
for m in NATIVE:
    p = load_plugin(m)
    s = load_st(m)
    pn = p["normalized_mean"] if p else None
    sn = s["normalized_mean"] if s else None
    delta = (sn - pn) if (pn is not None and sn is not None) else None
    sc = f"{p['n_scored'] if p else '-'}/{s['n_scored'] if s else '-'}"
    print(
        f"{m:30s} "
        f"{('%.1f'%pn) if pn is not None else '-':>8s} "
        f"{('%.1f'%sn) if sn is not None else '-':>12s} "
        f"{('%+.1f'%delta) if delta is not None else '-':>7s}  "
        f"{sc:>12s}"
    )

print("\n按领域 normalized（plugin → server-tool）：")
domains = ["Academic","Finance","General Knowledge","Law","Medicine",
           "Needle in a Haystack","Personalized Assistant",
           "Shopping/Product Comparison","Technology","UX Design"]
for m in NATIVE:
    p, s = load_plugin(m), load_st(m)
    print(f"\n{m}")
    for d in domains:
        pv = (p or {}).get("by_domain", {}).get(d, {}).get("normalized_mean")
        sv = (s or {}).get("by_domain", {}).get(d, {}).get("normalized_mean")
        if pv is None and sv is None:
            continue
        ps = "-" if pv is None else f"{pv:.0f}"
        ss = "-" if sv is None else f"{sv:.0f}"
        print(f"  {d:28s} {ps:>4s} -> {ss:>4s}")
