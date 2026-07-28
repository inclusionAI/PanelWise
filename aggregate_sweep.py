"""读取 output/sweep/*/summary_*.json，打印跨模型对比表。"""
import glob
import json
import os

rows = []
for path in glob.glob("output/sweep/*/summary_*.json"):
    try:
        s = json.load(open(path))
    except Exception:
        continue
    rows.append(s)

rows.sort(key=lambda s: s.get("normalized_mean", 0), reverse=True)

print(f"\n{'model':34s} {'norm':>6s} {'pass':>6s} {'scored':>7s} {'fail':>5s} {'cost$':>7s} {'sec':>6s}")
print("-" * 80)
for s in rows:
    print(
        f"{s['agent_model']:34s} "
        f"{s.get('normalized_mean',0):6.1f} "
        f"{s.get('pass_rate_mean',0):6.1f} "
        f"{s.get('n_scored',0):7d} "
        f"{s.get('n_agent_failed',0):5d} "
        f"{s.get('cost',{}).get('total_cost_usd',0):7.2f} "
        f"{s.get('elapsed_s',0):6.0f}"
    )

print("\n按领域 normalized（行=模型，列=领域）：")
domains = sorted({d for s in rows for d in s.get("by_domain", {})})
hdr = "model".ljust(30) + "".join(f"{d[:10]:>11s}" for d in domains)
print(hdr)
for s in rows:
    line = s["agent_model"].split("/")[-1][:29].ljust(30)
    for d in domains:
        v = s.get("by_domain", {}).get(d, {}).get("normalized_mean")
        line += f"{'-' if v is None else f'{v:.0f}':>11s}"
    print(line)

total = sum(s.get("cost", {}).get("total_cost_usd", 0) for s in rows)
print(f"\n横扫总成本: ${total:.2f}  (模型数: {len(rows)})")
