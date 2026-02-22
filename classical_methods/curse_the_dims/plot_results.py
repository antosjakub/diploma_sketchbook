

import json
import matplotlib.pyplot as plt


with open("results/timings_n=64,d=2.json") as f:
    data = json.load(f)


timer_report = data['inner_timers']

labels = list(timer_report.keys())
#times  = [timer_report[k]["time_ns"] / 1e9 for k in labels]
times  = [timer_report[k]["allocated_bytes"] / 1e6 for k in labels]

fig, ax = plt.subplots()
bars = ax.barh(labels, times)
ax.bar_label(bars, fmt="%.4f s")
ax.set_xlabel("Time (s)")
ax.set_title("Timer breakdown")
plt.tight_layout()
plt.savefig("timings.png", dpi=150)