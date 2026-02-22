
import os
import json

# in one plot, show:
# - diff float types = Float64,33
# - matrix L vs vector Ut
# - discret n=32,48,64
# which combinations enables us to go to dim=5?


timer_report = {}
for filename in os.listdir('results'):
    if not filename.endswith(".json"):
        continue
    
    with open(f'results/{filename}') as f:
        data = json.load(f)

    name = filename.split('.')[0]
    timer_report[name] = data['inner_timers']

print(f'Plotting for: {list(timer_report.keys())}')

x = []
y = []
for key, value in timer_report.items():
    t,n,d,ft = key.split(',')
    n = int(n.split('=')[1])
    d = int(d.split('=')[1])
    ft = int(ft.split('=')[1])
    if t == 'sparse' and n == 64 and ft == 64:
        x.append(d)
        y.append(timer_report[key]["init Ut"]["allocated_bytes"])
import numpy as np
x = np.array(x)
y = np.array(y)
indices = np.argsort(x)
x = x[indices]
y = y[indices]
print(x)
print(y)

byte_names = ['B', 'kB', 'MB', 'GB', 'TB']
powers = np.arange(4,10+1, dtype=int)
values = 10**(powers % 3)
labels = np.array(byte_names)[((powers // 3))]
y_ticks = []
for l, v in zip(labels, values):
    y_ticks.append(f'{str(v)} {l}')


import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.grid(visible=True)
ax.set_xticks(x)
ax.set_xlabel("d (dimension)")
ax.set_yscale('log')
ax.set_yticks(10**powers)
ax.set_yticklabels(y_ticks)
ax.set_ylabel("allocation size")
ax.plot(x, y)
ax.scatter(x, y)
plt.show()
plt.savefig("plot.png", dpi=150)

#labels = list(timer_report.keys())
##times  = [timer_report[k]["time_ns"] / 1e9 for k in labels]
#times  = [timer_report[k]["allocated_bytes"] / 1e6 for k in labels]
#
#fig, ax = plt.subplots()
#bars = ax.barh(labels, times)
#ax.bar_label(bars, fmt="%.4f s")
#ax.set_xlabel("Time (s)")
#ax.set_title("Timer breakdown")
#plt.tight_layout()
#plt.savefig("timings.png", dpi=150)