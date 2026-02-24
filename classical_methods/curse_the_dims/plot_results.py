
import os
import json
import numpy as np

# in one plot, show:
# - diff float types = Float64,33
# - matrix L vs vector Ut
# - discret n=32,48,64
# which combinations enables us to go to dim=5?

# index 0 should have the base unit, the other separated by 3 orders
time_units = ['ns', 'μs', 'ms', 's']
byte_units = ['B', 'kB', 'MB', 'GB', 'TB']

def get_report_dict():
    timer_report = {}
    for filename in os.listdir('results'):
        if not filename.endswith(".json"):
            continue
        
        with open(f'results/{filename}') as f:
            data = json.load(f)

        name = filename.split('.')[0]
        timer_report[name] = data['inner_timers']
    return timer_report

def extract_data(timer_report, report_name, metric_name):
    data = {}
    for key in timer_report.keys():
        t,n,d,ft = key.split(',')
        n = int(n.split('=')[1])
        d = int(d.split('=')[1])
        ft = int(ft.split('=')[1])
        if t == 'op' and ft == 64:
            identifier = f'n={n}'
            if identifier not in data:
                data[identifier] = {'x': [], 'y': []}
            data[identifier]['x'].append(d)
            data[identifier]['y'].append(timer_report[key][report_name][metric_name])
    data = dict(sorted(data.items(), reverse=True))
    print(data.keys())
    for identifier, vals in data.items():
        x = vals['x']
        y = vals['y']
        x = np.array(x)
        y = np.array(y)
        indices = np.argsort(x)
        x = x[indices]
        y = y[indices]
        data[identifier]['x'] = x
        data[identifier]['y'] = y
    return data


def create_plot(powers, unit_names, title, xlabel, ylabel, filename, weight=1):
    values = 10**(powers % 3)
    labels = np.array(unit_names)[((powers // 3))]
    y_ticks = []
    for l, v in zip(labels, values):
        y_ticks.append(f'{str(v)} {l}')

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.grid(visible=True)
    ax.set_xlabel(xlabel)
    ax.set_yscale('log')
    ax.set_yticks(10**powers)
    ax.set_yticklabels(y_ticks)
    ax.set_ylabel(ylabel)
    for identifier, vals in data.items():
        x = vals['x']
        y = weight*vals['y']
        ax.plot(x, y, label=identifier)
        ax.scatter(x, y)
    ax.set_xticks(x)
    ax.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()



if __name__ == "__main__":

    timer_reports = get_report_dict()
    # cg times
    data = extract_data(timer_reports, report_name="solve CG", metric_name="time_ns")
    powers = np.arange(5,10+1, dtype=int)
    create_plot(
        powers,
        time_units,
        title=r"Duration of one CG run (solving $Ax=b$)",
        xlabel="d (dimension)",
        ylabel="t (time)",
        filename="fdm_cg_runtime.png",
    )
    # total allocation
    data = extract_data(timer_reports, report_name="init Ut", metric_name="allocated_bytes")
    powers = np.arange(3,10+1, dtype=int)
    create_plot(
        powers,
        byte_units,
        title=r"Total allocation size (5 vectors of size $N = n^d$)",
        xlabel="d (dimension)",
        ylabel="allocation size",
        filename="fdm_total_alloc.png",
        weight=5
    )







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