import re



time_ms = []
mem_MB = []
dims = []
for d in range(1,9+1):
    report = open(f"prof_rep_layers=d={d}.txt").read()

    targets = {"loss", "backward"}

    # Column indices based on header splitting
    header_line = "Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg       CPU Mem  Self CPU Mem    # of Calls"
    columns = ["Name", "Self CPU %", "Self CPU", "CPU total %", "CPU total", "CPU time avg", "CPU Mem", "Self CPU Mem", "# of Calls"]

    results = {}

    for line in report.splitlines():
        # Strip and split on 2+ spaces
        parts = re.split(r'\s{2,}', line.strip())
        if parts and parts[0] in targets:
            name = parts[0]
            results[name] = {
                "CPU total": parts[4],  # index 4 = CPU total
                "CPU Mem":   parts[6],  # index 6 = CPU Mem
            }

    print(f"d={d}")
    #for name, metrics in results.items():
    metrics = results["loss"]
    print(f"  CPU Mem   : {metrics['CPU Mem']}")
    print(f"  CPU total : {metrics['CPU total']}")
    mem = 0.1*float(results['loss']['CPU Mem'][:-3])
    time = 0.1*float(results['loss']['CPU total'][:-2])
    print(f"  [loss]:", [mem])
    print(f"  [loss]:", [time])
    time_ms.append(time)
    mem_MB.append(mem)
    dims.append(d)

import matplotlib.pyplot as plt
def plot_fig(arr, name, title, ylabel):
    # Plot L2
    plt.figure(figsize=(10, 5))
    plt.xlabel('d (dim)')
    plt.ylabel(f'{ylabel}')
    plt.plot(dims, arr)
    plt.title(f'{title}')
    plt.grid(True)
    plt.savefig(f'{name}.png', dpi=150)

is_weak_form = True
if is_weak_form:
    weak_form_title = " - weak formulation"
    weak_form_name = "_weak_form"
else:
    weak_form_title = ""
    weak_form_name = "_pde"

arr = mem_MB
name = f"loss_mem_{weak_form_name}"
title = f"AD computational graph allocation memory {weak_form_title}"
ylabel = "allocated memory [MB]"
plot_fig(arr, name, title, ylabel)

arr = time_ms
name = f"loss_time_{weak_form_name}"
title = f"Loss computation time {weak_form_title}"
ylabel = "time (ms)"
plot_fig(arr, name, title, ylabel)
