import torch
import derivatives, architecture
import utility
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--d", default=4, type=int)
parser.add_argument("--layers", default="128,128,128,128", type=str)
parser.add_argument("--der_mode", default="grad", type=str, help="grad, lapl, div")
parser.add_argument("--n_calloc_pnts", default=10_000, type=int)
parser.add_argument("--n_steps", default=10, type=int)
parser.add_argument("--n_steps_active", default=3, type=int)
parser.add_argument("--n_steps_wait", default=3, type=int)
parser.add_argument("--n_steps_warmup", default=2, type=int)
args = parser.parse_args()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device", device)

D = args.d+1
layers = utility.layers_from_string(args.layers)

model = architecture.PINN(D, layers, 1)
#model = torch.compile(model)

n_steps = args.n_steps
n_res = args.n_calloc_pnts
der_mode = args.der_mode

from torch.profiler import profile, ProfilerActivity
from torch.profiler import record_function

cpu_time_schedule = torch.profiler.schedule(wait=args.n_steps_wait, warmup=args.n_steps_warmup, active=args.n_steps_active, repeat=1)
mem_schedule = torch.profiler.schedule(
    wait=0, warmup=0,
    active=args.n_steps_active+args.n_steps_wait+args.n_steps_warmup,
    repeat=1)

prof_ctx = profile(
    activities=[ProfilerActivity.CPU],
    profile_memory=True,
    record_shapes=True,
    with_stack=True,
    #schedule=cpu_time_schedule
    schedule=mem_schedule
)

for s in range(n_steps):

    if s == 0:
        prof_ctx.__enter__()

    X = torch.rand(n_res, D, dtype=torch.float32, device=device)
    X.requires_grad = True
    #print(X.shape)

    with record_function("der-comp"):
        u = model(X)
        if der_mode == 'grad':
            u_grad = derivatives.compute_grad(X, u, torch.ones_like(u))
        elif der_mode == 'lapl':
            u, grad_u, spatial_laplace_u = derivatives.compute_derivatives(model, X)
        elif der_mode == 'div':
            u_grad = derivatives.compute_grad(X, u, torch.ones_like(u))
            u_t = u_grad[:,-1:]
            u_jac = u_grad[:,:-1]
            model_u_grad = lambda X_in: u_jac
            s, jac = derivatives.compute_score_and_jacobian(model_u_grad, X)

    prof_ctx.step()

prof_ctx.__exit__(None, None, None)
# save results
prof_ctx.export_chrome_trace("prof_trace.json")
print("https://ui.perfetto.dev/")
# 
report = prof_ctx.key_averages().table(sort_by="cpu_time_total", row_limit=20)
with open("prof_report.txt", "w") as f:
    f.write(report)

# profiling guide:
# https://huggingface.co/blog/torch-profiler
# 
# - The "Self" columns measure time spent only inside the event itself, excluding its children.
# - The "total" columns include the event and all of its children together.
# 
# analyze the .json file with:
# https://ui.perfetto.dev/


print()
my_events = []
prof_ctx.key_averages()[0].key
for event in prof_ctx.key_averages():
    if event.key in ["sample", "forward-pass", "der-comp"]:
        my_events.append(event)

for ev in my_events:
    print(f"{ev.key}")
    count = ev.count
    print(f"""\
- cpu_time={ev.cpu_time/1000:f}[ms]
- cpu_mem_usage_avg={ev.cpu_memory_usage/1024**2/count}[MB]
- cpu_mem_usage_tot={ev.cpu_memory_usage/1024**2}[MB] (count={count})
""")
