import torch
from torch.profiler import profile, ProfilerActivity, record_function
from contextlib import nullcontext

# ============ CONFIGURATION ============
ENABLE_PROFILING = True  # Set to False to disable all profiling
NUM_STEPS = 10000
PROFILE_WINDOWS = [
    (1000, 1020),   # Window 1: steps 1000-1019
    (3000, 3020),   # Window 2: steps 3000-3019
    (6000, 6020),   # Window 3: steps 6000-6019
    (9000, 9020),   # Window 4: steps 9000-9019
]
# =======================================

class Profiler:
    """Simple wrapper for conditional profiling"""
    def __init__(self, enabled=True):
        self.enabled = enabled
        self.active_profiler = None
        self.results = []
    
    def should_profile(self, step):
        if not self.enabled:
            return False
        return any(start <= step < end for start, end in PROFILE_WINDOWS)
    
    def start(self, step):
        if not self.enabled:
            return
        
        # Check if this is the first step of a window
        for start, end in PROFILE_WINDOWS:
            if step == start:
                self.active_profiler = profile(
                    activities=[ProfilerActivity.CPU],
                    profile_memory=True,
                    record_shapes=True,
                )
                self.active_profiler.__enter__()
                print(f"Started profiling at step {step}")
                break
    
    def stop(self, step):
        if not self.enabled or self.active_profiler is None:
            return
        
        # Check if this is the last step of a window
        for start, end in PROFILE_WINDOWS:
            if step == end - 1:
                self.active_profiler.__exit__(None, None, None)
                self.results.append(self.active_profiler)
                print(f"Stopped profiling at step {step}")
                self.active_profiler = None
                break
    
    def get_context(self):
        """Returns profiling context or nullcontext if disabled"""
        if self.enabled and self.active_profiler is not None:
            return record_function
        return lambda name: nullcontext()
    
    def print_results(self):
        if not self.enabled or not self.results:
            print("No profiling results available")
            return
        
        print("\n" + "="*80)
        print("PROFILING RESULTS")
        print("="*80)
        
        for i, prof in enumerate(self.results, 1):
            print(f"\n--- Window {i} ---")
            print(prof.key_averages().table(
                sort_by="cpu_time_total",
                row_limit=15
            ))
            for evt in prof.key_averages():
                if evt.key in {"forward_pass", "backward_pass", "optimizer_step", "autodiff_derivatives"}:
                    print(evt)

        
        ## Export for visual inspection
        #for i, prof in enumerate(self.results, 1):
        #    prof.export_chrome_trace(f"profile_window_{i}.json")
        
        #print("\n" + "="*80)
        #print(f"Exported {len(self.results)} trace files: profile_window_1.json, ...")
        #print("View at: chrome://tracing or https://ui.perfetto.dev/")
        #print("="*80)



def training_step(model, optimizer, data_points, collocation_points, profiler):
    """Your PINN training step with selective profiling"""
    rec = profiler.get_context()  # Gets record_function or nullcontext
    
    optimizer.zero_grad()
    
    # Profile only what you want
    with rec("forward_pass"):
        u_pred = model(collocation_points)
    
    with rec("autodiff_derivatives"):
        u_x = torch.autograd.grad(
            u_pred.sum(), collocation_points,
            create_graph=True, retain_graph=True
        )[0]
        
        u_xx = torch.autograd.grad(
            u_x.sum(), collocation_points,
            create_graph=True
        )[0]
    
    with rec("loss_computation"):
        # Example: Poisson equation residual
        residual = u_xx - torch.sin(collocation_points)
        loss = torch.mean(residual**2)
    
    with rec("backward_pass"):
        loss.backward()
    
    with rec("optimizer_step"):
        optimizer.step()
    
    return loss.item()


def train_pinn(model, optimizer, data_points, collocation_points):
    """Main training loop"""
    profiler = Profiler(enabled=ENABLE_PROFILING)
    
    for step in range(NUM_STEPS):
        profiler.start(step)
        
        loss = training_step(model, optimizer, data_points, collocation_points, profiler)
        
        profiler.stop(step)
        
        if step % 1000 == 0:
            print(f"Step {step:5d}, Loss: {loss:.6f}")
    
    # Print and export results
    profiler.print_results()


# ============ EXAMPLE USAGE ============
if __name__ == "__main__":
    # Simple example setup
    torch.manual_seed(42)
    
    model = torch.nn.Sequential(
        torch.nn.Linear(1, 32),
        torch.nn.Tanh(),
        torch.nn.Linear(32, 32),
        torch.nn.Tanh(),
        torch.nn.Linear(32, 1)
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Generate dummy data
    data_points = torch.linspace(0, 1, 100).reshape(-1, 1).requires_grad_(True)
    collocation_points = torch.linspace(0, 1, 200).reshape(-1, 1).requires_grad_(True)
    
    # Run training
    train_pinn(model, optimizer, data_points, collocation_points)