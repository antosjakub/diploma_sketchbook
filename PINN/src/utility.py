

import json
def json_dump(file_path, d):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(d, f, indent=4)

def json_load(file_path):
    with open(file_path, "r", encoding='utf-8') as f:
        d = json.load(f)
    return d


def get_duration(dt):
    h = dt // 3600
    m = (dt - 3600 * h) // 60
    s = dt - 3600 * h - 60 * m
    return int(h),int(m),s


from torch.profiler import profile, ProfilerActivity
from contextlib import nullcontext

class Profiler():
    def __init__(self,report_filename, start_step, end_step):
        self.report_filename = report_filename
        self.start_step = start_step
        self.end_step = end_step

    def make(self) -> None:
        self.prof_ctx = profile(
            activities=[ProfilerActivity.CPU],
            profile_memory=True,
            record_shapes=True,
            with_stack=True,
        )

    def start(self, si):
        if si == self.start_step:
            self.prof_ctx.__enter__()
            print(f"\n[Profiler] Started at step {si+1}")

    def exit(self, si):
        if si == self.end_step-1:
            self.prof_ctx.__exit__(None, None, None)
            print(f"\n[Profiler] Stopped. Results:")
            prof_report = self.prof_ctx.key_averages().table(sort_by="cpu_time_total", row_limit=20)
            print(prof_report)
            # save profiler report
            #prof_ctx.export_chrome_trace(f"run_latest/{profiler_report_filename}.json")
            with open(f"{self.report_filename}.txt", "w") as f:
                f.write(prof_report)

#class Profiler_Dummy()
#    def __init__(self):
#        pass
#    def make(self):
#        pass
#    def start(self, si):
#        pass
#    def exit(self, si):
#        pass


def layers_from_string(layers_string):
    return list(map(lambda x: int(x), layers_string.split(",")))



import inspect
def get_module_classes(module):
    return {
        name: obj
        for name, obj in inspect.getmembers(module, inspect.isclass)
    }



def header(dir_name):
    import torch
    import architecture
    import pde_models
    print(f"Will be working in directory '{dir_name}'...")

    model_metadata = json_load(f"{dir_name}/model_metadata.json")
    pde_metadata = json_load(f"{dir_name}/pde_metadata.json")

    d = model_metadata["args"]["d"]
    D = d+1
    model_class_name = model_metadata["model_class"]
    print(model_class_name)
    model = get_module_classes(architecture)[model_class_name](D, layers_from_string(model_metadata["args"]["layers"]))
    model.load_state_dict(torch.load(f'{dir_name}/model.pth'))
    model.eval()

    pde_class_name = pde_metadata["pde_class"]
    print(pde_class_name)
    pde_model = get_module_classes(pde_models)[pde_class_name](d)
    pde_model.load_pde_metadata(pde_metadata)
    u_analytic = pde_model.u_analytic

    return model, u_analytic, pde_metadata, model_metadata


def identity_fn(x):
    return x


import torch
class ScorePINNTestingSuite:
    """
    Testing suite for Score-PINN models.

    Works for both training stages:
      - score_pde:  model outputs s(x,t) of shape (N, d),  analytic_fn = pde_model.s_analytic
      - ll_ode:     model outputs q(x,t) of shape (N, 1),  analytic_fn = pde_model.q_analytic

    Pass the appropriate analytic function at construction time so the suite
    stays agnostic to which stage is being tested.
    """

    def __init__(self, d, analytic_fn, keep_in_cache=True):
        """
        d           : number of spatial dimensions
        analytic_fn : callable X (N, d+1) -> target (N, output_dim)
        """
        self.d = d
        self.analytic_fn = analytic_fn
        self.keep_in_cache = keep_in_cache
        self.test_data_ready = False

    def make_test_data(self, pde_model, n_test_points, T=1.0, seed=4242):
        """Sample test points (x ~ p0, t ~ Uniform[0,T]) and cache analytic targets."""
        cuda_devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            x = pde_model.sample_x0(n_test_points)
            t = torch.rand(n_test_points, 1) * T

        X = torch.cat([x, t], dim=1).detach().cpu()

        with torch.no_grad():
            target = self.analytic_fn(X).detach().cpu()

        payload = {
            "metadata": {"d": self.d, "N": n_test_points, "seed": seed},
            "data": {"X": X, "target": target},
        }
        if self.keep_in_cache:
            self.payload = payload
        self.test_data_ready = True

    def test_model(self, model, test_bs=100_000):
        """
        Compute L2 RMSE, mean L1, and max relative error between model output
        and the pre-cached analytic targets.

        Returns (l2_err, l1_err, rel_err).
        """
        import time
        a = time.time()

        if not self.test_data_ready:
            raise ValueError("Call make_test_data before test_model.")

        payload = self.payload
        X = payload["data"]["X"]
        target = payload["data"]["target"]
        N = X.shape[0]
        output_dim = target.shape[1]
        eps = 1e-10

        sum_sq = 0.0
        sum_abs = 0.0
        max_rel = 0.0

        model.eval()
        with torch.no_grad():
            for i in range(0, N, test_bs):
                j = min(i + test_bs, N)
                X_chunk = X[i:j]
                target_chunk = target[i:j]

                pred = model(X_chunk)
                err = pred - target_chunk

                sum_sq += torch.sum(err ** 2).item()
                sum_abs += torch.sum(err.abs()).item()

                rel_chunk = (err.abs() / (target_chunk.abs() + eps)).max().item()
                if rel_chunk > max_rel:
                    max_rel = rel_chunk
        model.train()

        n_elements = N * output_dim
        l2_err = (sum_sq / n_elements) ** 0.5
        l1_err = sum_abs / n_elements
        rel_err = max_rel

        b = time.time()
        print(f"Testing took: {b - a:.3f}s")
        return l2_err, l1_err, rel_err


class TestingSuite:
    def __init__(self, d, keep_in_cache=True):
        self.d = d
        self.test_file_exists = True
        self.test_file_path = ""
        self.keep_in_cache = keep_in_cache
    
    def connect_test_data(self, file_path: str):
        import os
        if os.path.exists(file_path):
            payload = torch.load(file_path, map_location="cpu")
            metadata = payload["metadata"]
            if (metadata["d"] != self.d):
                raise ValueError(
                    f"Dimension mismatch. Testing suite has d={self.d}, but the loaded data have d={metadata['d']}."
                )
            assert payload["data"]["X"].shape[1] == self.d+1
            assert payload["data"]["u_true"].shape[1] == 1
            assert payload["data"]["X"].shape[0] == payload["data"]["u_true"].shape[0]
        if self.keep_in_cache: self.payload = payload
        self.test_file_exists = True
        self.test_file_path = file_path


    def make_test_data(self, pde_model, n_test_calloc_points, file_path, sampling_strategy="lhs", seed=4242):
        # Create once, deterministic.
        cuda_devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            import sampling
            X, _, _ = sampling.sample_collocation_points(
                self.d,
                n_test_calloc_points,
                0,
                0,
                sampling_strategy=sampling_strategy,
                device="cpu",
            )

        # Optional: pre-store analytic truth to avoid recomputing every test call.
        with torch.no_grad():
            u_true = pde_model.u_analytic(X)

        payload = {
            "metadata": {
                "d": self.d,
                "N": n_test_calloc_points,
                "sampling_strategy": sampling_strategy,
                "seed": seed,
            },
            "data": {
                "X": X,
                "u_true": u_true,
            }
        }
        if self.keep_in_cache:
            self.payload = payload
        else:
            torch.save(payload, file_path)
        self.test_file_exists = True
        self.test_file_path = file_path


    def test_model(self, model, test_bs=100_000, device="cpu"):

        import time
        a = time.time()

        if not self.test_file_exists:
            raise ValueError(
                "Make or Connect test data first before testing."
            )

        try:
            if self.keep_in_cache:
                payload = self.payload
            else:
                payload = torch.load(self.test_file_path)
            X = payload["data"]["X"]
            u_true = payload["data"]["u_true"]
        except:
            raise "Unable to load the testing data."

        N = X.shape[0]
        sum_l2 = 0.0
        sum_l1 = 0.0
        max_rel = 0.0
        eps = 1e-10

        model.eval()
        with torch.no_grad():
            for i in range(0, N, test_bs):
                j = min(i + test_bs, N)
                X_chunk = X[i:j]
                u_true_chunk = u_true[i:j]

                u_pred = model(X_chunk)
                err = u_pred - u_true_chunk

                sum_l2 += torch.sum(err**2).item()
                sum_l1 += torch.sum(err.abs()).item()

                rel_chunk = ( (err-eps) / (u_true_chunk-eps) ).abs().max().item()
                if rel_chunk > max_rel:
                    max_rel = rel_chunk
        model.train()

        l2_err = (sum_l2 / N)**(1/2)
        l1_err = sum_l1 / N
        rel_err = max_rel

        b = time.time()
        print(f"Testing took: {b-a}s")
        return l2_err, l1_err, rel_err