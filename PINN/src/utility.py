

import json
import os
import torch
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

def get_duration_h_m_s(t1, t2, label="Smthg"):
    h, m, s = get_duration(t2 - t1)
    parts = [f"{label} completed in:"]
    if h > 0: parts.append(f"{h} hours")
    if m > 0: parts.append(f"{m} minutes")
    parts.append(f"{s} seconds")
    return " ".join(parts)


from torch.profiler import profile, ProfilerActivity
from contextlib import nullcontext

try:
    import psutil
except ImportError:
    psutil = None


class MemoryTracker:
    def __init__(self, device):
        self.device = device
        self.process = psutil.Process(os.getpid()) if psutil is not None else None
        self.use_gpu = getattr(device, "type", device) == "cuda" and torch.cuda.is_available()
        self.peak_process_rss_bytes = 0
        self.history = {
            "step": [],
            "process_rss_mb": [],
            "process_peak_rss_mb": [],
            "gpu_allocated_mb": [],
            "gpu_reserved_mb": [],
            "gpu_peak_allocated_mb": [],
            "gpu_peak_reserved_mb": [],
        }
        if self.use_gpu:
            torch.cuda.reset_peak_memory_stats(device)

    def sample(self, step):
        process_rss_bytes = self.process.memory_info().rss if self.process is not None else None
        if process_rss_bytes is not None:
            self.peak_process_rss_bytes = max(self.peak_process_rss_bytes, process_rss_bytes)

        gpu_allocated_bytes = None
        gpu_reserved_bytes = None
        gpu_peak_allocated_bytes = None
        gpu_peak_reserved_bytes = None
        if self.use_gpu:
            gpu_allocated_bytes = torch.cuda.memory_allocated(self.device)
            gpu_reserved_bytes = torch.cuda.memory_reserved(self.device)
            gpu_peak_allocated_bytes = torch.cuda.max_memory_allocated(self.device)
            gpu_peak_reserved_bytes = torch.cuda.max_memory_reserved(self.device)

        sample = {
            "step": step,
            "process_rss_mb": self._bytes_to_mb(process_rss_bytes),
            "process_peak_rss_mb": self._bytes_to_mb(self.peak_process_rss_bytes) if process_rss_bytes is not None else None,
            "gpu_allocated_mb": self._bytes_to_mb(gpu_allocated_bytes),
            "gpu_reserved_mb": self._bytes_to_mb(gpu_reserved_bytes),
            "gpu_peak_allocated_mb": self._bytes_to_mb(gpu_peak_allocated_bytes),
            "gpu_peak_reserved_mb": self._bytes_to_mb(gpu_peak_reserved_bytes),
        }
        for key, value in sample.items():
            self.history[key].append(value)
        return sample

    def format_sample(self, sample):
        parts = []
        if sample["process_rss_mb"] is not None:
            parts.append(
                f"RAM RSS/peak: {sample['process_rss_mb']:.1f}/{sample['process_peak_rss_mb']:.1f} MB"
            )
        if sample["gpu_allocated_mb"] is not None:
            parts.append(
                "GPU alloc/res/peak: "
                f"{sample['gpu_allocated_mb']:.1f}/{sample['gpu_reserved_mb']:.1f}/{sample['gpu_peak_allocated_mb']:.1f} MB"
            )
        return ", ".join(parts)

    @staticmethod
    def _bytes_to_mb(value):
        if value is None:
            return None
        return value / (1024 ** 2)


class Profiler():
    def __init__(self,report_filename, start_step, end_step, device='cpu'):
        self.report_filename = report_filename
        self.start_step = start_step
        self.end_step = end_step
        self.use_gpu = True if device.type=='cuda' else False
        self.activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA] if self.use_gpu else [ProfilerActivity.CPU]

    def make(self) -> None:
        self.prof_ctx = profile(
            activities=self.activities,
            profile_memory=True,
            record_shapes=True,
            with_stack=True,
        )

#    def _safe_attr(self, event, attr_name, default=None):
#        try:
#            return getattr(event, attr_name)
#        except (AttributeError, AssertionError):
#            return default
#
#    def _event_to_dict(self, event):
#        return {
#            "name": self._safe_attr(event, "key"),
#            "count": self._safe_attr(event, "count"),
#            "cpu_time_total_us": self._safe_attr(event, "cpu_time_total"),
#            "self_cpu_time_total_us": self._safe_attr(event, "self_cpu_time_total"),
#            "device_time_total_us": self._safe_attr(event, "device_time_total"),
#            "self_device_time_total_us": self._safe_attr(event, "self_device_time_total"),
#            "cpu_memory_usage_bytes": self._safe_attr(event, "cpu_memory_usage"),
#            "self_cpu_memory_usage_bytes": self._safe_attr(event, "self_cpu_memory_usage"),
#            "device_memory_usage_bytes": self._safe_attr(event, "device_memory_usage"),
#            "self_device_memory_usage_bytes": self._safe_attr(event, "self_device_memory_usage"),
#            "is_user_annotation": bool(self._safe_attr(event, "is_user_annotation", False)),
#            "input_shapes": self._safe_attr(event, "input_shapes"),
#        }
#
#    def _event_sort_key(self, row):
#        return (
#            row.get("device_time_total_us") or 0.0,
#            row.get("cpu_time_total_us") or 0.0,
#            row.get("self_device_time_total_us") or 0.0,
#            row.get("self_cpu_time_total_us") or 0.0,
#        )
#
#    def _write_event_summary(self, events, output_path, label):
#        rows = [self._event_to_dict(event) for event in events]
#        rows.sort(key=self._event_sort_key, reverse=True)
#        payload = {
#            "label": label,
#            "sort_priority": [
#                "device_time_total_us",
#                "cpu_time_total_us",
#                "self_device_time_total_us",
#                "self_cpu_time_total_us",
#            ],
#            "num_events": len(rows),
#            "events": rows,
#        }
#        json_dump(output_path, payload)

    def start(self, si):
        if si == self.start_step:
            self.prof_ctx.__enter__()
            print(f"\n[Profiler] Started at step {si}")

    def exit(self, si):
        if si == self.end_step:
            self.prof_ctx.__exit__(None, None, None)
            print(f"\n[Profiler] Stopped at step {si}.")
            self.prof_ctx.export_chrome_trace(f"{self.report_filename}_chroma.json")
            avg = self.prof_ctx.key_averages()
            sort_by = "device_time_total" if self.use_gpu else "cpu_time_total"

            report = avg.table(sort_by, row_limit=20)
            with open(f"{self.report_filename}.txt", "w") as f:
                f.write(report)

            #operator_events = [evt for evt in avg if not evt.is_user_annotation]
            #annotation_events = [evt for evt in avg if evt.is_user_annotation]

            #self._write_event_summary(operator_events, f"{self.report_filename}_torch_ops.json", "internal profiler operator events")
            #self._write_event_summary(annotation_events, f"{self.report_filename}_record_fun.json", "record_function events")





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
    return list(map(lambda x: int(x.strip()), layers_string.split(",")))


def floats_from_string_list(string):
    return list(map(lambda x: float(x.strip()), string.split(",")))


import inspect
def get_module_classes(module):
    return {
        name: obj
        for name, obj in inspect.getmembers(module, inspect.isclass)
    }



def header(dir_name, device=None):
    import torch
    import architecture
    import pde_models
    print(f"Will be working in directory '{dir_name}'...")

    model_metadata = json_load(f"{dir_name}/model_metadata.json")
    pde_metadata = json_load(f"{dir_name}/pde_metadata.json")

    d = model_metadata["args"]["d"]
    D = d+1
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_class_name = model_metadata["model_class"]
    print(model_class_name)
    model = get_module_classes(architecture)[model_class_name](D, layers_from_string(model_metadata["args"]["layers"])).to(device)
    model.load_state_dict(torch.load(f'{dir_name}/model.pth', map_location=device))
    model.eval()

    pde_class_name = pde_metadata["pde_class"]
    print(pde_class_name)
    pde_model = get_module_classes(pde_models)[pde_class_name](d)
    pde_model.load_pde_metadata(pde_metadata)
    u_analytic = pde_model.u_analytic

    return model, u_analytic, pde_metadata, model_metadata


def identity_fn(y,x):
    return y


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
            t = torch.rand(n_test_points, 1, device=x.device, dtype=x.dtype) * T

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

    def test_model(self, model, test_bs=100_000, device="cpu"):
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
                X_chunk = X[i:j].to(device)
                target_chunk = target[i:j].to(device)

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
            X, _, _, _ = sampling.sample_collocation_points(
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
                payload = torch.load(self.test_file_path, map_location="cpu")
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
                X_chunk = X[i:j].to(device)
                u_true_chunk = u_true[i:j].to(device)

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


def generate_SPD(d, eps=1e-10, device=None, dtype=torch.float32):
    B = torch.randn(d, d, device=device, dtype=dtype)
    A = torch.mm(B, B.t())
    # to ensure PD
    jitter = torch.eye(d, device=device, dtype=dtype) * eps
    A = A + jitter
    return A

def make_fn_0_care_t(fn):
    return lambda X: fn(X[:,:-1])


class TimeMarchModel:
    """
    collects all the pinn models together 
    """
    def __init__(self, time_march_discr, model_list, debugg=False):
        self.t_disc = time_march_discr
        self.model_list = model_list
        assert len(self.model_list) == len(self.t_disc)-1
        self.debugg = debugg
    def __call__(self, X):
        Y = torch.zeros((len(X),1), dtype=X.dtype, device=X.device)
        t_disc = self.t_disc
        for i in range(len(t_disc)-1):
            #frac = 0.1
            #t_left = t_disc[i] - frac*(t_disc[i]-t_disc[i-1]) if i-1>=0 else t_disc[i]
            #t_right = t_disc[i+1] + frac*(t_disc[i+2]-t_disc[i+1]) if i+2 <= len(t_disc)-1 else t_disc[i+1]
            #t_left_list.append(t_left)
            #t_right_list.append(t_right)
            t_left = t_disc[i]
            t_right = t_disc[i+1]
            if self.debugg:
                print(t_left, t_right)
            t = X[:,-1]
            if i == 0:
                mask = (t < t_right)
            elif i == len(t_disc)-1:
                mask = (t_left <= t)
            else:
                mask = (t_left <= t) & (t < t_right)
            Y[mask] = self.model_list[i](X[mask])
        return Y


    def test(self):
        model = lambda X: torch.sum(X, dim=1, keepdim=True)
        self.model_list = 3*[model]
        self.t_disc = [0.0, 0.5, 1.5, 3.5]
        self.debugg = True
        bs = 1_000
        d = 5
        X = torch.rand((bs, d+1))
        X[:,-1] *= 3.5
        Y = self(X)
