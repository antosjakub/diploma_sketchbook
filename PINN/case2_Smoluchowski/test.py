
import os, sys
src_dir = os.path.join(os.path.dirname(__file__), '../src/')
sys.path.append(src_dir)
import sampling


import torch
class TestingSuiteFP:
    """
    rel_l2 and linf
        - IC and TC - importrance smapl
        - n_test_pnts
    SDE data over full traj - need to be large to be able to calc the prob
    (just do lhs for now)
    """
    def __init__(self, d, device: torch.device, test_bs=1000, test_norm_slices=[0.0, 1.0]):
        self.d = d
        self.device = device
        self.test_bs = test_bs
        self.test_file_exists = False
        self.test_file_path = None
        self.keep_in_cache = True
        self.analytic_terms = ("ic", "tc")
        self.test_norm_slices = test_norm_slices
    
    def make_test_data(self, model, pde_model, sampling_type, sampling_settings, ic_fn, tc_fn, file_path, device, seed=4242):
        f_ic_full_domain = sampling_settings.get("f_ic_full_domain", 1)
        f_ic_trajs = sampling_settings.get("f_ic_trajs", 1)
        strategy = sampling_settings.get("sampling_strategy", "lhs")
        spatial_domain = sampling_settings.get("spatial_domain")
        # for ic and for tc:
        n_ic_tc = sampling_settings.get("n_res_points")
        n_norm_points = sampling_settings.get("n_res_points")
        d = self.d
        T = sampling_settings.get("T")
        lo = spatial_domain[:, 0]
        hi = spatial_domain[:, 1]
        self.volume = (spatial_domain[:, 1] - spatial_domain[:, 0]).prod()
        #

        cuda_devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            # sample IC : lhs + p_0 based
            n_ic_full_domain = f_ic_full_domain * n_ic_tc // (f_ic_full_domain + f_ic_trajs)
            n_ic_trajs = n_ic_tc - n_ic_full_domain
            x0_ic_trajs = pde_model.sample_x0(n_ic_trajs)
            X_ic_trajs = sampling.contruct_trajs_ic(x0_ic_trajs, n_ic_trajs)
            X_ic_full_domain = sampling.sample_ic(n_ic_full_domain, d, sampling_strategy=strategy, device=device)
            X_ic_full_domain = sampling.scale_samples__spatial(X_ic_full_domain, lo, hi)
            X_ic = torch.cat([
                X_ic_trajs,
                X_ic_full_domain
            ], dim=0)
            # sample TC : lhs + p_inf based
            n_tc_full_domain = f_ic_full_domain * n_ic_tc // (f_ic_full_domain + f_ic_trajs)
            n_tc_trajs = n_ic_tc - n_tc_full_domain
            x0_tc_trajs = pde_model.sample_xinf(n_tc_trajs)
            X_tc_trajs = sampling.contruct_trajs_ic(x0_tc_trajs, n_tc_trajs)
            X_tc_trajs[:,-1] = T
            X_tc_full_domain = sampling.sample_ic(n_tc_full_domain, d, sampling_strategy=strategy, device=device)
            X_tc_full_domain = sampling.scale_samples__spatial(X_tc_full_domain, lo, hi)
            X_tc_full_domain[:,-1] = T
            X_tc = torch.cat([
                X_tc_trajs,
                X_tc_full_domain
            ], dim=0)
            # sample SDE time windows ?
            #X_pde_list = []
            #for i in range(len(time_slices)):
            #    X_pde = sampling.sample_domain(n_interior, d+1, sampling_strategy=strategy, device=device)
            #    X_pde = sampling.scale_samples__spatial(X_pde, lo, hi)
            #    X_pde[:,-1] = time_slices[i]
            #    X_pde_list.append(X_pde)
            X_norm = sampling.sample_domain(n_norm_points, d+1, sampling_strategy=strategy, device=device)
            X_norm = sampling.scale_samples__spatial(X_norm, lo, hi)
            X_norm = X_norm[:,:-1]

        data = {}
        data[f"X_ic"] = X_ic
        data[f"X_tc"] = X_tc
        #data[f"X_time_slices"] = X_pde_list
        data[f"X_norm"] = X_norm
        with torch.no_grad():
            data[f"analytic_ic"] = ic_fn(X_ic[:,:-1])
            data[f"analytic_tc"] = tc_fn(X_tc[:,:-1])

        payload = {
            "metadata": {
                "d": self.d,
                "seed": seed,
                "sampling_type": sampling_type,
                "sampling_settings": sampling_settings,
            },
            "data": data,
            "norm_slices": self.test_norm_slices
        }
        self.payload = payload
        self.test_file_exists = True
        self.test_file_path = file_path
        #torch.save(payload, file_path)


    def test_model(self, model, pde_model, device, ignore_bc=True):
        """
        repot
        - linf, rel_l2 with keys ic, tc
        - prob using SDE at time slices (keys) t = 0.0, 0.25, 0.5, 5.0 
        """
        test_bs = self.test_bs

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
            data = payload["data"]
        except Exception as exc:
            raise RuntimeError("Unable to load the testing data.") from exc

        metrics_rel_l2 = {}
        metrics_linf = {}
        model.eval()
        eps = 1e-12

        for term in ("ic", "tc"):
            X = data[f"X_{term}"]
            analytic = data[f"analytic_{term}"]
            err_sq = 0.0
            err_max = -1.0
            target_sq = 0.0
            n = len(X)
            with torch.no_grad():
                for i in range(0, n, test_bs):
                    j = min(i + test_bs, n)
                    X_chunk = X[i:j].to(device)
                    u_true_chunk = analytic[i:j].to(device)
                    u_pred_chunk = model(X_chunk)
                    # rel_l2
                    err_sq += torch.sum((u_pred_chunk - u_true_chunk) ** 2).item()
                    target_sq+= torch.sum(u_true_chunk ** 2).item()
                    # linf
                    err_max_curr = torch.abs(u_pred_chunk - u_true_chunk).max()
                    if err_max_curr > err_max:
                        err_max = err_max_curr
            metrics_rel_l2[term] = (err_sq/ max(target_sq, eps)) ** 0.5
            metrics_linf[term] = err_max.item()

        metrics_norm = {} 
        time_slices = payload["norm_slices"]
        #X_list = payload["X_times_slices"]
        X_norm = data["X_norm"]
        with torch.no_grad():
            for i in range(len(time_slices)):
                X = torch.cat([
                    X_norm,
                    time_slices[i]*torch.ones((X_norm.shape[0],1))
                ], dim=1)
                p = model(X)
                integral_est = self.volume * p.mean()
                metrics_norm[str(time_slices[i])] = integral_est.item()

        model.train()

        b = time.time()
        print(f"-- Testing took: {(b-a):.4f}s")
        return metrics_rel_l2, metrics_linf, metrics_norm