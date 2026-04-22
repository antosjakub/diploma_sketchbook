(venv) [antos_j@feed-aura case1_OrnsteinUhlenbeck]$ python grid_search.py
Grid search root: gridsearch__hardcoded__2026-04-22--18-37-08
Base combos: 4
Planned runs: 4

[1/4] layers=128_128_128_128__f_pde_full_domain=1__f_pde_trajs=7
Running: python main_score_pinn_hardcoded.py --config gridsearch__hardcoded__2026-04-22--18-37-08/layers=128_128_128_128__f_pde_full_domain=1__f_pde_trajs=7/ll_ode/config.json
OK  ll_ode  839.8s

[2/4] layers=128_128_128_128__f_pde_full_domain=7__f_pde_trajs=1
Running: python main_score_pinn_hardcoded.py --config gridsearch__hardcoded__2026-04-22--18-37-08/layers=128_128_128_128__f_pde_full_domain=7__f_pde_trajs=1/ll_ode/config.json
OK  ll_ode  836.1s

[3/4] layers=256_256_256_256__f_pde_full_domain=1__f_pde_trajs=7
Running: python main_score_pinn_hardcoded.py --config gridsearch__hardcoded__2026-04-22--18-37-08/layers=256_256_256_256__f_pde_full_domain=1__f_pde_trajs=7/ll_ode/config.json
OK  ll_ode  967.9s

[4/4] layers=256_256_256_256__f_pde_full_domain=7__f_pde_trajs=1
Running: python main_score_pinn_hardcoded.py --config gridsearch__hardcoded__2026-04-22--18-37-08/layers=256_256_256_256__f_pde_full_domain=7__f_pde_trajs=1/ll_ode/config.json
OK  ll_ode  972.9s

Grid search completed in: 1 hours 16.677940130233765 seconds
ok=4, failed=0
Summary: gridsearch__hardcoded__2026-04-22--18-37-08/summary.json