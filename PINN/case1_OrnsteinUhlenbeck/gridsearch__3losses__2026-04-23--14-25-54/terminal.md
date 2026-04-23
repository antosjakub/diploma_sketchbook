(venv) [antos_j@feed-aura case1_OrnsteinUhlenbeck]$ python grid_search.py
Grid search root: gridsearch__3losses__2026-04-23--14-25-54
Base combos: 4
Planned runs: 8

[1/4] use_adaptive_weights=True__n_steps_decay=800
Running: python main_score_pinn_3losses.py --config gridsearch__3losses__2026-04-23--14-25-54/use_adaptive_weights=True__n_steps_decay=800/score_pde/config.json
OK  score_pde  528.2s
Running: python main_score_pinn_3losses.py --config gridsearch__3losses__2026-04-23--14-25-54/use_adaptive_weights=True__n_steps_decay=800/ll_ode/config.json
OK  ll_ode  566.5s

[2/4] use_adaptive_weights=True__n_steps_decay=2000
Running: python main_score_pinn_3losses.py --config gridsearch__3losses__2026-04-23--14-25-54/use_adaptive_weights=True__n_steps_decay=2000/score_pde/config.json
OK  score_pde  536.5s
Running: python main_score_pinn_3losses.py --config gridsearch__3losses__2026-04-23--14-25-54/use_adaptive_weights=True__n_steps_decay=2000/ll_ode/config.json
OK  ll_ode  596.5s

[3/4] use_adaptive_weights=False__n_steps_decay=800
Running: python main_score_pinn_3losses.py --config gridsearch__3losses__2026-04-23--14-25-54/use_adaptive_weights=False__n_steps_decay=800/score_pde/config.json
OK  score_pde  561.7s
Running: python main_score_pinn_3losses.py --config gridsearch__3losses__2026-04-23--14-25-54/use_adaptive_weights=False__n_steps_decay=800/ll_ode/config.json
OK  ll_ode  611.3s

[4/4] use_adaptive_weights=False__n_steps_decay=2000
Running: python main_score_pinn_3losses.py --config gridsearch__3losses__2026-04-23--14-25-54/use_adaptive_weights=False__n_steps_decay=2000/score_pde/config.json
OK  score_pde  550.0s
Running: python main_score_pinn_3losses.py --config gridsearch__3losses__2026-04-23--14-25-54/use_adaptive_weights=False__n_steps_decay=2000/ll_ode/config.json
OK  ll_ode  586.1s

Grid search completed in: 1 hours 15 minutes 36.82722306251526 seconds
ok=8, failed=0
Summary: gridsearch__3losses__2026-04-23--14-25-54/summary.json
(venv) [antos_j@feed-aura case1_OrnsteinUhlenbeck]$