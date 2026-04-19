python main_score_pinn_hardcoded.py --ic_type=laplace --mode=score_pde --sampling_type=trajectories --T=4.0 --n_steps=1_950 --layers=128,128,128,128 --resampling_frequency=500 --L_min=-4.0 --L_max=4.0 --nt_steps=1_000 --n_trajs=10_000

python main_score_pinn_hardcoded.py --ic_type=laplace --mode=ll_ode --sampling_type=trajectories --T=4.0 --n_steps=1_950 --layers=128,128,128,128 --resampling_frequency=500 --L_min=-4.0 --L_max=4.0 --nt_steps=1_000 --n_trajs=10_000


python main_score_pinn_3losses.py --ic_type=laplace --mode=score_pde --sampling_type=domain --T=4.0 --n_steps=1_950 --layers=128,128,128,128 --resampling_frequency=500 --L_min=-4.0 --L_max=4.0 --res_points=100_000 --use_adaptive_loss