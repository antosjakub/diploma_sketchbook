

source ../../venv/bin/activate
python main_vanilla_pinn.py --starting_model=run_latest_vanilla_adam/model.pth --use_lbfgs --n_steps=49 --bs=50_000 --n_res_points=50_000 --f_pde_trajs=4 --f_ic_trajs=4 --output_dir=run_latest_vanilla_lbfgs