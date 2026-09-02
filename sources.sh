
cp -r /storage/praha1/home/simulant_antos/PINN/src/*.py PINN/src/
cp -r /storage/praha1/home/simulant_antos/PINN/case0_HeatEq/*.py PINN/case0_HeatEq
(BOOKWORM)simulant_antos@galdor19:/scratch.ssd/simulant_antos/job_22021524.pbs-m1$ mkdir -p PINN/src
(BOOKWORM)simulant_antos@galdor19:/scratch.ssd/simulant_antos/job_22021524.pbs-m1$ mkdir -p PINN/case0_HeatEq
(BOOKWORM)simulant_antos@galdor19:/scratch.ssd/simulant_antos/job_22021524.pbs-m1$ lsr /storage/praha1/home/simulant_antos/PINN/src/
-bash: lsr: command not found
(BOOKWORM)simulant_antos@galdor19:/scratch.ssd/simulant_antos/job_22021524.pbs-m1$ ls /storage/praha1/home/simulant_antos/PINN/src/
ls: cannot access '/storage/praha1/home/simulant_antos/PINN/src/': No such file or directory
(BOOKWORM)simulant_antos@galdor19:/scratch.ssd/simulant_antos/job_22021524.pbs-m1$ ls /storage/praha1/home/simulant_antos/diploma_sketchbook/PINN/src/
1d_plot.py		derivatives.py	    pde_models.py	       sampling.py		  utility.py			 viz_lj.py
architecture.py		grid_search.py	    pde_models_sde.py	       score_pinns.md		  visualize_fn.py		 viz_sde_trajectory_sampling.ipynb
causal_weighting.ipynb	lbfgs.md	    pp.py		       smoluchowski.ipynb	  visualize_solution_3anims.py
compile.py		loss.py		    profiler_grid_search       sp_report.md		  visualize_solution_3plots.py
compile.sh		main_old.py	    report.md		       test_derivatives.py	  visualize_training_metrics.py
compile_loop.py		main_runner.py	    run_SM-DW_score_pde--bosf  test_sampling_boundary.py  viz.py
compile_loop.sh		main_score_pinn.py  run_utils.py	       trainers.py		  viz_DW_pot.ipynb
(BOOKWORM)simulant_antos@galdor19:/scratch.ssd/simulant_antos/job_22021524.pbs-m1$ cp -r /storage/praha1/home/simulant_antos/diploma_sketchbook/PINN/src/*.py PINN/src/
(BOOKWORM)simulant_antos@galdor19:/scratch.ssd/simulant_antos/job_22021524.pbs-m1$ cp -r /storage/praha1/home/simulant_antos/diploma_sketchbook/PINN/src/*.py PINN/src/
(BOOKWORM)simulant_antos@galdor19:/scratch.ssd/simulant_antos/job_22021524.pbs-m1$ ls PINN/src/
1d_plot.py	 derivatives.py  main_runner.py      pp.py		  test_sampling_boundary.py  visualize_solution_3anims.py   viz_lj.py
architecture.py  grid_search.py  main_score_pinn.py  run_utils.py	  trainers.py		     visualize_solution_3plots.py
compile.py	 loss.py	 pde_models.py	     sampling.py	  utility.py		     visualize_training_metrics.py
compile_loop.py  main_old.py	 pde_models_sde.py   test_derivatives.py  visualize_fn.py	     viz.py
(BOOKWORM)simulant_antos@galdor19:/scratch.ssd/simulant_antos/job_22021524.pbs-m1$ cp -r /storage/praha1/home/simulant_antos/diploma_sketchbook/PINN/case0_HeatEq/*.py PINN/case0_HeatEq/
(BOOKWORM)simulant_antos@galdor19:/scratch.ssd/simulant_antos/job_22021524.pbs-m1$ ls
PINN  copy_over.sh  source_pinn.sh