
```bash
ssh simulant_antos@tarkil.metacentrum.cz
```

pbs-get-job-history job_ID

copy out
```bash
scp simulant_antos@tarkil.metacentrum.cz:~/

scp simulant_antos@tarkil.metacentrum.cz:~/26-06-17--12:46:30__JOBNAME=large_calloc/profiler_grid_search
```

scp -r simulant_antos@tarkil.metacentrum.cz:~/diploma_sketchbook/PINN/case0_HeatEq/gridsearch__26-07-08--15:37:47__GS_lambdas_fixed_dataset/ cluster_res/


rsync -avz --exclude='*.png'
username@cluster_address:/path/to/source_folder /path/to/destination_folder 

rsync -avz --exclude='*.png' simulant_antos@tarkil.metacentrum.cz:~/diploma_sketchbook/PINN/case0_HeatEq/gridsearch__26-07-08--10:11:39__GS_6_bs_and_resample/ cluster_res/

rsync -avz --exclude='*.png' simulant_antos@tarkil.metacentrum.cz:~/diploma_sketchbook/PINN/case0_HeatEq/gridsearch__26-07-08--10:11:39__GS_6_bs_and_resample/ cluster_res/

gridsearch__26-07-11--15:32:36__GS_d6_bs2048

rsync -avz --exclude='*.png' simulant_antos@tarkil.metacentrum.cz:~/diploma_sketchbook/PINN/case0_HeatEq/gridsearch__26-07-09--20:01:51__GS_d4/ cluster_d4/


%/storage/praha1/home/simulant_antos/diploma_sketchbook/PINN/case0_HeatEq


## job info

job status
- Q=queued
- R=running
- F=finished

```bash
qstat -u simulant_antos # lists your running and queuing jobs
qstat -x -u simulant_antos # lists your running, queuing and finished jobs

qstat $jobID # info bout a job
qdel $jobID # deletes a job
```

### GPU

```bash
qstat -fw $jobID | grep gpu
qstat -H -fw $jobID | grep gpu
```
returns info about:
- [%] how much the GPU(s) have been used during the job
- [Wh] how much energy was burned on all GPUs for a given job
- [%] the maximum peak of GPU memory usage




## launch interactive job

select=1:ncpus=1:mem=10gb:scratch_local=10gb
```bash
qsub -I -l walltime=01:00:00 -l select=1:ncpus=1:ngpus=1:mem=10gb:gpu_mem=10gb:scratch_local=10gb:cuda_version=13.2
```



## create package folder and install via pip

```bash
export TMPDIR=$SCRATCHDIR

# list available python versions
module avail python/

# pick the latest one
module add python/3.11.11-gcc-10.2.1-555dlyc

export PYTHONUSERBASE=/storage/praha1/home/simulant_antos/venv_pinn_cuda13

mkdir $PYTHONUSERBASE

# install via pip, use the --user flag to install to PYTHONUSERBASE folder
pip3 install torch --index-url https://download.pytorch.org/whl/cu130 --user
pip3 install numpy matplotlib --user

# create a tar file
tar -cf venv_pinn_cuda13.tar venv_pinn_cuda13/
```


## then always set in job bash file


### either (point python to the folder in storage):
```bash
module add python/3.11.11-gcc-10.2.1-555dlyc

# setup system variables 
export PYTHONUSERBASE=/storage/praha1/home/simulant_antos/venv_pinn_cuda13

export PATH=$PYTHONUSERBASE/bin:$PATH
export PYTHONPATH=$PYTHONUSERBASE/lib/python3.11/site-packages:$PYTHONPATH
```

### or (copy into local machine):
```bash
module add python/3.11.11-gcc-10.2.1-555dlyc

cd $SCRATCHDIR/

## copy the folder
cp -r /storage/praha1/home/simulant_antos/venv_pinn_cuda13/ .
## copy the .tar file
cp /storage/praha1/home/simulant_antos/venv_pinn_cuda13.tar .
tar -xf venv_pinn_cuda13.tar

# setup system variables 
export PYTHONUSERBASE=venv_pinn_cuda13
export PATH=$PYTHONUSERBASE/bin:$PATH
export PYTHONPATH=$PYTHONUSERBASE/lib/python3.11/site-packages:$PYTHONPATH
```

## submitting a job

```bash
qsub -I -l walltime=01:00:00 -l select=1:ncpus=1:ngpus=1:mem=10gb:gpu_mem=10gb:scratch_local=10gb:cuda_version=13.2
```

### run.sh

```bash
#!/bin/bash
#PBS -N jobe_noname
#PBS -o out.o
#PBS -e out.e

#PBS -l walltime=00:30:00
#PBS -l select=1:ncpus=1:ngpus=1:mem=10gb:gpu_mem=10gb:scratch_local=10gb:cuda_version=13.2

home_run_dir="${PBS_O_WORKDIR}/jobname=${PBS_JOBNAME}_jobid=${PBS_JOBID}"
mkdir -p $home_run_dir

infofile=$home_run_dir/node_info.txt
touch $infofile
echo -e "--- GPU Status ---" >> "${infofile}"
nvidia-smi >> "${infofile}" 2>&1


# load the python module
module add python/3.11.11-gcc-10.2.1-555dlyc

# point python to where the packages are inst
export PYTHONUSERBASE=/storage/praha1/home/simulant_antos/venv_pinn_cuda13
export PATH=$PYTHONUSERBASE/bin:$PATH
export PYTHONPATH=$PYTHONUSERBASE/lib/python3.11/site-packages:$PYTHONPATH

# copy the files to the scratch directory
cp -r ${PBS_O_WORKDIR}/diploma_sketchbook/PINN $SCRATCHDIR/

# go to the scratchdir
cd ${SCRATCHDIR}


cd PINN/case1_OrnsteinUhlenbeck/
#python grid_search.py
python main_vanilla_pinn.py

# SAVE RESULTS TO MY STORAGE
# - cp
cp $PBS_O_WORKDIR/out.o $home_run_dir/out.o
cp $PBS_O_WORKDIR/out.e $home_run_dir/out.e
cp -r run_latest_vanilla/ $home_run_dir/

cd ../../

# automatic cleaner
clean_scratch
```