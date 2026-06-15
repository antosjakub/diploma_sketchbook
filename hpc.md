
```bash
ssh simulant_antos@tarkil.metacentrum.cz
```



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

#PBS -l walltime=00:30:00
#PBS -l select=1:ncpus=1:ngpus=1:mem=10gb:gpu_mem=10gb:scratch_local=10gb:cuda_version=13.2
#PBS -N jobe_namer


# name a job info file
infofile=job_info.${PBS_JOBID}

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

# create the job info file
touch ${infofile}

## save a basic info about the job 
#echo -e "Hello world at `date` from user ${USER}!\n" >> ${infofile}
#echo -e "$PBS_JOBID is running on node `hostname -f` in a scratch directory $SCRATCHDIR\n" >> ${infofile}
## copy the outputs back to where qsub was called
#cp ${infofile} ${PBS_O_WORKDIR}/
#cp out_count.txt ${PBS_O_WORKDIR}/

cd PINN/case1_OrnsteinUhlenbeck/
#python grid_search.py
python main_vanilla_pinn.py

cd ../../
cp ${infofile} ${PBS_O_WORKDIR}/
#cp out_count.txt ${PBS_O_WORKDIR}/

# apply automatic cleaner
clean_scratch

```