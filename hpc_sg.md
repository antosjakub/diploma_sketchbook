
# SG++ setup on cluster


interactive job
```
qsub -I -l walltime=02:00:00 -l select=1:ncpus=1:cpu_vendor=amd:mem=10gb:scratch_local=10gb
```

redirect temporary files into scratch
```
mkdir $SCRATCHDIR/tmp
export TMPDIR=$SCRATCHDIR/tmp
```

## software

- scons (written in python, thus needs python)
- swig (for python bindings)
- gcc
- eigen

```
module add python/3.11.11-gcc-10.2.1-555dlyc
module add gcc
module add eigen
module add swig
```
those two will not work
```
module add scons
module add boost
```
need to install via pip...

env var my home
```
my_home=/storage/praha1/home/simulant_antos
```

### setup python 
create folder for stuff
```
export PYTHONUSERBASE=$my_home/venv_sg_amd
mkdir $PYTHONUSERBASE
```
update pip
```
python3 -m pip install --user --upgrade pip
```
scons
```
python3 -m pip install --user scons
```
point python there
```
export PATH=$PYTHONUSERBASE/bin:$PATH
export PYTHONPATH=$PYTHONUSERBASE/lib/python3.11/site-packages:$PYTHONPATH
```

#### setup numpy headers as well
```
python3 -m pip install --user numpy
```
expose numpy headers to compiler
```
export NUMPY_INCLUDE=$(python3 -c "import numpy; print(numpy.get_include())")
export CPATH="$NUMPY_INCLUDE:$CPATH"
```
check that the header exists
```
ls "$NUMPY_INCLUDE/numpy/arrayobject.h"
```




### SG
make the install storage and go there
```
mkdir $my_home/software_sg_amd
cd $my_home/software_sg_amd
```

```
git clone https://github.com/SGpp/SGpp.git
cd SGpp/
```

check how many cores yo got
```
nproc
```

then use slightly less - like 8
```
scons -j 8 PYDOC=0 SG_JAVA=0
```

add locations to paths
```
export sg_path=$my_home/software_sg_amd/SGpp
export LD_LIBRARY_PATH="$sg_path/lib:$LD_LIBRARY_PATH"
export PYTHONPATH="$sg_path/lib/:$PYTHONPATH"
```


how to setup python bindings later on (see the very end): https://github.com/SGpp/SGpp/wiki/Linux-(GCC-Clang-ICC)