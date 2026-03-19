

sudo dnf install
- git
- make
- cmake
- gcc
- lapack-devel
- openblas-devel
- eigen3-devel
- tbb-devel
- scons
- swig (for python bindings)

---
cd external/SGpp/

scons -j$(nproc)

#export LD_LIBRARY_PATH="$(pwd)/lib/sgpp:$LD_LIBRARY_PATH"

export PYTHONPATH=/home/antos_j/diploma_sketchbook/SG/external/SGpp/lib/pysgpp:$PYTHONPATH
export LD_LIBRARY_PATH=/home/antos_j/diploma_sketchbook/SG/external/SGpp/lib:$LD_LIBRARY_PATH


#create sgpp_pde_solver
#
#g++ -std=c++14 -O3 main.cpp \
#  -I base/src \
#  -I pde/src \
#  -I solver/src \
#  -L lib/sgpp \
#  -lsgppbase -lsgpppde -lsgppsolver \
#  -o sgpp_pde_solver




