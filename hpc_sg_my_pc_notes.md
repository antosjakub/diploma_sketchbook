

## SG++ setup on fedora laptop

sudo dnf install
- git
- make
- cmake
- lapack-devel
- openblas-devel
- tbb-devel

- eigen3-devel

- gcc
- scons
- swig (for python bindings)

```
cd SGpp/
scons -j$(nproc)
```


```
export PYTHONPATH=/home/antos_j/diploma_sketchbook/SG/external/SGpp/lib/pysgpp:$PYTHONPATH
export LD_LIBRARY_PATH=/home/antos_j/diploma_sketchbook/SG/external/SGpp/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH="$(pwd)/lib/sgpp:$LD_LIBRARY_PATH"
```

create sgpp_pde_solver

```
g++ -std=c++14 -O3 main.cpp \
  -I base/src \
  -I pde/src \
  -I solver/src \
  -L lib/sgpp \
  -lsgppbase -lsgpppde -lsgppsolver \
  -o sgpp_pde_solver
```



https://docs.metacentrum.cz/en/docs/computing/resources/resources

not needed?





