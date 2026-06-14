
import pysgpp
import matplotlib.pyplot as plt

def return_grid_pnts(grid_storage):
    xs = []
    ys = []
    for i in range(grid_storage.getSize()):
        gp = grid_storage.getPoint(i)
        x = gp.getStandardCoordinate(0)
        y = gp.getStandardCoordinate(1)
        xs.append(x)
        ys.append(y)
    return xs, ys


import math
#f = lambda x0, x1: 16.0 * (x0-1)*x0 * (x1-1)*x1*x1
#f = lambda x0, x1: math.sin(math.pi*x0)*math.e**(-10*x1)
#f = lambda x0, x1: math.sin(math.pi*x0)*math.e**(-10*x1)
#f = lambda x0, x1: 3.9
f = lambda x0, x1: 1/( (x0-0.45)**2 + (x1-0.45)**2 )

dim = 2
#grid = pysgpp.Grid.createLinearGrid(dim)
grid = pysgpp.Grid.createLinearBoundaryGrid(dim)
gridStorage = grid.getStorage()

level = 3
gridGen = grid.getGenerator()
gridGen.regular(level)
print("number of initial grid points:    {}".format(gridStorage.getSize()))

ref_steps = 10

### plot
xs, ys = return_grid_pnts(gridStorage)
plt.scatter(xs,ys, c='b')
plt.savefig("grid_1.png")
plt.close()

alpha = pysgpp.DataVector(gridStorage.getSize())
print("length of alpha vector:           {}".format(alpha.getSize()))

# Obtain function values and refine adaptively
for refnum in range(ref_steps):
    # set function values in alpha
    for i in range(gridStorage.getSize()):
        gp = gridStorage.getPoint(i)
        alpha[i] = f(gp.getStandardCoordinate(0), gp.getStandardCoordinate(1))
    pysgpp.createOperationHierarchisation(grid).doHierarchisation(alpha)

    gridGen.refine(pysgpp.SurplusRefinementFunctor(alpha, 1))
    print("refinement step {}, new grid size: {}".format(refnum+1, gridStorage.getSize()))
    alpha.resizeZero(gridStorage.getSize())

for refnum in range(ref_steps):
    for i in range(gridStorage.getSize()):
        gp = gridStorage.getPoint(i)
        alpha[i] = f(gp.getStandardCoordinate(0), gp.getStandardCoordinate(1))
    pysgpp.createOperationHierarchisation(grid).doHierarchisation(alpha)

    removedSeq = pysgpp.SizeVector()
    gridGen.coarsen(pysgpp.SurplusCoarseningFunctor(alpha, 2*dim, 1e-7), removedSeq)
    print("coarsening step {}, new grid size: {}".format(refnum+1, gridStorage.getSize()))
    if removedSeq.size() > 0:
        print("- removed indices:", [removedSeq[i] for i in range(removedSeq.size())])
    alpha.resizeZero(gridStorage.getSize())

### plot
xs, ys = return_grid_pnts(gridStorage)
plt.scatter(xs,ys, c='b')
plt.savefig("grid_2.png")
plt.close()