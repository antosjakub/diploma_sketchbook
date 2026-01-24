import torch

from functorch import vmap
from functorch import jacrev
from functorch import jvp

device = "cuda" if torch.cuda.is_available() else "cpu"


class Potential():
    def value(self, points): 
        return vmap(self.value_point)(points)

    def grad(self, points):
        return vmap(jacrev(self.value_point))(points)

    def hessian_point(self, point):
        return jacrev(jacrev(self.value_point))(point)

    def hessian(self, points):
        return vmap(self.hessian_point)(points)

    def laplace(self, points):
        points = points.to(device).requires_grad_(True)
        return torch.einsum("j i i -> j", self.hessian(points))

    def laplace_hutchinson_point(self, point):
        num_vectors = 100
        vectors = torch.randn(num_vectors, len(point))
        grad = lambda point: jacrev(self.value_point)(point)
        jvp = lambda v: torch.dot(v, jvp(grad, (point,), (v,))[1])
        return torch.sum(vmap(jvp)(vectors))/num_vectors

    def laplace_hutchinson(self, points):
        return vmap(self.laplace_hutchinson_point, randomness="same")(points)

    def laplace_new_point(self, point):
        vectors = torch.eye(len(point))
        grad = lambda point: jacrev(self.value_point)(point)
        jvp = lambda v: torch.dot(v, jvp(grad, (point,), (v,))[1])
        return torch.sum(vmap(jvp)(vectors))

    def laplace_new(self, points):
        return vmap(self.laplace_new_point, randomness="same")(points)


class LJ(Potential):
    def __init__(self, epsilon, r0, device):
        super().__init__()
        self.device = device
        self.epsilon = epsilon
        self.r0 = r0

    def value_point(self, point):
        point = point.reshape(-1,3)
        m = point.shape[0]
        result = 0
        r = lambda i, j: torch.linalg.vector_norm(point[i, :]-point[j, :])
        for i in range(m):
          for j in range(i+1,m):
            rij = self.r0/r(i,j)
            result = result + 4 * self.epsilon * (rij**12-rij**6)
        return result