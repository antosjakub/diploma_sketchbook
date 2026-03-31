"""
Tests for compute_derivatives_fd against autodiff (compute_derivatives).
Run with:  python test_derivatives.py
"""
import torch
from derivatives import compute_derivatives, compute_derivatives_fd
from architecture import PINN


class AnalyticModel(torch.nn.Module):
    """u(x0, x1, t) = sin(x0) + x1^2 + t
       u_t = 1,  u_x0x0 = -sin(x0),  u_x1x1 = 2
    """
    def forward(self, X):
        return torch.sin(X[:, 0:1]) + X[:, 1:2] ** 2 + X[:, 2:3]


def test_accuracy():
    D, N = 3, 512
    model = AnalyticModel()
    X = torch.randn(N, D) * 0.5

    X.requires_grad_(True)
    u_ad, g_ad, l_ad = compute_derivatives(model, X)
    X.requires_grad_(False)

    print("Accuracy vs autodiff (analytic model):")
    print(f"  {'h':>6}  {'grad_err':>10}  {'lap_err':>10}")
    for h in [1e-1, 1e-2, 1e-3]:
        u_fd, g_fd, l_fd = compute_derivatives_fd(model, X, h=h)
        g_err = (g_ad - g_fd).abs().max().item()
        l_err = (l_ad - l_fd).abs().max().item()
        print(f"  {h:.0e}  {g_err:10.2e}  {l_err:10.2e}")


def test_grad_flows():
    D, N = 3, 256
    model = PINN(D, layers=[64, 64])
    X = torch.randn(N, D)

    u, g, l = compute_derivatives_fd(model, X)
    loss = (g[:, -1:] - l.sum(1, keepdim=True)).pow(2).mean()
    loss.backward()

    has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.parameters()
    )
    assert has_grad, "No gradients reached model parameters"
    print("Gradient flow: OK")


def test_batch_size_scales():
    """Forward pass batch is N*(1 + 2D), independent of d."""
    import time
    N = 1024
    for d in [5, 10, 15, 20]:
        D = d + 1
        model = PINN(D, layers=[128, 128])
        X = torch.randn(N, D)
        t0 = time.time()
        for _ in range(100):
            compute_derivatives_fd(model, X)
        print(f"  d={d:2d}  100x FD: {time.time() - t0:.3f}s  (batch={N*(1+2*D)})")


if __name__ == "__main__":
    test_accuracy()
    print()
    test_grad_flows()
    print()
    print("Timing (single forward pass, batch = N*(1+2D)):")
    test_batch_size_scales()
