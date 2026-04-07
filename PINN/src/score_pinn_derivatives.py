import torch
from torch.profiler import record_function


def compute_score_and_jacobian(model, X):
    """
    Compute the score s(X) and its Jacobian with respect to all inputs.

    Parameters
    ----------
    model : callable
        Neural network mapping X -> s, where:
        X.shape (batch_size, D)
        s.shape (batch_size, d)
    X : torch.Tensor
        shape = (batch_size, D), with requires_grad=True

    Returns
    -------
    s : torch.Tensor
        s.shape = (batch_size, d)
    jac : torch.Tensor
        Jacobian of s wrt all inputs X
        jac.shape = (batch_size, d, D),
        where jac[:, i, j] = d s_i / d X_j
    """
    if X.ndim != 2:
        raise ValueError("X must have shape (batch_size, D).")
    if not X.requires_grad:
        raise ValueError("X must have requires_grad=True.")

    with record_function("score_forward"):
        s = model(X)

    if s.ndim != 2:
        raise ValueError("model(X) must return shape (batch_size, d).")
    if X.shape[0] != s.shape[0]:
        raise ValueError("Batch size mismatch between X and model(X).")

    _, d = s.shape

    jac_cols = []

    with record_function("score_jacobian"):
        for i in range(d):
            grad_si = torch.autograd.grad(
                outputs=s[:, i].sum(),
                inputs=X,
                create_graph=True,
                retain_graph=True,
            )[0]  # shape: (bs, D)
            jac_cols.append(grad_si.unsqueeze(1))  # shape: (bs, 1, D)

    jac = torch.cat(jac_cols, dim=1)  # shape: (bs, d, D)

    return s, jac

import torch
from torch.profiler import record_function


def compute_score_dt_div(model, X):
    """
    Compute:
        - s(X):      score output
        - dt_s(X):   time derivative of each score component
        - div_s(X):  spatial divergence of the score field

    Parameters
    ----------
    model : callable
        Neural network mapping X -> s, where:
            X has shape (batch_size, D)
            s has shape (batch_size, d)
        Typically D = d + 1, with the last input coordinate being time.
    X : torch.Tensor
        Input tensor of shape (batch_size, D), with requires_grad=True.

    Returns
    -------
    s : torch.Tensor
        Score output of shape (batch_size, d)
    dt_s : torch.Tensor
        Time derivative of the score, shape (batch_size, d),
    div_s : torch.Tensor
        Spatial divergence of the score, shape (batch_size, 1),
    """

    if X.ndim != 2:
        raise ValueError("X must have shape (batch_size, D).")
    if not X.requires_grad:
        raise ValueError("X must have requires_grad=True.")

    bs, D = X.shape

    with record_function("score_forward"):
        s = model(X)

    if s.ndim != 2:
        raise ValueError("model(X) must return shape (batch_size, d).")

    bs_s, d = s.shape
    if bs_s != bs:
        raise ValueError("Batch size mismatch between X and model(X).")

    if D != d + 1:
        raise ValueError(
            f"Expected X.shape[1] = d + 1, but got D={D} and d={d}."
        )

    dt_components = []
    div_s_components = []
    #div_s = torch.zeros(bs, 1, device=X.device, dtype=X.dtype)

    with record_function("score_dt_div"):
        for i in range(d):
            grad_si = torch.autograd.grad(
                outputs=s[:, i].sum(),
                inputs=X,
                create_graph=True,
                retain_graph=True,
            )[0]  # shape: (bs, D)

            # der of si wrt t
            dt_components.append(grad_si[:, -1:])
            # der of si wrt xi
            div_s_components.append(grad_si[:, i:i+1])

            # spatial divergence uses diagonal spatial derivative ∂s_i/∂x_i
            #div_s = div_s + grad_si[:, i:i+1]

    dt_s = torch.cat(dt_components, dim=1)  # shape: (bs, d)
    div_s = torch.sum(torch.cat(div_s_components, dim=1), dim=1).unsqueeze(1) # shape: (bs,1)

    return s, dt_s, div_s





def divergence_of_matrix_field(M, X, spatial_dim=None):
    """
    Compute the divergence of a batch of matrix fields.

    If M has shape (batch_size, d, d), then returns div_M with shape (batch_size, d),
    defined by

        (div M)_i = sum_j ∂ M_{ij} / ∂ x_j

    where derivatives are taken with respect to the spatial coordinates only.

    Parameters
    ----------
    M : torch.Tensor
        Matrix field of shape (batch_size, d, d).
    X : torch.Tensor
        Input tensor of shape (batch_size, D), with requires_grad=True.
        Typically X = [x, t], so D = d + 1.
    spatial_dim : int or None
        Number of spatial dimensions.
        If None, defaults to X.shape[1] - 1, assuming last input is time.

    Returns
    -------
    div_M : torch.Tensor
        Divergence of M, shape (batch_size, d).
    """
    if M.ndim != 3:
        raise ValueError("M must have shape (batch_size, d, d).")
    if X.ndim != 2:
        raise ValueError("X must have shape (batch_size, D).")
    if not X.requires_grad:
        raise ValueError("X must have requires_grad=True.")

    bs, d1, d2 = M.shape
    bs_x, D = X.shape

    if bs != bs_x:
        raise ValueError("Batch size mismatch between M and X.")
    if d1 != d2:
        raise ValueError("M must be square in its last two dimensions.")

    d = d1

    if spatial_dim is None:
        spatial_dim = D - 1  # assume last column is time

    if spatial_dim != d:
        raise ValueError(
            f"Expected spatial_dim == matrix size, got spatial_dim={spatial_dim}, matrix size={d}."
        )

    div_components = []

    with record_function("matrix_divergence"):
        for i in range(d):
            comp_i = 0.0
            for j in range(d):
                grad_Mij = torch.autograd.grad(
                    outputs=M[:, i, j].sum(),
                    inputs=X,
                    create_graph=True,
                    retain_graph=True,
                )[0]  # shape: (bs, D)

                comp_i = comp_i + grad_Mij[:, j]  # derivative wrt x_j only

            div_components.append(comp_i.unsqueeze(1))  # shape: (bs, 1)

    div_M = torch.cat(div_components, dim=1)  # shape: (bs, d)
    return div_M



if __name__ == "__main__":

    # sample X
    X = X.requires_grad_(True)
    s, dt_s, div_s = compute_score_dt_div(score_model, X)
    print(s.shape)      # (bs, d)
    print(dt_s.shape)   # (bs, d)
    print(div_s.shape)  # (bs, 1)

    # sample X
    X = X.requires_grad_(True)
    s, jac = compute_score_and_jacobian(score_model, X)
    # Shapes:
    # s   -> (bs, d)
    # jac -> (bs, d, d+1)

    # Then you can extract yourself:
    s_jac_spatial = jac[:, :, :-1]   # (bs, d, d)
    s_dt = jac[:, :, -1]           # (bs, d)
    div_s = torch.diagonal(s_jac_spatial, dim1=1, dim2=2).sum(dim=1, keepdim=True)  # (bs, 1)

    # Example for M = G G^T
    # M should have shape (bs, d, d)
    div_M = divergence_of_matrix_field(M, X)  # shape (bs, d)