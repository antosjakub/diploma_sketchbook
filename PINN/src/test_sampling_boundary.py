"""
Tests for sample_hypercube_boundary in sampling.py.

Run with:  python test_sampling_boundary.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D

from sampling import sample_hypercube_boundary


# ─────────────────────────────────────────────────────────────────────────────
# Unit tests
# ─────────────────────────────────────────────────────────────────────────────

def test_unit_normals(n=2000, dims=(1, 2, 3, 5, 8)):
    """Normals must be unit vectors in R^d."""
    for d in dims:
        pts, nrm = sample_hypercube_boundary(n, d)
        norms = nrm.norm(dim=1)
        assert torch.allclose(norms, torch.ones(n), atol=1e-6), \
            f"d={d}: normals not unit (min={norms.min():.4f}, max={norms.max():.4f})"
    print("PASS  test_unit_normals")


def test_on_boundary(n=2000, dims=(1, 2, 3, 5, 8)):
    """Each point must have at least one coordinate == 0 or == 1."""
    for d in dims:
        pts, _ = sample_hypercube_boundary(n, d)
        on_face = ((pts == 0) | (pts == 1)).any(dim=1)
        assert on_face.all(), \
            f"d={d}: {(~on_face).sum()} points not on any face"
    print("PASS  test_on_boundary")


def test_normal_orthogonal_to_face(n=2000, dims=(2, 3, 5)):
    """
    The normal must be perpendicular to every tangent vector of its face.
    Tangent vectors lie in the same face (same fixed dim + value), so we
    check: for a pair of points on the *same* face their difference vector
    is orthogonal to the shared normal.

    Strategy: group by (fixed_dim, fixed_value) and verify that within
    each group the normals are identical and every difference of two points
    is orthogonal to that normal.
    """
    for d in dims:
        pts, nrm = sample_hypercube_boundary(n, d)
        # Reconstruct which dim was fixed and to what value
        face_dim   = nrm.abs().argmax(dim=1)          # index of the unit-normal component
        face_value = ((nrm[range(n), face_dim] + 1) / 2).round().long()  # 0 or 1

        for fd in range(d):
            for fv in (0, 1):
                mask = (face_dim == fd) & (face_value == fv)
                if mask.sum() < 2:
                    continue
                p = pts[mask]   # (k, d)
                nn = nrm[mask]  # (k, d)  — should all be the same e_fd-unit-vec

                # Check normals are identical within group
                assert (nn == nn[0]).all(), f"d={d} fd={fd} fv={fv}: mixed normals"

                # Check tangent orthogonality: (p[i]-p[j]) · n == 0
                diffs = p - p[0:1]            # (k, d) differences from first point
                dots  = (diffs * nn[0]).sum(dim=1)
                assert torch.allclose(dots, torch.zeros(mask.sum()), atol=1e-6), \
                    f"d={d} fd={fd} fv={fv}: tangent not ⊥ normal (max|dot|={dots.abs().max():.2e})"

    print("PASS  test_normal_orthogonal_to_face")


def test_normal_points_outward(n=2000, dims=(2, 3, 5)):
    """(point - centre) · normal > 0  (outward convention)."""
    centre = 0.5
    for d in dims:
        pts, nrm = sample_hypercube_boundary(n, d)
        dot = ((pts - centre) * nrm).sum(dim=1)
        assert (dot > 0).all(), \
            f"d={d}: {(dot<=0).sum()} normals point inward (min dot={dot.min():.4f})"
    print("PASS  test_normal_points_outward")


def test_face_distribution(n=10_000, d=3, tol=0.10):
    """All 2d faces should be roughly equally sampled."""
    pts, nrm = sample_hypercube_boundary(n, d)
    face_dim   = nrm.abs().argmax(dim=1)
    face_value = ((nrm[range(n), face_dim] + 1) / 2).round().long()
    n_faces = 2 * d
    expected = n / n_faces
    for fd in range(d):
        for fv in (0, 1):
            cnt = ((face_dim == fd) & (face_value == fv)).sum().item()
            assert abs(cnt - expected) / expected < tol, \
                f"Face (dim={fd}, val={fv}): count={cnt}, expected≈{expected:.0f}"
    print("PASS  test_face_distribution")


# ─────────────────────────────────────────────────────────────────────────────
# Visual tests
# ─────────────────────────────────────────────────────────────────────────────

def plot_2d(n=200, arrow_scale=0.12, save=None):
    """2-D unit square: boundary points + outward normal arrows."""
    pts, nrm = sample_hypercube_boundary(n, d=2)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_xlim(-0.15, 1.15)
    ax.set_ylim(-0.15, 1.15)
    ax.set_aspect("equal")
    ax.set_title("2-D hypercube boundary — outward normals")

    # colour by face
    face_dim = nrm.abs().argmax(dim=1)
    face_val = ((nrm[range(n), face_dim] + 1) / 2).round().long()
    colours = {(0, 0): "steelblue", (0, 1): "cornflowerblue",
               (1, 0): "firebrick",  (1, 1): "salmon"}
    labels  = {(0, 0): "x₀=0 (left)",   (0, 1): "x₀=1 (right)",
               (1, 0): "x₁=0 (bottom)", (1, 1): "x₁=1 (top)"}

    for fd in range(2):
        for fv in (0, 1):
            mask = (face_dim == fd) & (face_val == fv)
            p = pts[mask].numpy()
            nn = nrm[mask].numpy()
            c = colours[(fd, fv)]
            ax.scatter(p[:, 0], p[:, 1], s=8, color=c, zorder=3)
            for i in range(len(p)):
                ax.annotate("", xy=(p[i, 0] + arrow_scale * nn[i, 0],
                                    p[i, 1] + arrow_scale * nn[i, 1]),
                            xytext=(p[i, 0], p[i, 1]),
                            arrowprops=dict(arrowstyle="->", color=c, lw=1.2))

    patches = [mpatches.Patch(color=colours[k], label=labels[k])
               for k in colours]
    ax.legend(handles=patches, fontsize=8)

    # draw unit square
    sq = plt.Polygon([(0,0),(1,0),(1,1),(0,1)], fill=False,
                     edgecolor="k", lw=1.5)
    ax.add_patch(sq)

    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=150)
        print(f"Saved 2-D plot → {save}")
    else:
        plt.show()


def plot_3d(n=400, arrow_len=0.12, save=None):
    """3-D unit cube: boundary points + outward normal arrows."""
    pts, nrm = sample_hypercube_boundary(n, d=3)

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("3-D hypercube boundary — outward normals")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_zlim(0, 1)
    ax.set_xlabel("x₀"); ax.set_ylabel("x₁"); ax.set_zlabel("x₂")

    face_dim = nrm.abs().argmax(dim=1)
    face_val = ((nrm[range(n), face_dim] + 1) / 2).round().long()

    cmap = {(0,0):"navy", (0,1):"dodgerblue",
            (1,0):"darkgreen", (1,1):"limegreen",
            (2,0):"darkred",  (2,1):"tomato"}

    p = pts.numpy()
    nn = nrm.numpy()
    fd = face_dim.numpy()
    fv = face_val.numpy()

    for i in range(n):
        c = cmap[(fd[i], fv[i])]
        ax.scatter(p[i,0], p[i,1], p[i,2], color=c, s=6, zorder=3)
        ax.quiver(p[i,0], p[i,1], p[i,2],
                  nn[i,0]*arrow_len, nn[i,1]*arrow_len, nn[i,2]*arrow_len,
                  color=c, linewidth=0.8, arrow_length_ratio=0.4)

    patches = [mpatches.Patch(color=cmap[(fd_, fv_)],
                               label=f"dim{fd_}={'0' if fv_==0 else '1'}")
               for fd_ in range(3) for fv_ in (0, 1)]
    ax.legend(handles=patches, fontsize=7)

    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=150)
        print(f"Saved 3-D plot → {save}")
    else:
        plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-plots", action="store_true",
                        help="skip visual tests (unit tests always run)")
    parser.add_argument("--save-dir", default=None,
                        help="if set, save plots to this directory instead of showing")
    args = parser.parse_args()

    print("=== Unit tests ===")
    test_unit_normals()
    test_on_boundary()
    test_normal_orthogonal_to_face()
    test_normal_points_outward()
    test_face_distribution()
    print("All unit tests passed.\n")

    if not args.no_plots:
        save2d = os.path.join(args.save_dir, "boundary_2d.png") if args.save_dir else None
        save3d = os.path.join(args.save_dir, "boundary_3d.png") if args.save_dir else None
        print("=== Visual tests ===")
        plot_2d(save=save2d)
        plot_3d(save=save3d)
