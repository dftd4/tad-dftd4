import torch

from .damping import rational_damping
from .utils import real_pairs
from .typing import DampingFunction, Tensor


def dispersion(
    numbers: Tensor,
    positions: Tensor,
    c6: Tensor,
    rvdw: Tensor | None = None,
    r4r2: Tensor | None = None,
    damping_function: DampingFunction = rational_damping,
    cutoff: Tensor | None = None,
    s6: float = 1.0,
    s8: float = 1.0,
    **kwargs
) -> Tensor:
    """
    Calculate dispersion energy between pairs of atoms.
    Parameters
    ----------
    numbers : Tensor
        Atomic numbers of the atoms in the system.
    positions : Tensor
        Cartesian coordinates of the atoms in the system.
    c6 : Tensor
        Atomic C6 dispersion coefficients.
    rvdw : Tensor
        Van der Waals radii of the atoms in the system.
    r4r2 : Tensor
        r⁴ over r² expectation values of the atoms in the system.
    damping_function : Callable
        Damping function evaluate distance dependent contributions.
        Additional arguments are passed through to the function.
    s6 : float
        Scaling factor for the C6 interaction.
    s8 : float
        Scaling factor for the C8 interaction.
    """
    if cutoff is None:
        cutoff = torch.tensor(50.0, dtype=positions.dtype)
    if r4r2 is None:
        r4r2 = data.sqrt_z_r4_over_r2[numbers].type(positions.dtype)
    if rvdw is None:
        rvdw = data.vdw_rad_d3[numbers.unsqueeze(-1), numbers.unsqueeze(-2)].type(
            positions.dtype
        )
    if numbers.shape != positions.shape[:-1]:
        raise ValueError("Shape of positions is not consistent with atomic numbers")
    if numbers.shape != r4r2.shape:
        raise ValueError(
            "Shape of expectation values is not consistent with atomic numbers"
        )

    eps = torch.tensor(torch.finfo(positions.dtype).eps, dtype=positions.dtype)
    mask = real_pairs(numbers, diagonal=False)
    distances = torch.where(
        mask,
        torch.cdist(positions, positions, p=2, compute_mode="use_mm_for_euclid_dist"),
        eps,
    )

    qq = 3 * r4r2.unsqueeze(-1) * r4r2.unsqueeze(-2)
    c8 = c6 * qq

    t6 = torch.where(
        mask * (distances <= cutoff),
        damping_function(6, distances, rvdw, qq, **kwargs),
        torch.tensor(0.0, dtype=distances.dtype),
    )
    t8 = torch.where(
        mask * (distances <= cutoff),
        damping_function(8, distances, rvdw, qq, **kwargs),
        torch.tensor(0.0, dtype=distances.dtype),
    )

    e6 = -0.5 * torch.sum(c6 * t6, dim=-1)
    e8 = -0.5 * torch.sum(c8 * t8, dim=-1)

    return s6 * e6 + s8 * e8
