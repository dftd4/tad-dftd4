"""
Calculation of coordination number with various counting functions.
"""

import torch

from .. import defaults
from ..data import cov_rad_d3, pauling_en
from ..typing import CountingFunction, Tensor
from ..utils import real_pairs
from .count import erf_count


def get_coordination_number(
    numbers: Tensor,
    positions: Tensor,
    counting_function: CountingFunction = erf_count,
    rcov: Tensor | None = None,
    en: Tensor | None = None,
    cutoff: Tensor | None = None,
    **kwargs,
) -> Tensor:
    """
    Compute fractional coordination number using an exponential counting function.

    Args:
        numbers (Tensor): Atomic numbers of molecular structure.
        positions (Tensor): Atomic positions of molecular structure
        counting_function (Callable): Calculate weight for pairs.
        rcov (Tensor, optional): Covalent radii for each species. Defaults to `None`.
        cutoff (Tensor, optional): Real-space cutoff. Defaults to `None`.
        kwargs: Pass-through arguments for counting function.

    Raises:
        ValueError: If shape mismatch between `numbers`, `positions` and `rcov`
        is detected.

    Returns:
        cn (Tensor): Coordination numbers for all atoms
    """

    if cutoff is None:
        cutoff = torch.tensor(defaults.D4_CN_CUTOFF, dtype=positions.dtype)

    if rcov is None:
        rcov = cov_rad_d3[numbers]
    rcov = rcov.type(positions.dtype).to(positions.device)

    if en is None:
        en = pauling_en[numbers]
    en = en.type(positions.dtype).to(positions.device)

    if numbers.shape != rcov.shape:
        raise ValueError(
            f"Shape of covalent radii {rcov.shape} is not consistent with "
            f"({numbers.shape})."
        )
    if numbers.shape != positions.shape[:-1]:
        raise ValueError(
            f"Shape of positions ({positions.shape[:-1]}) is not consistent "
            f"with atomic numbers ({numbers.shape})."
        )

    mask = real_pairs(numbers)

    distances = torch.where(
        mask,
        torch.cdist(positions, positions, p=2, compute_mode="use_mm_for_euclid_dist"),
        positions.new_tensor(torch.finfo(positions.dtype).eps),
    )

    # Eq. 6
    endiff = torch.abs(en.unsqueeze(-2) - en.unsqueeze(-1))
    den = defaults.D4_K4 * torch.exp(
        -((endiff + defaults.D4_K5) ** 2.0) / defaults.D4_K6
    )

    rc = rcov.unsqueeze(-2) + rcov.unsqueeze(-1)
    cf = torch.where(
        mask * (distances <= cutoff),
        den * counting_function(distances, rc, **kwargs),
        positions.new_tensor(0.0),
    )
    return torch.sum(cf, dim=-1)
