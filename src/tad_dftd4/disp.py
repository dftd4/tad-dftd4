import torch

from . import data
from .model import D4Model
from .damping import rational_damping, dispersion_atm
from ._typing import DampingFunction, Tensor, Any
from .utils import real_pairs
from ncoord import get_coordination_number_d4, erf_count
from charges import get_charges
from . import defaults
from .cutoff import Cutoff

from ._typing import CountingFunction, WeightingFunction


def dftd4(
    numbers: Tensor,
    positions: Tensor,
    charge: Tensor,
    param: dict[str, Tensor],
    *,
    model: D4Model | None = None,
    rcov: Tensor | None = None,
    r4r2: Tensor | None = None,
    q: Tensor | None = None,
    cutoff: Cutoff | None = None,
    counting_function: CountingFunction = erf_count,
    damping_function: DampingFunction = rational_damping,
) -> Tensor:
    if cutoff is None:
        cutoff = Cutoff()
    if rcov is None:
        rcov = data.cov_rad_d3[numbers].type(positions.dtype).to(positions.device)
    if r4r2 is None:
        r4r2 = (
            data.sqrt_z_r4_over_r2[numbers].type(positions.dtype).to(positions.device)
        )
    if numbers.shape != positions.shape[:-1]:
        raise ValueError(
            "Shape of positions is not consistent with atomic numbers.",
        )
    if numbers.shape != r4r2.shape:
        raise ValueError(
            "Shape of expectation values is not consistent with atomic numbers.",
        )
    if model is None:
        model = D4Model()
    if q is None:
        q = get_charges(numbers, positions, charge, cutoff=cutoff.cn_eeq)

    cn = get_coordination_number_d4(
        numbers, positions, counting_function, rcov, cutoff=cutoff.cn
    )
    weights = model.weight_references(numbers, cn, q)
    c6 = model.get_atomic_c6(numbers, weights)

    energy = dispersion2(
        numbers, positions, param, c6, r4r2, damping_function, cutoff.disp2
    )

    return energy


def dispersion2(
    numbers: Tensor,
    positions: Tensor,
    param: dict[str, Tensor],
    c6: Tensor,
    r4r2: Tensor,
    damping_function: DampingFunction,
    cutoff: Tensor,
    **kwargs: Any,
) -> Tensor:
    """
    Calculate dispersion energy between pairs of atoms.

    Parameters
    ----------
    numbers : Tensor
        Atomic numbers of the atoms in the system.
    positions : Tensor
        Cartesian coordinates of the atoms in the system.
    param : dict[str, Tensor]
        DFT-D3 damping parameters.
    c6 : Tensor
        Atomic C6 dispersion coefficients.
    r4r2 : Tensor
        r⁴ over r² expectation values of the atoms in the system.
    damping_function : Callable
        Damping function evaluate distance dependent contributions.
        Additional arguments are passed through to the function.

    Returns
    -------
    Tensor
        Atom-resolved two-body dispersion energy.
    """
    mask = real_pairs(numbers, diagonal=False)
    distances = torch.where(
        mask,
        torch.cdist(positions, positions, p=2, compute_mode="use_mm_for_euclid_dist"),
        positions.new_tensor(torch.finfo(positions.dtype).eps),
    )

    qq = 3 * r4r2.unsqueeze(-1) * r4r2.unsqueeze(-2)
    c8 = c6 * qq

    t6 = torch.where(
        mask * (distances <= cutoff),
        damping_function(6, distances, qq, param, **kwargs),
        positions.new_tensor(0.0),
    )
    t8 = torch.where(
        mask * (distances <= cutoff),
        damping_function(8, distances, qq, param, **kwargs),
        positions.new_tensor(0.0),
    )

    e6 = -0.5 * torch.sum(c6 * t6, dim=-1)
    e8 = -0.5 * torch.sum(c8 * t8, dim=-1)

    s6 = param.get("s6", positions.new_tensor(defaults.S6))
    s8 = param.get("s8", positions.new_tensor(defaults.S8))
    return s6 * e6 + s8 * e8


def dispersion3(
    numbers: Tensor,
    positions: Tensor,
    param: dict[str, Tensor],
    cutoff: Tensor,
    c6: Tensor,
) -> Tensor:
    """
    Three-body dispersion term. Currently this is only a wrapper for the
    Axilrod-Teller-Muto dispersion term.

    Parameters
    ----------
    numbers : Tensor
        Atomic numbers of the atoms in the system.
    positions : Tensor
        Cartesian coordinates of the atoms in the system.
    param : dict[str, Tensor]
        Dictionary of dispersion parameters. Default values are used for
        missing keys.
    cutoff : Tensor
        Real-space cutoff.
    c6 : Tensor
        Atomic C6 dispersion coefficients.

    Returns
    -------
    Tensor
        Atom-resolved three-body dispersion energy.
    """
    s9 = param.get("s9", positions.new_tensor(defaults.S9))
    a1 = param.get("a1", positions.new_tensor(defaults.A1))
    a2 = param.get("a2", positions.new_tensor(defaults.A2))
    alp = param.get("alp", positions.new_tensor(defaults.ALP))

    return dispersion_atm(numbers, positions, cutoff, c6, s9, a1, a2, alp)
