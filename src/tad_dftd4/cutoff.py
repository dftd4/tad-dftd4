from ._typing import Tensor
from . import defaults
import torch


class Cutoff:
    """
    Collection of real-space cutoffs.
    """

    def __init__(
        self,
        disp2: float | Tensor = defaults.D4_DISP2_CUTOFF,
        disp3: float | Tensor = defaults.D4_DISP3_CUTOFF,
        cn: float | Tensor = defaults.D4_CN_CUTOFF,
        cn_eeq: float | Tensor = defaults.D4_CN_EEQ_CUTOFF,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        if isinstance(disp2, float):
            disp2 = torch.tensor(disp2, device=device, dtype=dtype)
        if isinstance(disp3, float):
            disp3 = torch.tensor(disp3, device=device, dtype=dtype)
        if isinstance(cn, float):
            cn = torch.tensor(cn, device=device, dtype=dtype)
        if isinstance(cn_eeq, float):
            cn_eeq = torch.tensor(cn_eeq, device=device, dtype=dtype)

        self.disp2 = disp2
        self.disp3 = disp3
        self.cn = cn
        self.cn_eeq = cn_eeq
