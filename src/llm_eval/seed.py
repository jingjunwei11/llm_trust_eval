from __future__ import annotations

import random


def set_global_seed(seed: int, *, deterministic: bool = True) -> None:
    """Pin global RNG state across libraries for reproducible runs.

    This only fixes *global* randomness: dataset shuffling, dict iteration
    order, dropout/initialization defaults, and the numpy RandomState that
    sklearn's default ``random_state=None`` draws from. Per-sample generation
    determinism is handled separately by passing an explicit ``seed`` to the
    model's ``generate`` (which calls ``torch.manual_seed``).

    Args:
        seed: Integer seed applied to every library.
        deterministic: Enable cuDNN deterministic kernels (slower but stable).
    """
    seed = int(seed)
    # 注意：PYTHONHASHSEED 必须在解释器启动前设置才生效，运行中改 os.environ
    # 对当前进程无效，因此在 run.sh 里 export，而不是在这里。
    random.seed(seed)

    import numpy as np

    np.random.seed(seed)

    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # sklearn exposes no global ``seed()``; estimators created with the default
    # ``random_state=None`` read from the global numpy RandomState seeded above,
    # so no further action is needed.

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
