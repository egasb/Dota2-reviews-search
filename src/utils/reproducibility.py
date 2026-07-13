def set_seed(seed: int = 20260505) -> None:
    """Set seeds across random, numpy, and torch for strict reproducibility."""
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    # Lazy import of torch
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
