try:
    from .training_repa_decouple_baseline import REPATrainer as _REPATrainer
except ImportError:
    from training_repa_decouple_baseline import REPATrainer as _REPATrainer


class REPATrainer(_REPATrainer):
    """Encoder-only REPA trainer without DINO-side target normalization."""

    pass
