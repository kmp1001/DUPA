try:
    from .training_repa_enc_norm import REPATrainer as _REPATrainer
except ImportError:
    from training_repa_enc_norm import REPATrainer as _REPATrainer


class REPATrainer(_REPATrainer):
    """Encoder-only REPA trainer with DINO-side target normalization."""

    pass
