"""spkproof - deterministic checks for speaker-verification studies.

The checks here come from a published experiment and its diagnostics:
Gkilis (2026), "Intra-speaker vocal variation and speaker-embedding
displacement", doi:10.5281/zenodo.21921958.
"""
from .f0 import DEFAULT_CEILING_ST, Finding, Utterance, check_f0, semitones

__version__ = "0.1.0"
__all__ = ["check_f0", "Utterance", "Finding", "semitones", "DEFAULT_CEILING_ST", "__version__"]
