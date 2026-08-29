"""spkproof - deterministic checks for speaker-verification studies.

Four kinds of check, and they answer different questions:

    check_f0         is this MEASUREMENT contaminated?
                     Pitch trackers fail on rough phonation, in one direction.
    check_descriptor is this DESCRIPTOR an artifact of pitch?
                     Sweep F0 with the spectral envelope held fixed. Anything
                     that moves is measuring harmonic spacing, not the voice.
    check_design     is this COMPARISON already decided by its own structure?
                     Duration tracking condition, impostor pools that drift,
                     enrollment leaking into test.
    score_panel      which encoder can I TRUST for my speakers, my conditions?
                     Benchmark leaderboards are computed on modal read speech.

The first three read a table you already have. The fourth reads a score table
and answers with a paired bootstrap over speakers, corrected for multiplicity.
No audio, no model, no dependencies.

Sources:
Gkilis (2026), "Intra-speaker vocal variation and speaker-embedding
displacement", doi:10.5281/zenodo.21921958 - the F0 checks.
Gkilis (2026), "Speaker encoders disagree about who you are when you shout" -
the design checks, the panel and the descriptor gate, each rule written because
it caught a real error in that work.
"""
from .descriptor import (
    CONTROL_NAME,
    DEFAULT_TOLERANCE,
    SweepSignal,
    SweepValue,
    check_descriptor,
    check_sweep_table,
    f0_sensitive_control,
    measure_sweep,
    sweep_signals,
    write_wav,
)
from .descriptor import Finding as DescriptorFinding
from .design import Finding as DesignFinding
from .design import Trial, check_design
from .f0 import DEFAULT_CEILING_ST, Finding, Utterance, check_f0, semitones
from .panel import (
    Comparison,
    Family,
    PanelResult,
    all_pairs,
    d_prime,
    declare_family,
    equal_error_rate,
    family_summary,
    paired_compare,
    rank_and_warn,
    resample_note,
    score_panel,
    worst_condition,
)
from .stats import holm, holm_adjusted

__version__ = "0.3.0"
__all__ = [
    # measurement
    "check_f0", "Utterance", "Finding", "semitones", "DEFAULT_CEILING_ST",
    # descriptors
    "check_descriptor", "check_sweep_table", "sweep_signals", "measure_sweep",
    "write_wav", "f0_sensitive_control", "SweepSignal", "SweepValue",
    "DescriptorFinding", "DEFAULT_TOLERANCE", "CONTROL_NAME",
    # design
    "check_design", "Trial", "DesignFinding",
    # panel
    "score_panel", "rank_and_warn", "worst_condition", "PanelResult",
    "equal_error_rate", "d_prime", "resample_note",
    # inference
    "paired_compare", "declare_family", "all_pairs", "family_summary",
    "Comparison", "Family", "holm", "holm_adjusted",
    "__version__",
]
