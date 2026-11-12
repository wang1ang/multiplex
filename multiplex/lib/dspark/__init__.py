"""Code vendored from mlx-dspark 0.20.4 (https://github.com/ARahim3/mlx-dspark, MIT,
see LICENSE). Only what multiplex uses is taken; every file depends on nothing but
``mlx``. Re-sync by copying again.

- ``skinny_qmm.py``, ``small_m_qmm.py`` (whole files): small-M quantized matmul for
  the speculative verify window.
- ``dflash2.py``: ``DFlashGroupedConv`` + ``CandidateSelector`` extracted from
  ``dflash_model.py``, the DFlash 2 drafter components.
"""
