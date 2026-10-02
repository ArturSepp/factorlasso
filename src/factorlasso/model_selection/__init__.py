"""Selection of the regularisation weight.

:class:`LassoModelCV` selects ``reg_lambda`` by expanding-window out-of-sample fit;
:class:`LassoModelDiagonalityCV` selects the smallest model whose held-out residuals pass the
strict-factor-structure test. The names are also exported from :mod:`factorlasso`.
"""

from factorlasso.model_selection._cv import LassoModelCV
from factorlasso.model_selection._diagonality import LassoModelDiagonalityCV

__all__ = [
    "LassoModelCV",
    "LassoModelDiagonalityCV",
]
