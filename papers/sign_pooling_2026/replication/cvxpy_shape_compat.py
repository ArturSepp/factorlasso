"""Scoped workaround for CVXPY issue3235, observed in installed CVXPY1.9.2.

https://github.com/cvxpy/cvxpy/issues/3235
Sum.shape_from_args reduces np.empty floats merely to infer an output shape.
Uninitialized payloads can raise invalid/overflow even with finite model inputs.
Within this context only, use initialized integer zeros for that shape probe.
Numeric evaluation, canonicalization and solver settings are unchanged. The
original method is restored on exit, including on exceptions; no installed
package or factorlasso source is modified. Use in single-threaded fit workers.
"""
from contextlib import contextmanager
import hashlib
import inspect

import cvxpy
from cvxpy.atoms.affine.sum import Sum
import numpy as np
from numpy.exceptions import AxisError


def safe_sum_shape(self):
    """Preserve NumPy axis validation and scalar/keepdims semantics, without floats."""
    try:
        return np.sum(np.zeros(self.args[0].shape, dtype=np.int8),
                      axis=self.axis, keepdims=self.keepdims).shape
    except (ValueError, AxisError, TypeError) as error:
        raise ValueError(f'Invalid arguments for cp.sum: {error}') from error


def compatibility_record():
    source = inspect.getsource(Sum.shape_from_args)
    return dict(cvxpy_version=cvxpy.__version__, shape_source=source,
                shape_source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                affected='np.empty(' in source,
                upstream_issue='https://github.com/cvxpy/cvxpy/issues/3235')


@contextmanager
def safe_cvxpy_sum_shape():
    """Patch only the known uninitialized shape probe and restore it afterwards."""
    original = Sum.shape_from_args
    if original is safe_sum_shape:
        yield
        return
    source = inspect.getsource(original)
    affected = 'np.empty(self.args[0].shape)' in source
    if affected:
        Sum.shape_from_args = safe_sum_shape
    try:
        yield
    finally:
        if affected:
            Sum.shape_from_args = original
