"""Endpoint objectives, training loss, and deterministic uninitialized-memory regression."""
from types import SimpleNamespace
import importlib
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

import cvxpy as cp
import numpy as np

from papers.sign_pooling_2026.replication import tuning_boundary_validation as T
from papers.sign_pooling_2026.replication.cvxpy_shape_compat import safe_cvxpy_sum_shape


class BoundaryTests(unittest.TestCase):
    def test_fresh_case_needs_no_prior_results(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            (out / 'shards').mkdir()
            with patch.object(T.L, 'METHODS', ('lasso',)):
                T.fit_one(('learned', T.L.DESIGNS[0], 1000, str(out), None))
            cv = T.pd.read_csv(next((out / 'shards').glob('*_cv.csv')))
            self.assertFalse(cv.reused.any())
            self.assertEqual(len(cv), len(T.FRACTIONS))
            self.assertTrue(np.isfinite(cv.cv_mse).all())

    def test_uninitialized_shape_bug_fails_then_fixed_path_passes(self):
        module = importlib.import_module('cvxpy.atoms.affine.sum')
        original = module.Sum.shape_from_args
        poisoned = np.array([np.inf, -np.inf])
        argument = SimpleNamespace(args=[SimpleNamespace(shape=(2,))], axis=None, keepdims=False)
        with patch.object(module.np, 'empty', return_value=poisoned), np.errstate(invalid='raise'):
            with self.assertRaisesRegex(FloatingPointError, 'invalid value encountered in reduce'):
                original(argument)
            with safe_cvxpy_sum_shape():
                self.assertEqual(module.Sum.shape_from_args(argument), ())
        self.assertIs(module.Sum.shape_from_args, original)

    def test_shape_and_axis_validation_agree(self):
        shapes = [(), (3,), (2,3), (2,3,4)]
        with safe_cvxpy_sum_shape():
            for shape in shapes:
                x = np.zeros(shape)
                for axis in (None, 0, -1, (), (0,), (0,1), (0,0), 5):
                    for keepdims in (False, True):
                        try:
                            expected = np.sum(x, axis=axis, keepdims=keepdims).shape
                        except (ValueError, TypeError):
                            with self.assertRaises(ValueError):
                                cp.sum(cp.Constant(x),axis=axis,keepdims=keepdims)
                        else:
                            self.assertEqual(cp.sum(cp.Constant(x),axis=axis,
                                                    keepdims=keepdims).shape,expected)

    def test_zero_penalty_matches_orthogonal_projection(self):
        x = np.vstack([np.eye(3),-np.eye(3)])
        truth = np.array([[2.,-3.,4.],[-2.,3.,4.]])
        prep = T.B.Prepared(x,x@truth.T,np.zeros(3),np.ones(3),
                            np.zeros(2),np.ones(2),np.array([0,0]))
        signs = np.array([[1.,1.,0.],[np.nan,-1.,np.nan]])
        np.testing.assert_allclose(T.zero_penalty(prep,signs),[[2,0,0],[-2,0,4]],atol=1e-12)

    def test_pure_group_solution_matches_closed_form(self):
        x = np.vstack([np.eye(3),-np.eye(3)])*np.sqrt(3)
        truth = np.array([[2.,-3.,4.],[-2.,3.,4.]])
        prep = T.B.Prepared(x,x@truth.T,np.zeros(3),np.ones(3),
                            np.zeros(2),np.ones(2),np.array([0,0]))
        original_protocol = T.protocol

        def unlicensed_protocol(candidate):
            result = original_protocol(candidate)
            result.solver = 'CLARABEL'
            return result

        for method,axis in [('hcgl_unconstrained',1),('fcgl_unconstrained',0)]:
            with patch.object(T, 'protocol', side_effect=unlicensed_protocol):
                b,_,lam,_ = T.path(prep,method,T.L.Candidate(.5,0,1.5),[.3,1.])
            norm=np.linalg.norm(truth,axis=axis,keepdims=True)
            expected=truth*np.maximum(1-lam[0]*np.sqrt(2)/(2*norm),0)
            np.testing.assert_allclose(b[0],expected,atol=2e-4)
            self.assertLess(np.max(abs(b[1])),2e-4)

    def test_surface_reuses_only_valid_training_cells(self):
        rng=np.random.default_rng(32)
        x,y=rng.normal(size=(30,4)),rng.normal(size=(30,3))
        calls=[]
        def path(prep,method,candidate,fractions,known_labels):
            calls.append(len(prep.x))
            return np.zeros((len(fractions),3,4)),np.zeros((len(fractions),3)),None,None
        with patch.object(T,'path',side_effect=path):
            cv,_=T.surface(x,y,'lasso',42,'learned')
        expected=sum(np.square((y[v]-y[t].mean(0))/y[t].std(0)).sum()/3/30
                     for t,v in T.B.cv_splits(30,3,42))
        np.testing.assert_allclose(cv.cv_mse,expected)
        self.assertEqual(calls,[20,20,20])
        bad=cv.copy()
        bad.loc[0,'seed']=99
        with self.assertRaises(ValueError):
            T.surface(x,y,'lasso',42,'learned',bad)


if __name__=='__main__':
    unittest.main()
