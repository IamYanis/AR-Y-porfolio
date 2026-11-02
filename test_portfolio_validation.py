import tempfile
import unittest
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
import pandas as pd
from risk_models import shrink_covariance
from portfolio_validation import minimum_risk, hold_period, fit_covariances, run_validation
from portfolio_validation import (compute_frontier, maximum_sharpe, return_limits,
                                  feasible_cloud, exclusion_weights)

class ValidationTests(unittest.TestCase):
    def test_covariance_singular_and_cash(self):
        rng=np.random.default_rng(2)
        x=rng.normal(0,.01,(100,3))
        x=np.column_stack([x,x[:,0]])
        c,s=shrink_covariance(x)
        self.assertGreater(np.linalg.eigvalsh(c).min(),0)
        self.assertTrue(0<=s<=1)
        np.testing.assert_allclose(c,shrink_covariance(x+5)[0],atol=1e-14)
        frame=pd.DataFrame(x,columns=['A','B','C','D'])
        frame['CASH_GBP']=.0001
        matrices,_=fit_covariances(frame)
        for cov in matrices.values():
            np.testing.assert_array_equal(cov[-1],np.zeros(5))

    def test_optimisation_nested_universe_and_target(self):
        cov=np.diag([.04,.09,.01,0.])
        old=[(0,.6),(0,.6),(0,0),(.2,.2)]
        new=[(0,.6),(0,.6),(0,.6),(.2,.2)]
        mu=np.array([.08,.1,.05,.03])
        a=minimum_risk(cov,old,mu,.075)
        b=minimum_risk(cov,new,mu,.075)
        self.assertAlmostEqual(b@mu,.075,places=7)
        self.assertLessEqual(b@cov@b,a@cov@a+1e-9)
        self.assertAlmostEqual(b[-1],.2)

    def test_minimum_risk_matches_analytic_solution(self):
        cov=np.diag([.04,.09,0.])
        w=minimum_risk(cov,[(0,.8),(0,.8),(.2,.2)])
        np.testing.assert_allclose(w,[.8*.09/.13,.8*.04/.13,.2],atol=2e-6)

    def test_hold_period_weight_drift(self):
        test=pd.DataFrame([[.1,0],[0,.1]])
        daily,end=hold_period(test,np.array([.5,.5]))
        np.testing.assert_allclose(daily,[.05,1.1/1.05-1])
        np.testing.assert_allclose(end,[.5,.5])

    def test_future_does_not_change_prior_weights(self):
        rng=np.random.default_rng(4)
        values=rng.normal(.0003,.01,(80,5))
        r=pd.DataFrame(values,index=pd.bdate_range('2025-01-01',periods=80),columns=list('ABCDE'))
        r['CASH_GBP']=.0001
        holdings={'A':20,'B':20,'C':20,'D':20,'E':0,'CASH_GBP':20}
        with tempfile.TemporaryDirectory() as folder:
            a=run_validation(r,holdings,{'E':0},.5,folder,40,20)
            changed=r.copy(); changed.iloc[60:,:5]*=10
            b=run_validation(changed,holdings,{'E':0},.5,folder,40,20)
            wa=a['validation_weights'];wb=b['validation_weights']
            pd.testing.assert_frame_equal(wa[wa.Fold==1],wb[wb.Fold==1])
            self.assertEqual(len(a['rolling_validation']),16) # 2 cash levels x 2 estimators x 2 universes x 2 folds
            f=a['rolling_validation']
            self.assertTrue((f.Period_Return_Cost_25bps<=f.Period_Return_Cost_0bps+1e-12).all())
            self.assertTrue((pd.to_datetime(f.Train_End)<pd.to_datetime(f.Test_Start)).all())


class FrontierTests(unittest.TestCase):
    def setUp(self):
        self.mu = np.array([.08, .10, .05, .03])
        self.cov = np.diag([.04, .09, .01, 0.])
        self.old = [(0, .6), (0, .6), (0, 0), (.2, .2)]
        self.new = [(0, .6), (0, .6), (0, .6), (.2, .2)]

    def test_branches_constraints_and_nested_frontiers(self):
        for bounds in [self.old, self.new]:
            f = compute_frontier(self.cov, self.mu, bounds, 12, include_inefficient=True)
            turning = f['minvar_weights'] @ self.mu
            self.assertTrue((f['returns'] >= turning-1e-8).all())
            self.assertTrue((f['inefficient_returns'] <= turning+1e-8).all())
            self.assertAlmostEqual(f['inefficient_returns'][-1], turning, places=7)
            self.assertAlmostEqual(f['inefficient_vols'][-1], f['vols'][0], places=7)
            for key in ['weights', 'inefficient_weights']:
                w = f[key]
                np.testing.assert_allclose(w.sum(axis=1), 1, atol=1e-7)
                np.testing.assert_allclose(w[:, -1], .2, atol=1e-9)
                self.assertTrue((w >= np.array(bounds)[:, 0]-1e-8).all())
                self.assertTrue((w <= np.array(bounds)[:, 1]+1e-8).all())
        for target in np.linspace(*return_limits(self.mu, self.old), 7):
            old = minimum_risk(self.cov, self.old, self.mu, target)
            new = minimum_risk(self.cov, self.new, self.mu, target)
            self.assertAlmostEqual(new @ self.mu, target, places=7)
            self.assertLessEqual(new @ self.cov @ new, old @ self.cov @ old+1e-8)

    def test_maximum_sharpe_matches_analytic_tangency(self):
        cov = np.diag([.04, .09, 0.])
        mu = np.array([.08, .13, .03])
        bounds = [(0, .8), (0, .8), (.2, .2)]
        frontier = compute_frontier(cov, mu, bounds, 12)
        w = maximum_sharpe(cov, mu, bounds, frontier, .03)
        risky = (mu[:2]-.03) / np.diag(cov)[:2]
        np.testing.assert_allclose(w, np.r_[.8*risky/risky.sum(), .2], atol=3e-6)
        self.assertGreaterEqual(w @ mu, frontier['returns'][0]-1e-7)
        self.assertAlmostEqual(w.sum(), 1, places=7)

    def test_degenerate_returns_and_undefined_sharpe(self):
        mu = np.full(4, .03)
        f = compute_frontier(self.cov, mu, self.new, 5, include_inefficient=True)
        self.assertEqual(len(f['returns']), 1)
        self.assertEqual(len(f['inefficient_returns']), 0)
        zero = compute_frontier(np.zeros((4, 4)), mu, self.new, 5)
        with self.assertRaisesRegex(RuntimeError, 'defined Sharpe'):
            maximum_sharpe(np.zeros((4, 4)), mu, self.new, zero, .03)

    def test_cloud_respects_cash_cap_and_inactive_candidates(self):
        for cash in [0., .3531]:
            bounds = [(0, .2)]*6+[(0, 0), (cash, cash)]
            cloud = feasible_cloud(bounds, 3000, seed=17)
            self.assertGreater(len(cloud), 0)
            np.testing.assert_allclose(cloud.sum(axis=1), 1, atol=1e-12)
            np.testing.assert_array_equal(cloud[:, -1], cash)
            np.testing.assert_array_equal(cloud[:, -2], 0)
            self.assertTrue((cloud[:, :6] <= .2+1e-10).all())
            self.assertTrue((cloud >= 0).all())

    def test_exclusion_marks_infeasibility_and_keeps_numerical_failures(self):
        mu = np.array([.04, .10, .06, .03])
        bounds = [(0, .8), (0, .8), (0, 0), (.2, .2)]
        w = exclusion_weights(self.cov, mu, bounds, 0, .086)
        np.testing.assert_allclose(w, [0, .8, 0, .2], atol=1e-8)
        self.assertIsNone(exclusion_weights(self.cov, mu, bounds, 1, .086))
        failure = SimpleNamespace(status=4, success=False, message='numerical failure')
        with patch('portfolio_validation.linprog', return_value=failure):
            with self.assertRaisesRegex(RuntimeError, 'numerical failure'):
                exclusion_weights(self.cov, mu, bounds, 0, .086)
        with patch('portfolio_validation.minimum_risk', side_effect=RuntimeError('solver failed')):
            with self.assertRaisesRegex(RuntimeError, 'solver failed'):
                exclusion_weights(self.cov, mu, bounds, 0, .086)
        minimum = minimum_risk(self.cov, self.new)
        with patch('portfolio_validation.minimum_risk',
                   side_effect=[minimum, RuntimeError('frontier point failed')]):
            with self.assertRaisesRegex(RuntimeError, 'frontier point failed'):
                compute_frontier(self.cov, self.mu, self.new, 5)

    def test_future_cutoff_is_rejected_before_creating_outputs(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)/'future'
            result = subprocess.run(
                [sys.executable, str(Path(__file__).with_name('AR-Y_portfolio.py')),
                 '--no-show', '--as-of', '2999-01-01', '--output-dir', str(output)],
                capture_output=True, text=True, timeout=30)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('--as-of cannot be later than today', result.stderr)
            self.assertFalse(output.exists())

if __name__=='__main__': unittest.main()
