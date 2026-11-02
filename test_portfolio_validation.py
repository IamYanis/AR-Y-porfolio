import tempfile
import unittest
import numpy as np
import pandas as pd
from risk_models import shrink_covariance
from portfolio_validation import minimum_risk, hold_period, fit_covariances, run_validation

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

if __name__=='__main__': unittest.main()
