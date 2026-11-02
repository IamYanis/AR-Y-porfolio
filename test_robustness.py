import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from robustness import proposed_weights,scenario_grid,return_estimate,evaluate_change

class RobustnessTests(unittest.TestCase):
    def setUp(self):
        self.assets=['A','B','C','D','E','F','CASH_GBP']
        self.current=np.array([.2,.2,.2,.2,0,0,.2])
    def test_funding_and_existing_control(self):
        w=proposed_weights(self.current,self.assets,('E','F'),.1)
        self.assertAlmostEqual(w.sum(),1)
        self.assertEqual(w[-1],.2)
        np.testing.assert_allclose(w[:4],[.175]*4)
        np.testing.assert_allclose(w[4:6],[.05,.05])
        with self.assertRaises(ValueError):proposed_weights(self.current,self.assets,('A',),.05)
        control=proposed_weights(self.current,self.assets,('A',),.05,allow_held=True)
        self.assertAlmostEqual(control[0],.2375)
        self.assertAlmostEqual(control.sum(),1)
        self.assertAlmostEqual(abs(w-self.current).sum(),.2)
    def test_per_side_cost_on_funded_trade(self):
        test=pd.DataFrame(np.zeros((10,len(self.assets))),columns=self.assets)
        w=proposed_weights(self.current,self.assets,('E',),.05)
        result,_,_=evaluate_change(test,w,self.current,np.zeros((len(w),len(w))))
        self.assertAlmostEqual(result['Traded_Notional'],.10)
        self.assertAlmostEqual(result['Net_Return_25bps'],-.00025)
        self.assertEqual(result['Base_Net_Return_25bps'],0)
        self.assertEqual(result['Realised_Risk_Reduction_pp'],0)

    def test_fractional_suggested_weights_keep_cash_and_existing_proportions(self):
        additions = ['PDBC', 'JNJ', 'KO', 'CVX', 'AAPL']
        assets = ['A', 'B', 'C', 'D', 'G']+additions+['OTHER', 'CASH_GBP']
        current = np.array([.15, .14, .13, .12, .1069]+[0.]*6+[.3531])
        suggested = np.array([.083714, .072565, .059003, .048145, .037616])
        saved = suggested.copy()
        for fraction in [0., .25, .5, .75, 1.]:
            w = proposed_weights(current, assets, additions, fraction*suggested.sum(),
                                  addition_weights=suggested)
            np.testing.assert_allclose(w[5:10], fraction*suggested, atol=1e-14)
            np.testing.assert_allclose(w[:5]/w[:5].sum(), current[:5]/.6469, atol=1e-14)
            self.assertAlmostEqual(w[:5].sum(), .6469-fraction*suggested.sum())
            self.assertEqual(w[-1], .3531)
            self.assertEqual(w[-2], 0)
            self.assertAlmostEqual(w.sum(), 1)
            self.assertTrue((w[:-1] <= .2).all() and (w >= 0).all())
            if fraction == 0:
                np.testing.assert_array_equal(w, current)
        np.testing.assert_array_equal(suggested, saved)
        # New optimisation weights take effect without changing the function.
        updated = suggested*.8
        w = proposed_weights(current, assets, additions, updated.sum(), addition_weights=updated)
        np.testing.assert_allclose(w[5:10], updated, atol=1e-14)
        unchanged = proposed_weights(current, assets, additions, 0, addition_weights=np.zeros(5))
        np.testing.assert_array_equal(unchanged, current)

    def test_invalid_unequal_sleeve_weights_are_rejected(self):
        for weights in [[-1., 2.], [np.nan, 1.], [1.], [0., 0.]]:
            with self.assertRaises(ValueError):
                proposed_weights(self.current, self.assets, ('E', 'F'), .1,
                                  addition_weights=weights)

    def test_return_assumption_changes_asset_relative_means(self):
        train=pd.DataFrame({'A':[0,0,0,1],'B':[.01]*4,'CASH_GBP':[.0001]*4})
        a=return_estimate(train,'historical_mean');b=return_estimate(train,'winsorised_mean')
        self.assertLess(b[0],a[0]);self.assertEqual(a[1],b[1]);self.assertEqual(a[2],b[2])
    def test_training_only_selection_and_unique_quarters(self):
        rng=np.random.default_rng(42)
        r=pd.DataFrame(rng.normal(.0003,.01,(80,6)),index=pd.bdate_range('2025-01-01',periods=80),columns=self.assets[:-1])
        r['CASH_GBP']=.0001
        holdings=dict(zip(self.assets,self.current*100))
        with tempfile.TemporaryDirectory() as folder:
            a,sa=scenario_grid(r,holdings,['E','F'],Path(folder)/'a',lookbacks=(20,40),caps=(.3,),test_days=20)
            later=r.copy();later.iloc[40:,:6]*=-4
            b,sb=scenario_grid(later,holdings,['E','F'],Path(folder)/'b',lookbacks=(20,40),caps=(.3,),test_days=20)
            pd.testing.assert_frame_equal(a['change_selection'],b['change_selection'])
            pd.testing.assert_frame_equal(a['proposed_weights'],b['proposed_weights'])
            first=a['change_forecasts'].query('Fold == 1').reset_index(drop=True)
            changed=b['change_forecasts'].query('Fold == 1').reset_index(drop=True)
            pd.testing.assert_frame_equal(first,changed)
            self.assertTrue((a['change_summary'].Evaluation_Quarters==2).all())
            self.assertFalse(a['practical_changes'].duplicated(['Fold','Change','Total_Allocation']).any())
            self.assertTrue((a['practical_changes'].Net_Return_25bps <= a['practical_changes'].Net_Return_0bps+1e-12).all())
            self.assertTrue((pd.to_datetime(a['optimised_scenarios'].Train_End)<pd.to_datetime(a['optimised_scenarios'].Test_Start)).all())
            weights=a['proposed_weights']
            np.testing.assert_allclose(weights.groupby(['Change','Total_Allocation']).Weight.sum(),1)

if __name__=='__main__':unittest.main()
