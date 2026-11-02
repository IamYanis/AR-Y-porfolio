from datetime import datetime
from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from fund_nav import ISIN
from paper_test import register,load_protocol,record_prices,read_observations,simulate,LONDON
from paper_report import generate_paper_report

class ForwardTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.baseline=self.root/'baseline';self.baseline.mkdir()
        self.assets=['A','B','CBGA.L',ISIN,'C','D','CASH_GBP']
        rng=np.random.default_rng(17)
        r=pd.DataFrame(rng.normal(.0002,.01,(252,6)),index=pd.bdate_range(end='2026-10-29',periods=252),columns=self.assets[:-1])
        r['CASH_GBP']=.0001;r.to_csv(self.baseline/'gbp_daily_returns.csv',index_label='Date')
        holdings=dict(zip(self.assets,[15,15,15,15,0,0,40]))
        (self.baseline/'run_manifest.json').write_text(json.dumps({'bond_series':'actual','holdings_percent':holdings,'candidate_assets':['C','D']}))
        pd.DataFrame({'Asset':self.assets[:-1],'Quote_Currency':['USD','USD','GBP','EUR','USD','USD']}).to_csv(self.baseline/'data_quality.csv',index=False)
        self.trial=self.root/'trial';self.now=datetime(2026,10,30,16,tzinfo=LONDON)
        self.p=register(self.trial,self.baseline,now=self.now)
    def tearDown(self):self.temp.cleanup()
    def prices(self,periods=8):
        dates=pd.bdate_range('2026-10-29',periods=periods)
        values=np.array([100*(1+.001*(i+1))**np.arange(periods) for i in range(len(self.p['active_assets']))]).T
        return pd.DataFrame(values,index=dates,columns=self.p['active_assets'])
    def later(self):return datetime(2027,5,31,16,tzinfo=LONDON)
    def test_registration_and_immutable_rules(self):
        self.assertEqual(self.p['earliest_entry_date'],'2026-11-02')
        self.assertEqual(self.p['training_end'],'2026-10-29')
        self.assertEqual(set(self.p['weights']),{'current','existing_bond','single','pair'})
        for weights in self.p['weights'].values():
            self.assertAlmostEqual(sum(weights.values()),1)
            self.assertEqual(weights['CASH_GBP'],.4)
        with self.assertRaises(FileExistsError):register(self.trial,self.baseline,now=self.now)
        path=self.trial/'protocol.json';path.write_text(path.read_text()+' ')
        with self.assertRaises(ValueError):load_protocol(self.trial)
    def test_no_future_or_pre_entry_results(self):
        prices=self.prices(12)
        ledger,_=record_prices(self.trial,prices,'2026-10-30',now=self.now)
        self.assertTrue(ledger.empty)
        self.assertFalse((self.trial/'observations.csv').exists())
        ledger,_=record_prices(self.trial,prices,'2026-11-04',now=self.later())
        self.assertEqual(ledger.Date.min(),pd.Timestamp('2026-11-02'))
        self.assertEqual(ledger.Date.max(),pd.Timestamp('2026-11-03'))
        with self.assertRaises(ValueError):record_prices(self.trial,prices,'2028-01-01',now=self.later())
    def test_uniform_price_rebase_does_not_create_false_returns(self):
        prices=self.prices(10)
        first,_=record_prices(self.trial,prices,'2026-11-05',now=self.later())
        second,result=record_prices(self.trial,prices*.5,'2026-11-10',now=self.later())
        pd.testing.assert_frame_equal(first,second.iloc[:len(first)].reset_index(drop=True))
        self.assertEqual(result['revisions'],0)
        asset=self.p['active_assets'][0]
        self.assertAlmostEqual(second[asset].iloc[-1],.001)
    def test_source_revisions_preserve_recorded_history(self):
        prices=self.prices(10)
        first,_=record_prices(self.trial,prices,'2026-11-05',now=self.later())
        changed=prices.copy();changed.loc['2026-11-03',self.p['active_assets'][0]]*=1.01
        second,result=record_prices(self.trial,changed,'2026-11-10',now=self.later())
        pd.testing.assert_frame_equal(first,second.iloc[:len(first)].reset_index(drop=True))
        self.assertGreater(result['revisions'],0)
        self.assertTrue((self.trial/'source_revision_log.csv').exists())
        path=self.trial/'observations.csv';path.write_text(path.read_text()+'\n')
        with self.assertRaises(ValueError):read_observations(self.trial)
    def test_calendar_cash_and_quarterly_costs(self):
        ledger,_=record_prices(self.trial,self.prices(100),'2027-04-01',now=self.later())
        row=ledger.loc[ledger.Date==pd.Timestamp('2026-11-09')].iloc[0]
        self.assertAlmostEqual(row.CASH_GBP,1.03**(3/365.25)-1)
        paths,events=simulate(self.p,ledger)
        self.assertEqual(set(events.query('Event == "Quarterly rebalance"').Date),{'2027-02-02'})
        initial=events.query('Event == "Initial allocation"').set_index('Portfolio')
        self.assertEqual(initial.loc['current','Traded_Notional_Fraction'],0)
        self.assertAlmostEqual(initial.loc['single','Traded_Notional_Fraction'],.10)
        self.assertTrue((paths.Net_Value_25bps_GBP<=paths.Net_Value_0bps_GBP+1e-8).all())
        np.testing.assert_allclose(paths.Net_Value_0bps_GBP,paths.Gross_Value_GBP)
        report,status=generate_paper_report(self.trial,now=self.later())
        quarters=pd.read_csv(report.parent/'quarters.csv')
        self.assertEqual(len(quarters),4)
        self.assertTrue(quarters.Realised_Volatility_pct.notna().all())
        self.assertEqual(status['completed_quarters'],1)
    def test_reports_remain_pending_until_observations_exist(self):
        report,status=generate_paper_report(self.trial,now=self.now)
        self.assertEqual(status['status'],'Awaiting entry valuation')
        summary=pd.read_csv(report.parent/'summary.csv')
        self.assertTrue(summary.Gross_Return_pct.isna().all())
        self.assertTrue(summary.Realised_Volatility_pct.isna().all())
        self.assertIn('No forward entry valuation',(report.parent/'SUMMARY_REPORT.md').read_text())
        record_prices(self.trial,self.prices(8),'2026-11-09',now=self.later())
        report,status=generate_paper_report(self.trial,now=self.later())
        summary=pd.read_csv(report.parent/'summary.csv')
        self.assertTrue(summary.Realised_Volatility_pct.isna().all())
        self.assertTrue(summary.Gross_Return_pct.notna().all())
    def test_refresh_uses_actual_nav_and_correct_fx(self):
        from paper_data import refresh_prices,cached_prices
        dates=pd.to_datetime(['2026-10-28','2026-10-29'])
        def close(ticker,*args,**kwargs):
            value=.8 if ticker=='USDGBP=X' else (.9 if ticker=='EURGBP=X' else 100.)
            return pd.Series(value,index=dates,name=ticker)
        with patch('paper_data.load_close',side_effect=close),patch('paper_data.load_fund_nav',return_value=pd.Series(100.,index=dates,name=ISIN)) as nav:
            prices,source=refresh_prices(self.trial,'2026-10-30')
            nav.assert_called_once()
        self.assertAlmostEqual(prices['A'].iloc[0],80.)
        self.assertAlmostEqual(prices[ISIN].iloc[0],90.)
        self.assertAlmostEqual(prices['CBGA.L'].iloc[0],100.)
        cached,_=cached_prices(self.trial)
        np.testing.assert_allclose(cached.to_numpy(),prices.to_numpy())
        path=self.trial/source/'gbp_prices.csv';path.write_text(path.read_text()+'\n')
        with self.assertRaises(ValueError):cached_prices(self.trial)

    def test_import_cache_and_update_idempotence(self):
        from paper_data import import_prices,cached_prices
        source=self.root/'prices.csv';self.prices(10).to_csv(source,index_label='Date')
        prices,_=import_prices(self.trial,source)
        cached,_=cached_prices(self.trial)
        pd.testing.assert_frame_equal(prices,cached)
        first,_=record_prices(self.trial,prices,'2026-11-10',now=self.later())
        second,result=record_prices(self.trial,cached,'2026-11-10',now=self.later())
        self.assertEqual(result['appended'],0)
        pd.testing.assert_frame_equal(first,second)

    def test_stale_initial_baseline_is_rejected(self):
        with self.assertRaises(ValueError):
            register(self.root/'stale',self.baseline,now=datetime(2026,11,20,16,tzinfo=LONDON))
        self.assertFalse((self.root/'stale').exists())

    def test_future_training_rows_do_not_change_selection(self):
        path=self.baseline/'gbp_daily_returns.csv';r=pd.read_csv(path,index_col='Date',parse_dates=True)
        r.loc[pd.Timestamp('2027-01-01')]=1000;r.to_csv(path,index_label='Date')
        other=register(self.root/'other',self.baseline,now=self.now)
        self.assertEqual(self.p['weights'],other['weights'])
        self.assertEqual(self.p['training_sha256'],other['training_sha256'])

if __name__=='__main__':unittest.main()
