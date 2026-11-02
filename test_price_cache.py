import tempfile
import unittest
from pathlib import Path
import pandas as pd
from price_cache import load_close

class CacheTests(unittest.TestCase):
    def test_reuse_refresh_and_failure_preserves_data(self):
        with tempfile.TemporaryDirectory() as folder:
            calls=[]
            def download(*args, **kwargs):
                calls.append(args)
                return pd.DataFrame({'Close':[100.+len(calls),102.]},
                                    index=pd.to_datetime(['2025-01-02','2025-01-03']))
            def load(refresh=False):
                return load_close('USDGBP=X','2025-01-01','2025-02-01',folder,download,refresh)
            first=load()
            pd.testing.assert_series_equal(first,load(), check_freq=False)
            self.assertEqual(len(calls),1)
            updated=load(True)
            self.assertEqual(updated.iloc[0],102.)
            def fail(*args,**kwargs):
                return pd.DataFrame()
            with self.assertRaises(ValueError):
                load_close('USDGBP=X','2025-01-01','2025-02-01',folder,fail,True)
            pd.testing.assert_series_equal(updated,load(),check_freq=False)
            self.assertEqual(len(list(Path(folder).glob('*.json'))),1)
            load_close('USDGBP=X','2025-01-01','2025-03-01',folder,download)
            self.assertEqual(len(calls),3)

if __name__=='__main__':
    unittest.main()
