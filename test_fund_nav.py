import tempfile
import unittest
from pathlib import Path
import pandas as pd
from fund_nav import parse_nav, ISIN, load_fund_nav

PAYLOAD = ('''<?xml version="1.0"?><ss:Workbook xmlns:ss="urn:schemas-microsoft-com:office:spreadsheet">
<ss:Worksheet ss:Name="Overview"><ss:Table>
<ss:Row><ss:Cell><ss:Data>ISIN</ss:Data></ss:Cell><ss:Cell><ss:Data>ISIN_VALUE</ss:Data></ss:Cell></ss:Row>
<ss:Row><ss:Cell><ss:Data>Share Class Currency</ss:Data></ss:Cell><ss:Cell><ss:Data>EUR</ss:Data></ss:Cell></ss:Row>
</ss:Table></ss:Worksheet><ss:Worksheet ss:Name="Historical"><ss:Table>
<ss:Row><ss:Cell><ss:Data>As Of</ss:Data></ss:Cell><ss:Cell><ss:Data>NAV per Share</ss:Data></ss:Cell></ss:Row>
<ss:Row><ss:Cell><ss:Data>30/Sept/2025</ss:Data></ss:Cell><ss:Cell><ss:Data>8.5</ss:Data></ss:Cell></ss:Row>
<ss:Row><ss:Cell><ss:Data>01/Mar/2022</ss:Data></ss:Cell><ss:Cell><ss:Data>10</ss:Data></ss:Cell></ss:Row>
</ss:Table></ss:Worksheet></ss:Workbook>''').replace('ISIN_VALUE',ISIN).encode()

class NavTests(unittest.TestCase):
    def test_exact_share_class_currency_and_unfilled_gap(self):
        s=parse_nav(PAYLOAD)
        self.assertEqual(s.name,ISIN)
        self.assertEqual(len(s),2)
        self.assertTrue(s.index.is_monotonic_increasing)
        with self.assertRaises(ValueError): parse_nav(PAYLOAD.replace(ISIN.encode(),b'WRONG_ISIN'))
        with self.assertRaises(ValueError): parse_nav(PAYLOAD.replace(b'>EUR<',b'>USD<'))
        with tempfile.TemporaryDirectory() as folder:
            Path(folder,ISIN+'_blackrock.xml').write_bytes(PAYLOAD)
            window=load_fund_nav('2025-01-01','2026-01-01',folder)
            self.assertEqual(list(window.index),[pd.Timestamp('2025-09-30')])

if __name__=='__main__':unittest.main()
