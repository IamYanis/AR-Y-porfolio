import base64
from pathlib import Path
import tempfile
import unittest
import pandas as pd
from research_report import Report

class ReportTests(unittest.TestCase):
    def test_html_escapes_data_and_embeds_chart(self):
        with tempfile.TemporaryDirectory() as folder:
            folder=Path(folder)
            chart=folder/'chart.png'
            chart.write_bytes(base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+a+j8AAAAASUVORK5CYII='))
            report=Report()
            report.heading('Results <checked>',1)
            report.paragraph('Risk < 2% & no assumptions hidden')
            report.table(pd.DataFrame({'Asset':['<script>alert(1)</script>'],'Risk':[.123]}))
            report.image(chart,'Risk comparison')
            report.save(folder)
            html=(folder/'SUMMARY_REPORT.html').read_text()
            self.assertIn('&lt;script&gt;',html)
            self.assertNotIn('<script>',html)
            self.assertIn('src="data:image/png;base64,',html)
            self.assertNotIn('src="http',html)
            self.assertIn('.123',(folder/'SUMMARY_REPORT.md').read_text())

if __name__=='__main__':unittest.main()
