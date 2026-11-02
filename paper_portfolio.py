"""Register and monitor a forward paper test, with immutable starting rules."""
import argparse
from datetime import datetime
from pathlib import Path
from paper_test import register,load_protocol,record_prices,LONDON
from paper_data import refresh_prices,import_prices,cached_prices
from paper_report import generate_paper_report


def main():
    root=Path(__file__).resolve().parent
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['init','update','report'])
    parser.add_argument('--trial-dir',type=Path)
    parser.add_argument('--baseline-dir',type=Path,default=root/'reports'/'research-latest'/'baseline'/'actual')
    parser.add_argument('--refresh-data',action='store_true',help='Explicitly download new raw asset/FX data and actual fund NAVs')
    parser.add_argument('--prices-file',type=Path,help='CSV of GBP adjusted price indices / actual fund NAV, with Date column')
    parser.add_argument('--as-of',default=datetime.now(LONDON).date().isoformat(),help='Exclusive update cutoff')
    args=parser.parse_args()
    if args.refresh_data and args.prices_file:parser.error('Choose refresh-data or a prices-file')
    if args.command!='update' and (args.refresh_data or args.prices_file):parser.error('Data inputs are used only by update')
    if args.trial_dir:directory=args.trial_dir.resolve()
    elif args.command=='init':directory=root/'paper_tests'/('forward-'+datetime.now(LONDON).date().isoformat())
    else:
        trials=sorted((root/'paper_tests').glob('*/protocol.json'),key=lambda p:p.stat().st_mtime)
        if not trials:parser.error('No registered trial; run init first')
        directory=trials[-1].parent
    if args.command=='init':
        register(directory,args.baseline_dir)
    else:load_protocol(directory)
    if args.command=='update':
        if args.refresh_data:prices,source=refresh_prices(directory,args.as_of)
        elif args.prices_file:prices,source=import_prices(directory,args.prices_file)
        else:prices,source=cached_prices(directory)
        if prices is not None:
            _,result=record_prices(directory,prices,args.as_of,source)
            print(f"Recorded {result['appended']} new valuation(s); logged {result['revisions']} source revision(s)")
        else:print('No price snapshot available; use update --refresh-data when new observations are available')
    report,status=generate_paper_report(directory)
    print('Status: '+status['status'])
    print('Report: '+str(report))

if __name__=='__main__':main()
