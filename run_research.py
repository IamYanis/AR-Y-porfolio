"""Run reproducible robustness checks and create Markdown and standalone HTML reports."""
import argparse
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib
import json
from pathlib import Path
import sys
import os
import subprocess
import pandas as pd
from robustness import scenario_grid
from research_report import generate_report


def main():
    root=Path(__file__).resolve().parent
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-dir',type=Path,default=None,
                        help='Existing baseline directory containing actual/ and proxy/ runs')
    parser.add_argument('--output-dir',type=Path,default=root/'reports'/'research-latest')
    parser.add_argument('--report-only',action='store_true',help='Regenerate reports from previously calculated sensitivity CSVs')
    parser.add_argument('--rebuild-baseline',action='store_true',
                        help='Recalculate both baseline runs from cached prices before robustness checks')
    parser.add_argument('--as-of',help='Exclusive date cutoff, used with --rebuild-baseline')
    args=parser.parse_args()
    if args.report_only and args.rebuild_baseline: parser.error('--report-only cannot rebuild calculations')
    if args.as_of and not args.rebuild_baseline: parser.error('--as-of requires --rebuild-baseline')
    output=args.output_dir.resolve()
    baseline=(args.baseline_dir or (output/'baseline' if args.rebuild_baseline or (output/'baseline').is_dir() else root/'reports'/'baseline-2026-10-30')).resolve()
    output.mkdir(parents=True,exist_ok=True)
    if args.rebuild_baseline:
        environment=dict(os.environ,MPLBACKEND='Agg')
        for series in ['actual','proxy']:
            command=[sys.executable,str(root/'AR-Y_portfolio.py'),'--no-show','--validate',
                     '--bond-series',series,'--output-dir',str(baseline/series)]
            if args.as_of: command+=['--as-of',args.as_of]
            subprocess.run(command,check=True,cwd=root,env=environment)
    inputs=[]
    for dataset in ['actual','proxy']:
        folder=baseline/dataset
        manifest=json.loads((folder/'run_manifest.json').read_text())
        if manifest['bond_series'] != dataset: raise ValueError('Baseline actual/proxy labels do not match manifests')
        for name in ['gbp_daily_returns.csv','run_manifest.json','matched_current_return.csv']:
            p=folder/name
            inputs.append({'dataset':dataset,'file':name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
        returns=pd.read_csv(folder/'gbp_daily_returns.csv',index_col='Date',parse_dates=True)
        if set(returns.columns)!=set(manifest['holdings_percent']):
            raise ValueError('Returns and recorded holdings do not match')
        if str(returns.index.max().date())!=manifest['return_end'] or len(returns)!=manifest['observations']:
            raise ValueError('Return data does not match baseline manifest')
        if returns.index.max()>=pd.Timestamp(manifest['as_of_exclusive']):
            raise ValueError('Return inputs exceed as-of cutoff')
        if not args.report_only:
            print(f'Running {dataset}: {len(returns)} observations',flush=True)
            scenario_grid(returns,manifest['holdings_percent'],manifest['candidate_assets'],output/dataset)
    result_hashes={str(p.relative_to(output)):hashlib.sha256(p.read_bytes()).hexdigest()
                   for label in ['actual','proxy'] for p in sorted((output/label).glob('*')) if p.suffix in ['.csv','.json']}
    provenance=output/'research_manifest.json' 
    if args.report_only:
        previous=json.loads(provenance.read_text())
        if previous['inputs']!=inputs or previous.get('output_sha256')!=result_hashes:
            raise ValueError('Baseline or sensitivity inputs changed; rerun calculations without --report-only')
    else:
        record={'generated_at_london':datetime.now(ZoneInfo('Europe/London')).isoformat(),
                'baseline_dir':os.path.relpath(baseline,output),'inputs':inputs,'output_sha256':result_hashes,'python':sys.version,
                'code_sha256':{p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in
                               ['run_research.py','robustness.py','research_report.py','portfolio_validation.py','risk_models.py']}}
        provenance.write_text(json.dumps(record,indent=2)+'\n')
    generate_report(output,baseline,root/'source_review')
    (output/'report_manifest.json').write_text(json.dumps({
        'generated_at_london':datetime.now(ZoneInfo('Europe/London')).isoformat(),
        'report_code_sha256':hashlib.sha256((root/'research_report.py').read_bytes()).hexdigest(),
        'research_manifest_sha256':hashlib.sha256(provenance.read_bytes()).hexdigest()},indent=2)+'\n')
    print(f"Report: {output/'SUMMARY_REPORT.html'}",flush=True)

if __name__=='__main__':main()
