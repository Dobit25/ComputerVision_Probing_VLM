"""Run the fixed extension queue after a successful gen/CUDA preflight."""
import argparse
import subprocess
import sys
from datetime import datetime, timezone

from .depth.common import ROOT, read_json
from .io_utils import write_json

STATUS = ROOT / 'results/integration_v2/execution_status.json'


def jobs():
    result = []
    for dataset in ['flowers102', 'oxford_pet', 'dtd', 'food101']:
        for smoke in [True, False]:
            name = f'{"smoke" if smoke else "full"}_prompt_{dataset}_review_v2'
            args = ['-m','vlm_probe.prompt_study','dataset','--dataset',dataset,'--run-name',name]
            if smoke: args.append('--smoke')
            result.append((name, args))
    for smoke in [True, False]:
        name = f'{"smoke" if smoke else "full"}_prompt_probing_review_v2'
        args = ['-m','vlm_probe.prompt_study','probing','--run-name',name]
        if smoke: args.append('--smoke')
        result.append((name, args))
    for set_name in ['exploratory','confirmation']:
        for smoke in [True, False]:
            name = f'{"smoke" if smoke else "full"}_natural_regions_{set_name}_review_v2'
            args = ['-m','vlm_probe.spatial_diagnostics.natural_regions','--set',set_name,'--run-name',name]
            if smoke: args.append('--smoke')
            result.append((name, args))
    return result


def main():
    p = argparse.ArgumentParser(); p.add_argument('--execute', action='store_true'); a = p.parse_args()
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    queue = [{'run_name': name, 'command': [sys.executable, *args], 'status': 'pending'} for name, args in jobs()]
    state = {'updated_at_utc': datetime.now(timezone.utc).isoformat(), 'jobs': queue}
    if not a.execute:
        state['status'] = 'prepared_not_executed'; write_json(STATUS, state); print(STATUS); return
    probe = subprocess.run([sys.executable,'-c',
        'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; print(torch.__version__); print(torch.cuda.get_device_name(0))'],
        capture_output=True, text=True, encoding='utf-8', errors='replace', cwd=ROOT)
    (STATUS.parent/'preflight.txt').write_text(probe.stdout + probe.stderr, encoding='utf-8')
    if probe.returncode:
        state.update(status='blocked_environment', error=probe.stderr.strip(),
                     explanation='No inference job started; Windows policy/environment requires user or administrator resolution.')
        write_json(STATUS,state); print(state['status'], state['error'], flush=True); return
    state['status']='running'; write_json(STATUS,state)
    for row in queue:
        out = ROOT / 'results' / row['run_name']
        if out.exists():
            if not (out/'manifest.json').exists() or read_json(out/'manifest.json')['status'] != 'completed':
                row['status']='existing_incomplete_run_requires_new_name'; state['status']='stopped'; break
        else:
            with (STATUS.parent/(row['run_name']+'.log')).open('w',encoding='utf-8') as log:
                task = subprocess.run(row['command'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            if task.returncode:
                row['status']='failed'; state['status']='stopped'; break
        check = subprocess.run([sys.executable,'-m','vlm_probe.extension_artifacts',row['run_name']],cwd=ROOT)
        row['status']='audited' if check.returncode==0 else 'audit_failed'
        if check.returncode: state['status']='stopped'; break
        write_json(STATUS,state)
    else: state['status']='completed_and_audited'
    write_json(STATUS,state); print(state['status'],flush=True)


if __name__ == '__main__': main()
