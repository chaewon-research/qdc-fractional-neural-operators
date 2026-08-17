"""Reference A100 timing harness for the paper protocol.

The spectral INT8 kernel itself is backend-specific. This harness enforces batch/warm-up/timed
batch protocol with CUDA events and records the median. Integrate a CUTLASS callable through
`step_fn`; the included fallback measures a supplied PyTorch callable and is useful for pipeline
validation, not as a substitute for CUTLASS throughput claims.
"""
from __future__ import annotations
import argparse, json, statistics, torch
from pathlib import Path

def time_cuda(step_fn,warmup=50,measure=100):
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required')
    for _ in range(warmup): step_fn()
    torch.cuda.synchronize(); times=[]
    for _ in range(measure):
        s,e=torch.cuda.Event(True),torch.cuda.Event(True); s.record(); step_fn(); e.record(); e.synchronize(); times.append(s.elapsed_time(e))
    return {'median_ms':statistics.median(times),'all_ms':times}

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--out',default='results/a100_timing.json'); p.add_argument('--allow-other-gpu',action='store_true'); a=p.parse_args()
    if not torch.cuda.is_available(): raise SystemExit('Run on CUDA/A100.')
    gpu=torch.cuda.get_device_name(0)
    if 'A100' not in gpu and not a.allow_other_gpu: raise SystemExit(f'Paper protocol targets A100; detected {gpu}. Pass --allow-other-gpu only for harness validation.')
    # Genuine FP32 comparison: disable Ampere TF32 shortcuts.
    torch.backends.cuda.matmul.allow_tf32=False
    if hasattr(torch.backends,'cudnn'): torch.backends.cudnn.allow_tf32=False
    x=torch.randn(16,64,32,32,device='cuda',dtype=torch.float16)
    def reference_step():
        y=torch.fft.rfft2(x); return torch.fft.irfft2(y,s=(32,32))
    res={'gpu':gpu,'tf32_enabled':False,'torch_version':torch.__version__,'cuda_version':torch.version.cuda,'protocol':{'batch_size':16,'warmup':50,'measure':100},'reference_fft_path':time_cuda(reference_step)}
    Path(a.out).parent.mkdir(parents=True,exist_ok=True); Path(a.out).write_text(json.dumps(res,indent=2)); print(json.dumps({k:v for k,v in res.items() if k!='reference_fft_path'},indent=2))
