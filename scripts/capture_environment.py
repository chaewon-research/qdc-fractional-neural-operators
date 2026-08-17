import json,platform,sys,subprocess,os
from pathlib import Path
import numpy,scipy,torch
info={'python':sys.version,'platform':platform.platform(),'numpy':numpy.__version__,'scipy':scipy.__version__,'torch':torch.__version__,'cuda_runtime':torch.version.cuda,'cuda_available':torch.cuda.is_available(),'float32_matmul_precision':torch.get_float32_matmul_precision(),'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),'cudnn_deterministic':bool(getattr(torch.backends.cudnn,'deterministic',False)),'cudnn_benchmark':bool(getattr(torch.backends.cudnn,'benchmark',False))}
if torch.cuda.is_available():
    info.update({'gpu':torch.cuda.get_device_name(0),'gpu_capability':torch.cuda.get_device_capability(0),'tf32_matmul':bool(torch.backends.cuda.matmul.allow_tf32),'tf32_cudnn':bool(getattr(torch.backends.cudnn,'allow_tf32',False))})
    try:
        q='driver_version,clocks.sm,power.limit'; out=subprocess.check_output(['nvidia-smi',f'--query-gpu={q}','--format=csv,noheader'],text=True,stderr=subprocess.DEVNULL).strip(); info['nvidia_smi']=out
    except Exception as e: info['nvidia_smi']=f'unavailable: {type(e).__name__}'
if os.getenv('CUTLASS_COMMIT'): info['cutlass_commit']=os.environ['CUTLASS_COMMIT']
# Best-effort library metadata; availability depends on the runtime image.
for key, cmd in {
    'nvcc_version':['nvcc','--version'],
}.items():
    try: info[key]=subprocess.check_output(cmd,text=True,stderr=subprocess.STDOUT).strip()
    except Exception: pass
try:
    import ctypes.util
    info['cufft_library']=ctypes.util.find_library('cufft')
    info['cusolver_library']=ctypes.util.find_library('cusolver')
except Exception: pass
Path('results/environment.json').parent.mkdir(parents=True,exist_ok=True); Path('results/environment.json').write_text(json.dumps(info,indent=2)); print(json.dumps(info,indent=2))
