import json, subprocess, sys
from pathlib import Path
import torch, yaml


def test_seed_config_generator_uses_pinned_upstream_layout(tmp_path):
    upstream=tmp_path/'upstream'; cfgdir=upstream/'realpdebench/configs/cylinder'; cfgdir.mkdir(parents=True)
    (cfgdir/'fno.yaml').write_text(yaml.safe_dump({
        'dataset_name':'cylinder','results_path':'old','checkpoint_path':'old.ckpt','seed':99,'is_use_tb':True
    }))
    base=tmp_path/'numerical.pth'; base.write_bytes(b'x')
    lock=tmp_path/'lock.json'; lock.write_text(json.dumps({'base_finetune_checkpoint':'cylinder/fno/numerical.pth'}))
    out=tmp_path/'cfgs'; data=tmp_path/'data'; data.mkdir()
    subprocess.run([
        sys.executable,'experiments/prepare_realpdebench_seed_configs.py',
        '--upstream-root',str(upstream),'--dataset-root',str(data),'--asset-lock',str(lock),
        '--base-checkpoint',str(base),'--out-dir',str(out)
    ],check=True,capture_output=True,text=True)
    files=sorted(out.glob('fno_cylinder_seed*.yaml')); assert len(files)==5
    c0=yaml.safe_load(files[0].read_text()); c4=yaml.safe_load(files[4].read_text())
    assert c0['seed']==0 and c4['seed']==4
    assert Path(c0['checkpoint_path'])==base.resolve()
    assert Path(c0['dataset_root'])==data.resolve()
    assert 'seed_0' in c0['results_path'] and c0['is_use_tb'] is False


def test_best_checkpoint_selector_uses_recorded_best_iteration(tmp_path):
    run=tmp_path/'seed_0'/'nested_run'; run.mkdir(parents=True)
    for it in (20,40,60):
        torch.save({'iteration':it,'best_iteration':40,'best_val_loss':0.123},run/f'model_{it:04d}.pth')
    out=tmp_path/'best.pth'
    subprocess.run([sys.executable,'scripts/select_realpdebench_best_checkpoint.py','--seed-root',str(tmp_path/'seed_0'),'--out',str(out)],check=True,capture_output=True,text=True)
    assert out.exists(); d=torch.load(out,map_location='cpu',weights_only=False); assert d['iteration']==40
    meta=json.loads(out.with_suffix(out.suffix+'.json').read_text()); assert meta['best_iteration']==40


def test_best_checkpoint_selector_metadata_is_portable_and_hashed(tmp_path):
    run=tmp_path/'seed_0'/'nested_run'; run.mkdir(parents=True)
    for it in (20,40):
        torch.save({'iteration':it,'best_iteration':40,'best_val_loss':0.123},run/f'model_{it:04d}.pth')
    out=tmp_path/'best.pth'
    subprocess.run([sys.executable,'scripts/select_realpdebench_best_checkpoint.py','--seed-root',str(tmp_path/'seed_0'),'--out',str(out)],check=True,capture_output=True,text=True)
    meta=json.loads(out.with_suffix(out.suffix+'.json').read_text())
    dumped=json.dumps(meta)
    assert str(tmp_path.resolve()) not in dumped
    assert meta['source_checkpoint_basename'] == 'model_0040.pth'
    assert len(meta['source_checkpoint_sha256']) == 64
    assert meta['source_checkpoint_sha256'] == meta['copied_checkpoint_sha256']
