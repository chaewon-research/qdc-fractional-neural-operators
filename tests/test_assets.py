import json
from pathlib import Path

def test_realpdebench_asset_lock_complete():
    p=Path("external/realpdebench_assets.lock.json")
    d=json.loads(p.read_text())
    required={"source_git_commit","dataset_repository","dataset_revision","model_repository","model_revision","base_finetune_checkpoint","licenses"}
    assert required <= set(d)
    for k in required:
        assert str(d[k]).strip()
    assert len(d["source_git_commit"]) == 40
    assert len(d["dataset_revision"]) == 40
    assert len(d["model_revision"]) == 40


def test_realpdebench_distinct_asset_licenses():
    d=json.loads(Path("external/realpdebench_assets.lock.json").read_text())
    assert d["licenses"]["source_repository"] == "CC BY-NC 4.0"
    assert d["licenses"]["dataset_repository"] == "CC BY-NC 4.0"
    assert d["licenses"]["model_repository"] == "CC BY 4.0"


def test_fresh_qdc_config_matches_frozen_selection_record():
    import yaml
    cfg=yaml.safe_load(Path("configs/fresh_scalar_qdc.yaml").read_text())
    rec=json.loads(Path("results/reference/qdc_selection_record.json").read_text())
    assert cfg["qdc"]["candidate_k"] == rec["candidate_ranks"]
    assert cfg["qdc"]["candidate_cost_fractions"] == rec["candidate_cost_fractions"]



def test_core_seed_semantics_explicit():
    import yaml
    cfg=yaml.safe_load(Path('configs/core_scalar.yaml').read_text())
    sem=cfg['seed_semantics']
    assert sem['data_seed_offsets'] == {'train':0,'validation':10000,'holdout':30000}
    assert 'same holdout within each run' in sem['quantization_matching']


def test_smoke_result_is_execution_only():
    d=json.loads(Path('results/smoke_execution_only.json').read_text())
    assert d['scientific_result'] is False
    assert d['not_comparable_to_paper'] is True
    assert d['no_nan_inf'] is True
    assert 'fp32_rel_l2' not in d


def test_highres_aggregator_preserves_absolute_summary():
    src=Path('experiments/aggregate_highres.py').read_text()
    assert 'absolute_summary' in src
    assert "['fp32','q6_weight','q6_activation']" in src


def test_realpdebench_output_is_portable_and_hashed():
    src=Path('experiments/realpdebench_cylinder.py').read_text()
    assert 'checkpoint_sha256' in src
    assert 'Path(a.checkpoint).resolve()' not in src


def test_additional_verified_audits_present():
    d=json.loads(Path('results/reference/additional_verified_audits.json').read_text())
    assert d['batch_size_audit']['ordering_activation_worse_all'] is True
    assert d['lobpcg_sweep']['first_95pct_crossing'] == {'32':82,'64':150,'128':200}
    assert d['target_solver_validation']['paper_robust_target_error_bound'] == 1e-11


def test_lobpcg_selection_rule_uses_validation_reference_recovery():
    from qdc_fno.lobpcg import select_smallest_rank_by_reference_recovery
    rows=[
        {'k':41,'mean_reference_rel_l2_recovery':0.70},
        {'k':82,'mean_reference_rel_l2_recovery':0.96},
        {'k':150,'mean_reference_rel_l2_recovery':1.00},
    ]
    got=select_smallest_rank_by_reference_recovery(rows,0.95)
    assert got['selected_k'] == 82
    assert abs(got['threshold']-0.95) < 1e-12


def test_lobpcg_reproduction_uses_validation_then_holdout():
    train=Path('experiments/train_higher_resolution.py').read_text()
    sweep=Path('scripts/reproduce_lobpcg_sweep.sh').read_text()
    assert '.validation_qdc.pt' in train
    assert "'split': split_name" in train
    assert '.validation_qdc.pt' in sweep
    assert 'select_lobpcg_rank.py' in sweep
    assert '.qdc.pt' in sweep
    assert 'holdout' in sweep.lower()


def test_a100_historical_json_separates_measurement_from_derived_arithmetic():
    d=json.loads(Path('results/reference/a100_historical_measurement.json').read_text())
    assert 'retained_measurements' in d and 'derived_arithmetic' in d
    assert 'median_batch_latency_ms' not in json.dumps(d['retained_measurements'])
    assert 'implied_batch_latency_ms' in d['derived_arithmetic']
    assert d['retained_measurements']['reported_setup_inclusive_examples_per_s_100_batches_rounded'] == 8290
