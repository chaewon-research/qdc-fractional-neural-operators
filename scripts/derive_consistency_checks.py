"""Derive arithmetic/statistical consistency checks from the authoritative aggregate results.

These values are consequences of the reported aggregates; they are not reconstructed raw observations.
"""
import json, math
from pathlib import Path
T4=2.7764451051977987
N=5
ref=json.loads(Path('results/reference/reported_results.json').read_text())
def implied_sd(triple):
    mean,lo,hi=map(float,triple); half=(hi-lo)/2.0; return half*math.sqrt(N)/T4
out={
 'student_t_df':4,
 'higher_resolution':{
  '64_rel_gap_implied_sample_sd':implied_sd(ref['higher_resolution']['64']['rel_gap']),
  '64_residual_gap_implied_sample_sd':implied_sd(ref['higher_resolution']['64']['residual_gap']),
  '128_rel_gap_implied_sample_sd':implied_sd(ref['higher_resolution']['128']['rel_gap']),
  '128_residual_gap_implied_sample_sd':implied_sd(ref['higher_resolution']['128']['residual_gap']),
 },
 'realpdebench':{
  'rel_gap_implied_sample_sd':implied_sd(ref['realpdebench_cylinder']['rel_gap']),
  'frmse_gap_implied_sample_sd':implied_sd(ref['realpdebench_cylinder']['frmse_gap']),
 },
 'a100':{
  'batch_size':16,
  'fp32_implied_batch_latency_ms':1000*16/ref['a100']['fp32']['examples_per_s'],
  'w8a8_implied_batch_latency_ms':1000*16/ref['a100']['w8a8']['examples_per_s'],
  'qdc_online_implied_batch_latency_ms':1000*16/ref['a100']['w8a8_qdc_rank41']['examples_per_s'],
  'fixed_operator_setup_ms':ref['a100']['w8a8_qdc_rank41_setup_inclusive']['setup_ms'],
  'amortization_batches':100,
  'amortization_examples':1600,
 }
}
online_ms=1000*out['a100']['amortization_examples']/ref['a100']['w8a8_qdc_rank41']['examples_per_s']
out['a100']['online_total_ms_for_1600']=online_ms
out['a100']['setup_inclusive_total_ms_for_1600']=online_ms+out['a100']['fixed_operator_setup_ms']
out['a100']['setup_inclusive_implied_examples_per_s']=1000*out['a100']['amortization_examples']/out['a100']['setup_inclusive_total_ms_for_1600']
saved_per_batch=out['a100']['fp32_implied_batch_latency_ms']-out['a100']['qdc_online_implied_batch_latency_ms']
out['a100']['setup_break_even_batches_vs_fp32']=out['a100']['fixed_operator_setup_ms']/saved_per_batch
out['a100']['setup_break_even_examples_vs_fp32']=16*out['a100']['setup_break_even_batches_vs_fp32']
def setup_inc_eps(batches):
    total=out['a100']['fixed_operator_setup_ms']+batches*out['a100']['qdc_online_implied_batch_latency_ms']
    return 1000*(16*batches)/total
out['a100']['setup_inclusive_examples_per_s_40_batches']=setup_inc_eps(40)
out['a100']['setup_inclusive_examples_per_s_50_batches']=setup_inc_eps(50)
Path('results/reference/derived_consistency_checks.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
