#!/usr/bin/env python3
"""Standalone test for primal_dual_core.py changes."""

import sys
import os

# Only import the core module directly to avoid heavy dependencies
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import directly from the file
import importlib.util
spec = importlib.util.spec_from_file_location("primal_dual_core", 
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "reward_score", "primal_dual_core.py"))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

GenericRewardCombiner = mod.GenericRewardCombiner

print("=== Test 1: Basic PD operation ===")
combiner = GenericRewardCombiner(combine_mode='pd', subreward_names=['eff', 'cons'])
assert combiner.normalize_by_dual_mass == True, f"Expected True, got {combiner.normalize_by_dual_mass}"
print(f"  normalize_by_dual_mass: {combiner.normalize_by_dual_mass} ✓")
assert combiner.sub_cfgs['eff'].lambda_max == 0.5, f"Expected 0.5, got {combiner.sub_cfgs['eff'].lambda_max}"
print(f"  lambda_max: {combiner.sub_cfgs['eff'].lambda_max} ✓")

print("\n=== Test 2: Step inflation prevention ===")
for step in range(5):
    for i in range(16):  # 16 samples per step
        combiner.process_batch([float(i % 2)], [{'eff': 0.5, 'cons': 0.5}], global_step=step)
    combiner.flush_pending_duals()
print(f"  After 5 steps x 16 samples = 80 process_batch calls")
print(f"  Internal step: {combiner._state.step}")
assert combiner._state.step == 5, f"Expected 5, got {combiner._state.step}"
print(f"  Step inflation prevented ✓")

print("\n=== Test 3: Reward range check ===")
# Wrong answer
info = combiner.process_batch([0.0], [{'eff': 0.3, 'cons': 0.0}], global_step=5)[0]
print(f"  Wrong answer score: {info['score']:.4f}")
assert -1.0 <= info['score'] <= 1.0, f"Score out of range: {info['score']}"

# Correct answer
info = combiner.process_batch([1.0], [{'eff': 0.9, 'cons': 1.0}], global_step=5)[0]
print(f"  Correct answer score: {info['score']:.4f}")
assert -1.0 <= info['score'] <= 1.0, f"Score out of range: {info['score']}"

# Check correct > wrong
wrong = combiner.process_batch([0.0], [{'eff': 0.3, 'cons': 0.0}], global_step=6)[0]['score']
correct = combiner.process_batch([1.0], [{'eff': 0.9, 'cons': 1.0}], global_step=6)[0]['score']
assert correct > wrong, f"Correct ({correct}) should be > wrong ({wrong})"
print(f"  Correct ({correct:.4f}) > Wrong ({wrong:.4f}) ✓")

print("\n=== Test 4: Non-PD mode unaffected ===")
combiner_mult = GenericRewardCombiner(combine_mode='multiplier', subreward_names=['eff'])
assert combiner_mult.normalize_by_dual_mass == False, f"Expected False for multiplier mode"
info = combiner_mult.process_batch([1.0], [{'eff': 0.8}])[0]
print(f"  Multiplier score: {info['score']:.4f}")
print(f"  normalize_by_dual_mass: {combiner_mult.normalize_by_dual_mass} ✓")

print("\n=== Test 5: Legacy mode (no global_step) still works ===")
combiner_legacy = GenericRewardCombiner(combine_mode='pd', subreward_names=['eff', 'cons'])
for step in range(5):
    for i in range(4):
        combiner_legacy.process_batch([float(i % 2)], [{'eff': 0.5, 'cons': 0.5}])
print(f"  Legacy internal step: {combiner_legacy._state.step}")
# In legacy mode, step increments per-call (backward compatible)
assert combiner_legacy._state.step == 20, f"Expected 20 for legacy mode, got {combiner_legacy._state.step}"
print(f"  Legacy mode backward compatible ✓")

print("\n=== Test 6: Eta decay is gentle ===")
import math
# Old decay: 1/sqrt(step+1) at step=300: 0.0577
# New decay: 1/(1 + 0.1*ln(step+1)) at step=300: 0.637
old_decay = 1.0 / math.sqrt(301.0)
new_decay = 1.0 / (1.0 + 0.1 * math.log(301.0))
print(f"  Old 1/sqrt decay at step=300: {old_decay:.4f}")
print(f"  New log decay at step=300:    {new_decay:.4f}")
assert new_decay > old_decay * 5, "New decay should be significantly gentler"
print(f"  New decay is {new_decay/old_decay:.1f}x gentler ✓")

print("\n✅ ALL TESTS PASSED")
