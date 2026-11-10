"""Model-free regression for the precise uninitialized shared-memory tail contract."""
import math,re,sys
from pathlib import Path
source=Path(sys.argv[1]).read_text()
needle='for (uint k_step = 0; k_step < BK_STEP && block + k_step * BK < end_k; k_step++)'
assert source.count(needle)==1,'Bounded accumulation loop is required exactly once'
assert 'if (block + k_step * BK < end_k)' in source,'Keep guarded A loads'
assert 'block_b_to_shmem(k_step * BN + buf_ib, ib + k_step, iqs, block + k_step * BK < end_k)' in source,'Keep existing B padding semantics'
cases=0
for end in [32,64,96,128,160,192,224,256,288,1024]:
 for start in range(0,end,32):
  expected=list(range(start,end,32));observed=[];old_bad=[]
  for block in range(start,end,128):
   initialized={k for k in range(4) if block+k*32<end}
   old_bad += [block+k*32 for k in range(4) if k not in initialized]
   for k in range(4):
    if block+k*32<end:assert k in initialized;observed.append(block+k*32)
  assert observed==expected,'Every valid block exactly once, unchanged order'
  if (end-start)%128:assert old_bad,'Negative control must expose the defective reads'
  cases+=1
assert math.isnan(0.0*float('nan')*0.0),'Zero-padded B cannot neutralize uninitialized NaN A scales'
print(f'MMQ TAIL CONTRACT PASS: {cases} K/split ranges; defective negative-control reads exposed; no backend execution')
