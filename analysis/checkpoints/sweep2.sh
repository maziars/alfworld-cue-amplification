#!/bin/bash
R=/workspace/tito-dressage/models
run(){ docker exec tito python3 /workspace/tito-dressage/_probe2.py $R/$1 /workspace/tito-dressage/_p_$2.json $2 2>/dev/null; }
run Qwen3.5-4B base
for I in 34 39 44 49; do run _hf_faithtext_$I ft$I; done
echo FT_SWEEP_DONE
for I in 19 29 39 49 99; do docker exec tito test -d $R/_hf_tito_$I && run _hf_tito_$I tito$I; done
for I in 4 14 24 34; do docker exec tito test -d $R/_hf_mimo100_$I && run _hf_mimo100_$I mimo100_$I; done
echo ALL_SWEEP_DONE
