#!/bin/bash
R=/workspace/tito-dressage/models
run(){ docker exec tito python3 /workspace/tito-dressage/_probe3.py $R/$1 /workspace/tito-dressage/_r_$2.json $2 2>/dev/null; }
run Qwen3.5-4B base
for I in 34 39 44 49; do run _hf_faithtext_$I ft$I; done
echo SWEEP4_DONE
