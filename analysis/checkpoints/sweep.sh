#!/bin/bash
R=/workspace/tito-dressage/models
for M in "Qwen3.5-4B:base" "_hf_faithtext_34:it34" "_hf_faithtext_39:it39" "_hf_faithtext_44:it44" "_hf_faithtext_49:it49"; do
  D=${M%%:*}; T=${M##*:}
  echo "############ $T ############"
  docker exec tito python3 /workspace/tito-dressage/_probe.py $R/$D /workspace/tito-dressage/_probe_$T.json $T 2>/dev/null | grep -E '^\S+\|'
done
echo SWEEP_DONE
