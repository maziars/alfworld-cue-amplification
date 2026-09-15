#!/bin/bash
set -u
HOSTROOT=${PROJECT_ROOT}
CROOT=/workspace/tito-dressage
SRC=${PEER_NODE}
mkdir -p $HOSTROOT/_shards
conv(){  # run_dir tag iter
  local CK=$1 TAG=$2 IT=$3
  local P=$(printf "iter_%07d" $IT)
  local OUT=$CROOT/models/_hf_${TAG}_${IT}
  docker exec tito test -f "$OUT/model.safetensors.index.json" && { echo "[$(date +%T)] SKIP $TAG $IT"; return; }
  docker exec tito test -d "$CROOT/models/$CK/$P" || { echo "[$(date +%T)] MISSING rank0 $TAG $IT"; return; }
  local S=$HOSTROOT/_shards/${TAG}_$P; mkdir -p $S
  for F in __1_0.distcp __1_1.distcp; do
    [ -f "$S/$F" ] || rsync -a --inplace -e "ssh -o StrictHostKeyChecking=no" \
      $SRC:$HOSTROOT/models/$CK/$P/$F "$S/$F" >/dev/null 2>&1
  done
  docker exec tito bash -lc "
    ST=$CROOT/models/_stage_${TAG}_$IT; rm -rf \$ST; mkdir -p \$ST
    C=$CROOT/models/$CK/$P
    ln -f \$C/__0_0.distcp \$C/__0_1.distcp \$C/common.pt \$ST/ 2>/dev/null || cp \$C/__0_*.distcp \$C/common.pt \$ST/
    cp \$C/.metadata \$ST/.metadata
    ln -f $CROOT/_shards/${TAG}_$P/__1_0.distcp $CROOT/_shards/${TAG}_$P/__1_1.distcp \$ST/ 2>/dev/null || cp $CROOT/_shards/${TAG}_$P/__1_*.distcp \$ST/"
  docker exec tito python3 $CROOT/slime/tools/convert_torch_dist_to_hf.py \
    --input-dir $CROOT/models/_stage_${TAG}_$IT --output-dir $OUT \
    --origin-hf-dir $CROOT/models/Qwen3.5-4B -f -a >/dev/null 2>&1
  docker exec tito test -f "$OUT/model.safetensors.index.json" && echo "[$(date +%T)] OK $TAG $IT" || echo "[$(date +%T)] FAIL $TAG $IT"
  docker exec tito rm -rf $CROOT/models/_stage_${TAG}_$IT; rm -rf $S
}
for IT in 19 29 39 49 99; do conv Qwen3.5-4B_colocate_run_v18 tito $IT; done
for IT in 4 14 24 34; do conv Qwen3.5-4B_colocate_mimo100 mimo100 $IT; done
df -h ${HOME_DIR} | tail -1
echo CONVALL_DONE
