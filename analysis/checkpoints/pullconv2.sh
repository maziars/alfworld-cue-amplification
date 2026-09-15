#!/bin/bash
set -u
HOSTROOT=${PROJECT_ROOT}
CROOT=/workspace/tito-dressage
SRC=${PEER_NODE}
CKPT=Qwen3.5-4B_colocate_faithtext
mkdir -p $HOSTROOT/_shards
for IT in 34 39 44 49; do
  P=$(printf "iter_%07d" $IT)
  OUT=$CROOT/models/_hf_faithtext_$IT
  if docker exec tito test -f "$OUT/model.safetensors.index.json"; then echo "[$(date +%T)] SKIP $IT"; continue; fi
  S=$HOSTROOT/_shards/$P; mkdir -p $S
  echo "[$(date +%T)] === iter $IT : pull rank-1 shards ==="
  for F in __1_0.distcp __1_1.distcp; do
    [ -f "$S/$F" ] || rsync -a --inplace --info=progress2 -e "ssh -o StrictHostKeyChecking=no" \
      $SRC:$HOSTROOT/models/$CKPT/$P/$F "$S/$F" 2>&1 | tail -1
  done
  echo "[$(date +%T)] === iter $IT : stage (hardlink local shards) ==="
  docker exec tito bash -lc "
    ST=$CROOT/models/_stage_$IT; rm -rf \$ST; mkdir -p \$ST
    CK=$CROOT/models/$CKPT/$P
    ln -f \$CK/__0_0.distcp \$CK/__0_1.distcp \$CK/common.pt \$ST/ 2>/dev/null || cp \$CK/__0_0.distcp \$CK/__0_1.distcp \$CK/common.pt \$ST/
    cp \$CK/.metadata \$ST/.metadata
    ln -f $CROOT/_shards/$P/__1_0.distcp $CROOT/_shards/$P/__1_1.distcp \$ST/ 2>/dev/null || cp $CROOT/_shards/$P/__1_*.distcp \$ST/
    ls -la \$ST | tr '\n' ' '; echo"
  echo "[$(date +%T)] === iter $IT : convert ==="
  docker exec tito python3 $CROOT/slime/tools/convert_torch_dist_to_hf.py \
    --input-dir $CROOT/models/_stage_$IT --output-dir $OUT \
    --origin-hf-dir $CROOT/models/Qwen3.5-4B -f -a 2>&1 | tail -4
  docker exec tito test -f "$OUT/model.safetensors.index.json" && echo "[$(date +%T)] CONVERTED $IT" || echo "[$(date +%T)] FAILED $IT"
  docker exec tito rm -rf $CROOT/models/_stage_$IT
  rm -rf $S
  df -h ${HOME_DIR} | tail -1
done
echo "[$(date +%T)] ALL DONE"
