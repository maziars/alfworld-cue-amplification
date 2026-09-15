import json, glob, gzip, collections
E="/workspace/tito-dressage/experiments/faithtext_tito"
for it in (5,30,42,48):
    f=E+"/trajectories/rollout_%d.jsonl.gz"%it
    T=[json.loads(l) for l in gzip.open(f,'rt')]
    sc=collections.Counter(t.get("segment_count") for t in T)
    sid=collections.Counter(t["session_id"] for t in T)
    dup=sum(1 for k,v in sid.items() if v>1)
    # per-traj: assistant turns in messages vs invalid_action_count
    bad=0; ex=None
    for t in T:
        na=sum(1 for m in (t.get("messages") or []) if m.get("role")=="assistant")
        ninv=sum(1 for m in (t.get("messages") or []) if m.get("role")=="tool" and (m.get("content") or "").startswith("Invalid tool call"))
        if ninv != (t.get("invalid_action_count") or 0):
            bad+=1
            if ex is None: ex=(t["session_id"], na, ninv, t.get("invalid_action_count"), t.get("num_steps"), t.get("segment_count"), t.get("segment_index"), t.get("selected_segment_index"))
    print("it%-3d n=%d  segment_count dist=%s  dup_session=%d  mismatch(inv_in_msgs != invalid_action_count)=%d/%d"%(it,len(T),dict(sc),dup,bad,len(T)))
    if ex: print("        example: sid=%s n_assist=%d inv_in_msgs=%d invalid_action_count=%s num_steps=%s seg_count=%s seg_idx=%s sel=%s"%ex)
