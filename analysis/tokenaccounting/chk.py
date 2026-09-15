import pickle, collections
C=pickle.load(open("/workspace/tito-dressage/_probe_ctx.pkl","rb"))
for a,b,lbl in (("E1_r5_TITOtok","E2_r5_MIMOtok","healthy r5"),("E3_r49_TITOtok","E4_r49_MIMOtok","corrupt r49")):
    for pos in ("TAG_START","AFTER_ONE_LT","AFTER_EQ","AFTER_SLASH"):
        x=C.get(f"{a}|{pos}"); y=C.get(f"{b}|{pos}")
        if not x or not y: continue
        same=sum(1 for u,v in zip(x,y) if u==v)
        dl=[abs(len(set(u))-len(set(v))) for u,v in zip(x,y)]
        print(f"{lbl:<12} {pos:<14} identical token windows: {same}/{len(x)}   differing: {len(x)-same}")
