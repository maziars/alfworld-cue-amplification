"""Unit tests for _apply_faithful_text (#58/#61 faithful-text MIMO).

Ast-loads the pure function from server.py (no heavy imports) and exercises it
with a mock tokenizer. Run: python3 test_faithful_text.py

#61: the original impl dropped the template's leading '<think>\\n' generation-prompt
prefill from the training context (it lives INSIDE the mask==1 span). These tests
now assert the prefill is PRESERVED as mask=0 context before the raw span, and that
a redundant template '<|im_end|>' is not duplicated.
"""
import ast, os

SERVER = os.environ.get("SERVER_PY", "/workspace/tito-dressage/dressage/proxy/server.py")
THINK, NL, IM_END = 248068, 198, 248046

def _load(fn_name):
    tree = ast.parse(open(SERVER).read())
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == fn_name:
            ns = {}
            exec(compile(ast.Module(body=[node], type_ignores=[]), SERVER, "exec"), ns)
            return ns[fn_name]
    raise AssertionError(f"{fn_name} not found in {SERVER}")

apply_faithful_text = _load("_apply_faithful_text")

class Tok:
    """decode -> 'a|b|c' ; encode -> content+1000, but SPECIAL tokens preserved
    (so encode(decode([<think>, \\n, ...])) round-trips the specials)."""
    SPECIAL = {THINK, NL, IM_END, 248069, 271}
    def decode(self, ids, skip_special_tokens=False):
        return "|".join(str(i) for i in ids)
    def encode(self, text, add_special_tokens=False):
        return [int(p) if int(p) in self.SPECIAL else int(p) + 1000 for p in text.split("|")]

class MergeTok:
    def decode(self, ids, skip_special_tokens=False): return "|".join(str(i) for i in ids)
    def encode(self, text, add_special_tokens=False): return [900000 + sum(int(p) for p in text.split("|"))]

def runs_of(mask):
    r=[]; i=0; n=len(mask)
    while i<n:
        if mask[i]==1:
            s=i
            while i<n and mask[i]==1: i+=1
            r.append((s,i))
        else: i+=1
    return r

def run():
    tok = Tok(); P = 0; n = 0

    # 1) span WITHOUT think-prefill -> whole span replaced (backward compat)
    nt, nm = apply_faithful_text([P,P,5,6,7,P], [0,0,1,1,1,0], [[5,6,7]], tok)
    assert nt==[P,P,1005,1006,1007,P] and nm==[0,0,1,1,1,0], (nt,nm); n+=1

    # 2) faithful span comes from SAMPLED ids, not the canonical span tokens
    nt, nm = apply_faithful_text([P,999,999,P], [0,1,1,0], [[5,6,7]], tok)
    assert nt==[P,1005,1006,1007,P] and nm==[0,1,1,1,0], (nt,nm); n+=1

    # 3) multi-turn, no prefill: two spans replaced, context preserved
    nt, nm = apply_faithful_text([P,1,2,P,P,8,9,P],[0,1,1,0,0,1,1,0],[[1,2],[8,9]],tok)
    assert nt==[P,1001,1002,P,P,1008,1009,P] and nm==[0,1,1,0,0,1,1,0], (nt,nm); n+=1

    # 4) count mismatch -> strict no-op
    t,m=[P,1,2,P,8,9,P],[0,1,1,0,1,1,0]
    nt,nm=apply_faithful_text(list(t),list(m),[[1,2]],tok)
    assert nt==t and nm==m; n+=1

    # 5) merge direction handled
    nt,nm=apply_faithful_text([P,5,6,7,P],[0,1,1,1,0],[[5,6,7]],MergeTok())
    assert nt==[P,900018,P] and nm==[0,1,0] and len(nt)==len(nm); n+=1

    # ---- #61 regression tests: think-prefill preservation ----
    # 6) span STARTS with <think>\n (as the real mask template renders it).
    #    canonical run = [<think>, \n, 100, 200] (mask=1); template <|im_end|> after (mask=0).
    #    raw sampled = [100,200,<|im_end|>].  Expect: <think>\n kept as mask=0 CONTEXT,
    #    raw as mask=1, redundant template <|im_end|> skipped (no double).
    tks=[9, THINK, NL, 100, 200, IM_END, 9]
    msk=[0, 1,     1,  1,   1,   0,      0]
    nt, nm = apply_faithful_text(list(tks), list(msk), [[100,200,IM_END]], tok)
    assert nt==[9, THINK, NL, 1100, 1200, IM_END, 9], nt
    assert nm==[0, 0,     0,  1,    1,     1,      0], nm
    # the prefill is present and mask=0, immediately before the mask=1 span
    rs = runs_of(nm)[0][0]
    assert rs>=2 and nt[rs-2]==THINK and nt[rs-1]==NL and nm[rs-2]==0 and nm[rs-1]==0, "prefill not preserved as context"
    # exactly one <|im_end|> (no double), and it is trained (mask=1)
    assert nt.count(IM_END)==1 and nm[nt.index(IM_END)]==1, "im_end double or untrained"
    n+=1

    # 7) BUGGY behavior would have produced no prefill + double im_end -> assert we are NOT that
    assert nt != [9, 1100, 1200, IM_END, IM_END, 9], "regressed to the buggy whole-span replacement"
    n+=1

    # 8) prefill preserved, but raw has NO trailing im_end -> do not skip template im_end
    tks=[9, THINK, NL, 100, 200, IM_END, 9]; msk=[0,1,1,1,1,0,0]
    nt, nm = apply_faithful_text(list(tks), list(msk), [[100,200]], tok)
    assert nt==[9, THINK, NL, 1100, 1200, IM_END, 9], nt
    assert nm==[0, 0, 0, 1, 1, 0, 0], nm          # template im_end stays (mask=0), not skipped
    n+=1

    # 9) invariant: every mask==1 run whose original span began with <think>\n is
    #    preceded in the OUTPUT by <think>\n at mask=0 (the property the bug violated)
    tks=[9, THINK,NL,10,IM_END, 5, THINK,NL,20,IM_END, 9]
    msk=[0, 1,1,1,1,             0, 1,1,1,1,            0]
    nt, nm = apply_faithful_text(list(tks), list(msk), [[10,IM_END],[20,IM_END]], tok)
    for s,e in runs_of(nm):
        assert s>=2 and nt[s-1]==NL and nt[s-2]==THINK and nm[s-1]==0 and nm[s-2]==0, f"run at {s} missing prefill"
    n+=1

    print(f"ALL {n} faithful-text tests passed (incl. #61 prefill-preservation regression tests)")

if __name__ == "__main__":
    run()
