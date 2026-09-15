"""Unit tests for _apply_faithful_mask (#54 MIMO faithful-mask ablation).

Loads the REAL function out of server.py via ast (no heavy imports), so it runs
under plain python3 on host or container:
    python3 dressage/proxy/tests/test_faithful_mask.py
    pytest  dressage/proxy/tests/test_faithful_mask.py
"""
import ast
import os

_CANDS = [
    os.environ.get("SERVER_PY"),
    "/home/maziar/projects/tito-dressage/dressage/proxy/server.py",
    "/workspace/tito-dressage/dressage/proxy/server.py",
]
_SERVER = next((p for p in _CANDS if p and os.path.exists(p)), None)
assert _SERVER, "could not locate server.py"
_src = open(_SERVER).read()
_fn_src = next(
    (ast.get_source_segment(_src, n) for n in ast.parse(_src).body
     if isinstance(n, ast.FunctionDef) and n.name == "_apply_faithful_mask"),
    None,
)
assert _fn_src, "_apply_faithful_mask not found in server.py"
_ns = {}
exec(_fn_src, _ns)
_apply_faithful_mask = _ns["_apply_faithful_mask"]

THINK, NL, IM_END = 248068, 198, 248046

def _fixture():
    # 3 assistant-masked runs in one re-rendered sequence:
    #   A: normal turn  (leading <think>\n ... im_end AFTER the run)
    #   B: no leading <think> (malformed turn; im_end AFTER)
    #   C: final turn at end-of-sequence (leading <think>\n; NO im_end after)
    tokens = [900, 901, THINK, NL, 500, 501, 502, IM_END, 198,
              910, 600, IM_END, 198,
              900, 901, 700, 701, IM_END, 198,
              900, 901, THINK, NL, 800, 801]
    mask =   [0, 0, 1, 1, 1, 1, 1, 0, 0,
              0, 0, 0, 0,
              0, 0, 1, 1, 0, 0,
              0, 0, 1, 1, 1, 1]
    assert len(tokens) == len(mask) == 25
    return tokens, mask

def _ones(mask):
    return {i for i, m in enumerate(mask) if m == 1}

def test_default_off_is_strict_noop():
    t, m = _fixture(); before = list(m)
    out = _apply_faithful_mask(t, m, train_imend=False, no_think=False)
    assert m == before, "default-off must leave the mask byte-identical"
    assert out is m

def test_faithful_both_matches_sampled_span():
    t, m = _fixture()
    _apply_faithful_mask(t, m, train_imend=True, no_think=True)
    assert _ones(m) == {4, 5, 6, 7, 15, 16, 17, 23, 24}, _ones(m)
    masked = [t[i] for i in sorted(_ones(m))]
    assert masked == [500, 501, 502, IM_END, 700, 701, IM_END, 800, 801], masked
    assert THINK not in masked

def test_train_imend_only():
    t, m = _fixture()
    _apply_faithful_mask(t, m, train_imend=True, no_think=False)
    assert _ones(m) == {2, 3, 4, 5, 6, 7, 15, 16, 17, 21, 22, 23, 24}, _ones(m)

def test_no_think_only():
    t, m = _fixture()
    _apply_faithful_mask(t, m, train_imend=False, no_think=True)
    assert _ones(m) == {4, 5, 6, 15, 16, 23, 24}, _ones(m)

def test_idempotent():
    t, m = _fixture()
    _apply_faithful_mask(t, m, train_imend=True, no_think=True); once = list(m)
    _apply_faithful_mask(t, m, train_imend=True, no_think=True)
    assert m == once, "faithful mask must be idempotent"

def test_run_at_end_no_imend_no_indexerror():
    t, m = _fixture()
    _apply_faithful_mask(t, m, train_imend=True, no_think=False)
    assert {21, 22, 23, 24} <= _ones(m)
    assert max(_ones(m)) < len(t)

def test_no_leading_think_run_untouched_by_no_think():
    t, m = _fixture()
    _apply_faithful_mask(t, m, train_imend=False, no_think=True)
    assert {15, 16} <= _ones(m)

def test_empty_and_all_zero():
    assert _apply_faithful_mask([], [], train_imend=True, no_think=True) == []
    assert _apply_faithful_mask([1, 2, 3], [0, 0, 0], train_imend=True, no_think=True) == [0, 0, 0]

if __name__ == "__main__":
    fns = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_")]
    for name, fn in fns:
        fn(); print(f"PASS {name}")
    print(f"\nALL {len(fns)} TESTS PASSED  (loaded from {_SERVER})")
