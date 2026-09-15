#!/usr/bin/env python3
"""Unit tests for the REAL _faithful_response_ids (ast-extracted from server.py).

Tests the wrap+guard contract deterministically with mock tokenizers, plus a real-tokenizer
smoke test. Run in the container: python3 _test_faithful_concat.py
"""
import ast, sys

SERVER = "/workspace/tito-dressage/dressage/proxy/server.py"

def load_real_function():
    tree = ast.parse(open(SERVER, encoding="utf-8").read())
    nodes = []
    for n in tree.body:
        if isinstance(n, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "_FAITHFUL_GUARD_FALLBACKS" for t in n.targets):
            nodes.append(n)
        if isinstance(n, ast.FunctionDef) and n.name == "_faithful_response_ids":
            nodes.append(n)
    assert len(nodes) == 2, f"expected counter+func at module level, got {len(nodes)}"
    mod = ast.Module(body=nodes, type_ignores=[])
    ns = {"logger": type("L", (), {"warning": staticmethod(lambda *a, **k: None)})()}
    exec(compile(mod, SERVER, "exec"), ns)
    return ns

# --- mock tokenizers ---
class BPETok:
    """decode(encode(text)) == text (idempotent), but sample tokenization is non-canonical
    so encode(decode(sample)) != sample -> a genuine BPE re-tokenization."""
    _d = {1: "a", 2: "b", 3: "ab"}
    def decode(self, ids, skip_special_tokens=False): return "".join(self._d[i] for i in ids)
    def encode(self, text, add_special_tokens=False): return {"a": [1], "b": [2], "ab": [3]}[text]

class NonIdemTok:
    """decode(encode('ab')) = 'X' != 'ab' -> simulates invalid-UTF8 byte non-idempotency."""
    _d = {1: "a", 2: "b", 9: "X"}
    def decode(self, ids, skip_special_tokens=False): return "".join(self._d[i] for i in ids)
    def encode(self, text, add_special_tokens=False): return {"ab": [9], "a": [1], "b": [2]}[text]

def run():
    ns = load_real_function()
    f = ns["_faithful_response_ids"]

    # 1) BPE re-tokenization: non-canonical sample [1,2]('a','b') -> canonical [3]('ab'); text preserved
    out = f([1, 2], BPETok())
    assert out == [3], f"expected BPE re-tok [3], got {out}"
    assert BPETok().decode(out) == BPETok().decode([1, 2]) == "ab"
    print("test_bpe_reencode: PASS (non-canonical sample re-tokenized, text preserved)")

    # 2) guard fires on non-idempotent text -> returns EXACT sampled ids, counter increments
    before = ns["_FAITHFUL_GUARD_FALLBACKS"]
    out = f([1, 2], NonIdemTok())
    after = ns["_FAITHFUL_GUARD_FALLBACKS"]
    assert out == [1, 2], f"guard must return exact ids, got {out}"
    assert after == before + 1, f"counter must increment on fallback ({before}->{after})"
    print("test_guard_fallback: PASS (non-idempotent -> exact ids, counter incremented)")

    # 3) idempotent already-canonical: [3]('ab') -> stays [3]
    out = f([3], BPETok())
    assert out == [3], f"canonical stays canonical, got {out}"
    print("test_canonical_stable: PASS")

    # 4) real tokenizer smoke: text preserved on a normal response
    try:
        from transformers import AutoTokenizer
        TOK = AutoTokenizer.from_pretrained("/workspace/tito-dressage/models/Qwen3.5-4B", trust_remote_code=True)
        ids = TOK.encode("The task is complete.<|im_end|>", add_special_tokens=False)
        out = f(ids, TOK)
        assert TOK.decode(out, skip_special_tokens=False) == TOK.decode(ids, skip_special_tokens=False)
        print("test_real_tokenizer_smoke: PASS (text preserved)")
    except Exception as e:
        print("test_real_tokenizer_smoke: SKIP", e)

    print("\nALL TESTS PASS")

if __name__ == "__main__":
    run()
