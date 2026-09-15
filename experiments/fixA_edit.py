import ast
P = "/workspace/tito-dressage/dressage/proxy/server.py"
old = '            request_logprob_start_len = -1 if token_build_mode == "tito" else 0'
new = '            request_logprob_start_len = -1 if token_build_mode in ("tito", "mimo") else 0'
s = open(P).read()
c = s.count(old)
assert c == 1, f"count={c}"
open(P + ".bak_fixA", "w").write(s)
ns = s.replace(old, new)
ast.parse(ns)
open(P, "w").write(ns)
print("fixA applied + AST ok; backup .bak_fixA")
