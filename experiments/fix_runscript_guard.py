import subprocess
P = "/workspace/tito-dressage/examples/scripts/run_alfworld_disagg_gb10_sync_repro.sh"
old = 'if [[ "${TOKEN_BUILD_MODE}" != "snapshot" && "${TOKEN_BUILD_MODE}" != "tito" ]]; then'
new = 'if [[ "${TOKEN_BUILD_MODE}" != "snapshot" && "${TOKEN_BUILD_MODE}" != "tito" && "${TOKEN_BUILD_MODE}" != "mimo" ]]; then'
s = open(P).read()
c = s.count(old)
assert c == 1, f"count={c}"
open(P + ".bak_mimo", "w").write(s)
open(P, "w").write(s.replace(old, new))
r = subprocess.run(["bash", "-n", P], capture_output=True, text=True)
print("bash -n:", "OK" if r.returncode == 0 else "FAIL\n" + r.stderr)
print("guard patched + backup .bak_mimo")
