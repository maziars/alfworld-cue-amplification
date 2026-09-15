"""Test the ENV-AUTHORITATIVE parser (SGLang qwen3_coder) on the exact malformations seen in rollouts."""
import json
from sglang.srt.function_call.qwen3_coder_detector import Qwen3CoderDetector
from sglang.srt.entrypoints.openai.protocol import Tool, Function
det=Qwen3CoderDetector()
tools=[Tool(type="function", function=Function(name="env_step", description="step",
      parameters={"type":"object","properties":{"command":{"type":"string"}},"required":["command"]}))]
def T(label, body):
    txt="reasoning here\n</think>\n\n"+body
    try:
        r=det.detect_and_parse(txt, tools)
        calls=[(c.name, c.parameters) for c in (r.calls or [])]
    except Exception as e:
        calls=[("EXC", str(e)[:60])]
    ok = bool(calls) and calls[0][0]=="env_step" and '"command": "go to desk 1"'.replace(' ','') in str(calls[0][1]).replace(' ','')
    print(f"{'PASS' if ok else 'FAIL':>4} | {label:<46} -> {calls if calls else 'NO CALL'}")

INNER='<parameter=command>\ngo to desk 1\n</parameter>\n</function>\n</tool_call>'
print("=== A. leading '<' accretion on <function=  (parity claim) ===")
for n in range(0,6):
    T(f"{'<'*n}function=env_step  ({n} extra '<')", "<tool_call>\n"+"<"*n+"<function=env_step>\n"+INNER)
print("\n=== B. same, but WITHOUT the <tool_call> opener ===")
for n in range(0,4):
    T(f"no <tool_call>, {n} extra '<'", "<"*n+"<function=env_step>\n"+INNER)
print("\n=== C. duplicated 'function=' prefix ===")
T("<function=<function=env_step>", "<tool_call>\n<function=<function=env_step>\n"+INNER)
T("<<function=<function=env_step>", "<tool_call>\n<<function=<function=env_step>\n"+INNER)
print("\n=== D. broken CLOSING tags ===")
T("</<parameter>", "<tool_call>\n<function=env_step>\n<parameter=command>\ngo to desk 1\n</<parameter>\n</function>\n</tool_call>")
T("</</function>", "<tool_call>\n<function=env_step>\n<parameter=command>\ngo to desk 1\n</parameter>\n</</function>\n</tool_call>")
T("<</tool_call>", "<tool_call>\n<function=env_step>\n<parameter=command>\ngo to desk 1\n</parameter>\n</function>\n<</tool_call>")
print("\n=== E. leading '<' on <parameter= ===")
for n in range(0,4):
    T(f"{n} extra '<' on parameter", "<tool_call>\n<function=env_step>\n"+"<"*n+"<parameter=command>\ngo to desk 1\n</parameter>\n</function>\n</tool_call>")
print("\n=== F. wrong name in slot ===")
for nm in ("call","tool_name","env_step"):
    T(f"<function={nm}>", "<tool_call>\n<function=%s>\n"%nm+INNER)
