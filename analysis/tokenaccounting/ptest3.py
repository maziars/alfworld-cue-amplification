from sglang.srt.function_call.qwen3_coder_detector import Qwen3CoderDetector
from sglang.srt.entrypoints.openai.protocol import Tool, Function
det=Qwen3CoderDetector()
print("bot_token=%r  eot_token=%r"%(getattr(det,'bot_token',None), getattr(det,'eot_token',None)))
tools=[Tool(type="function", function=Function(name="env_step", parameters={"type":"object","properties":{"command":{"type":"string"}},"required":["command"]}))]
INNER='<function=env_step>\n<parameter=command>\ngo to desk 1\n</parameter>\n</function>\n</tool_call>'
def T(label, txt):
    try:
        r=det.detect_and_parse(txt, tools); calls=[(c.name,c.parameters) for c in (r.calls or [])]
    except Exception as e: calls=[("EXC",str(e)[:50])]
    ok=bool(calls) and calls[0][0]=="env_step"
    print(f"{'PASS' if ok else 'FAIL':>4} | {label:<50} -> {calls if calls else 'NO CALL'}")
P="reasoning\n</think>\n\n"
print("\n=== ACCRETION ON THE <tool_call> OPENER ITSELF ===")
for n in range(0,5): T(f"{'<'*n}<tool_call>  ({n} extra)", P+"<"*n+"<tool_call>\n"+INNER)
print("\n=== OPENER PRESENT ONLY INSIDE REASONING (before </think>) ===")
T("opener in reasoning, body after </think>", "I should write <tool_call> first.\n</think>\n\n"+INNER)
print("\n=== OPENER VARIANTS ===")
T("<tool_calls> (plural)", P+"<tool_calls>\n"+INNER)
T("<tool_call> with space", P+"< tool_call>\n"+INNER)
T("no opener, closer present", P+INNER)
print("\n=== CANONICAL JSON FORM (what the PROMPT asks for) ===")
T("prompt's JSON template", P+'<tool_call>\n{"name": "env_step", "arguments": {"command": "go to desk 1"}}\n</tool_call>')
T("JSON form + prose mentions <tool_call> earlier", 'Let me put <tool_call> before it.\n</think>\n\n<tool_call>\n{"name": "env_step", "arguments": {"command": "go to desk 1"}}\n</tool_call>')
print("\n=== BROKEN PARAMETER CLOSER variants ===")
for c in ("</parameter>","</<parameter>","</ parameter>","<</parameter>","</paramete>"):
    T("closer=%r"%c, P+"<tool_call>\n<function=env_step>\n<parameter=command>\ngo to desk 1\n"+c+"\n</function>\n</tool_call>")
