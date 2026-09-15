from sglang.srt.function_call.qwen3_coder_detector import Qwen3CoderDetector
from sglang.srt.entrypoints.openai.protocol import Tool, Function
det=Qwen3CoderDetector()
tools=[Tool(type="function", function=Function(name="env_step", parameters={"type":"object","properties":{"command":{"type":"string"}},"required":["command"]}))]
def T(lbl, txt):
    try:
        r=det.detect_and_parse(txt, tools); c=[(x.name,x.parameters) for x in (r.calls or [])]
    except Exception as e: c=[("EXC",str(e)[:40])]
    print(f"{'CALL' if c else 'NONE':>5} | {lbl:<52} -> {c if c else ''}")
P="reasoning\n</think>\n\n"
FN="<function=env_step>\n<parameter=command>\ngo to desk 1\n</parameter>\n</function>"
print("== does SGLang need the CLOSER </tool_call>? ==")
T("opener + body + closer",      P+"<tool_call>\n"+FN+"\n</tool_call>")
T("opener + body, NO closer",    P+"<tool_call>\n"+FN)
T("opener + body, no closer, <|im_end|>", P+"<tool_call>\n"+FN+"<|im_end|>")
print("\n== opener present but body has NO valid <function=...> ==")
T("opener + junk only",          P+"<tool_call>\n<<<<<<<<<<<< garbage")
T("opener + <parameter only",    P+"<tool_call>\n<parameter=command>\ngo to desk 1\n</parameter>\n</tool_call>")
print("\n== body WITHOUT the opener (the terminal attractor shape) ==")
T("no opener, full body+closer", P+FN+"\n</tool_call>")
T("no opener, body only",        P+FN)
print("\n== opener only in reasoning ==")
T("opener in reasoning, body after </think>, closer present","I write <tool_call> first.\n</think>\n\n"+FN+"\n</tool_call>")
T("opener in reasoning, body after </think>, NO closer","I write <tool_call> first.\n</think>\n\n"+FN)
