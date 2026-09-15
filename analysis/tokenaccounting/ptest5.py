import sys; sys.path.insert(0,"/workspace/tito-dressage")
from dressage.recipes.alfworld.tools import first_command
from sglang.srt.function_call.qwen3_coder_detector import Qwen3CoderDetector
from sglang.srt.entrypoints.openai.protocol import Tool, Function
det=Qwen3CoderDetector()
tools=[Tool(type="function", function=Function(name="env_step", parameters={"type":"object","properties":{"command":{"type":"string"}},"required":["command"]}))]
def sg(txt):
    try:
        r=det.detect_and_parse(txt, tools)
        return [{"id":"x","function":{"name":c.name,"arguments":c.parameters}} for c in (r.calls or [])]
    except Exception: return []
def T(lbl, content):
    s=sg(content); tid,cmd,reason=first_command(content, s)
    print(f"  {lbl:<50} sglang={len(s)}  " + ("OK -> %r"%cmd if cmd else "INVALID: %s"%reason))
P="reasoning\n</think>\n\n"
FN="<function=env_step>\n<parameter=command>\ngo to desk 1\n</parameter>\n</function>"
J='{"name": "env_step", "arguments": {"command": "go to desk 1"}}'
print("=== FORMAT THE PROMPT INSTRUCTS ===")
T("prompt JSON, with closer", P+"<tool_call>\n"+J+"\n</tool_call>")
T("prompt JSON, NO closer",   P+"<tool_call>\n"+J)
T("prompt JSON + im_end",     P+"<tool_call>\n"+J+"\n</tool_call><|im_end|>")
print("=== XML FORM THE MODEL EMITS ===")
T("xml, with closer",         P+"<tool_call>\n"+FN+"\n</tool_call>")
T("xml, NO opener (terminal)",P+FN+"\n</tool_call>")
print("=== EDGE ===")
T("opener + mangled xml + closer", P+"<tool_call>\n<function<tool_name>\n<parameter=command>\ngo to desk 1\n</parameter>\n</function>\n</tool_call>")
T("opener + junk, no closer", P+"<tool_call>\n<<<<<<< garbage")
T("prose only",               P+"go to desk 1")
