"""Is a SINGLE '<' fatal at each position? Test against the real first_command chain."""
import sys; sys.path.insert(0,"/workspace/tito-dressage")
from dressage.recipes.alfworld.tools import first_command
from sglang.srt.function_call.qwen3_coder_detector import Qwen3CoderDetector
from sglang.srt.entrypoints.openai.protocol import Tool, Function
det=Qwen3CoderDetector()
tools=[Tool(type="function", function=Function(name="env_step", parameters={"type":"object","properties":{"command":{"type":"string"}},"required":["command"]}))]
def sg(t):
    try:
        r=det.detect_and_parse(t,tools)
        return [{"id":"x","function":{"name":c.name,"arguments":c.parameters}} for c in (r.calls or [])]
    except Exception: return []
def T(lbl, body):
    c="reasoning\n</think>\n\n"+body
    tid,cmd,rs=first_command(c, sg(c))
    print(f"  {'OK ' if cmd else 'FATAL'} | {lbl:<44} -> " + (repr(cmd) if cmd else rs))
OK="<tool_call>\n<function=env_step>\n<parameter=command>\ngo to desk 1\n</parameter>\n</function>\n</tool_call>"
print("=== baseline ==="); T("clean", OK)
print("\n=== ONE extra '<' right after '<function=' ===")
for ins in ("<","<<","<<<","<<<<"):
    T(f"<function={ins}env_step>", OK.replace("<function=env_step>", f"<function={ins}env_step>"))
T("<function=<function=env_step>", OK.replace("<function=env_step>","<function=<function=env_step>"))
print("\n=== extra '<' right after '</' , BY CLOSER TYPE ===")
for tag in ("parameter","function","tool_call"):
    for ins in ("<","<<"):
        T(f"</{ins}{tag}>", OK.replace(f"</{tag}>", f"</{ins}{tag}>"))
print("\n=== extra '<' BEFORE a tag word (the safe slot), for contrast ===")
for ins in ("<","<<","<<<<"):
    T(f"{ins}<function=env_step>", OK.replace("<function=env_step>", f"{ins}<function=env_step>"))
    T(f"{ins}<parameter=command>", OK.replace("<parameter=command>", f"{ins}<parameter=command>"))
