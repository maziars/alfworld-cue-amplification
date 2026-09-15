import torch, re
from transformers import AutoTokenizer
TOK=AutoTokenizer.from_pretrained("/workspace/tito-dressage/models/Qwen3.5-4B", trust_remote_code=True)
d=torch.load("/workspace/tito-dressage/experiments/faithtext_tito/debug/5.pt", map_location="cpu", weights_only=False)
txt=TOK.decode(list(d["samples"][0]["tokens"]), skip_special_tokens=False)
i=txt.find("</tool_call>")
print("=== assistant-end -> next-turn start ===")
print(repr(txt[i:i+520]))
print("\n=== all im_start role markers, in order (first 10) ===")
print(re.findall(r"<\|im_start\|>(\w+)", txt)[:10])
