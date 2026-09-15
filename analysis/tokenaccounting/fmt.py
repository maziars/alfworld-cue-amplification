import torch
from transformers import AutoTokenizer
TOK=AutoTokenizer.from_pretrained("/workspace/tito-dressage/models/Qwen3.5-4B", trust_remote_code=True)
d=torch.load("/workspace/tito-dressage/experiments/faithtext_tito/debug/5.pt", map_location="cpu", weights_only=False)
s=d["samples"][0]; txt=TOK.decode(list(s["tokens"]), skip_special_tokens=False)
i=txt.find("<tool_call>")
print("=== around first tool_call (turn boundary format) ===")
print(repr(txt[max(0,i-260):i+700]))
