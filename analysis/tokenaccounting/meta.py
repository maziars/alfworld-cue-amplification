import pickle, collections, sys
class U(pickle.Unpickler):
    def find_class(self, m, n):
        class D0:
            def __init__(self,*a,**k): pass
        if m.startswith("megatron") or m.startswith("glm"): return D0
        return super().find_class(m,n)
md=U(open(sys.argv[1],"rb")).load()
sd=md.storage_data
byfile=collections.defaultdict(lambda: collections.Counter())
for k,v in sd.items():
    fn=getattr(v,"relative_path","?")
    key=str(k); pre = "WEIGHT" if not key.startswith("optimizer") else "OPTIM"
    byfile[fn][pre]+=1
for fn in sorted(byfile): print(f"  {fn:<16} {dict(byfile[fn])}")
# are any weight tensors ONLY in rank-0 files?
w0=[str(k) for k,v in sd.items() if not str(k).startswith("optimizer") and getattr(v,"relative_path","").startswith("__0")]
w1=[str(k) for k,v in sd.items() if not str(k).startswith("optimizer") and getattr(v,"relative_path","").startswith("__1")]
print(f"\n  weight tensors in rank-0 files: {len(w0)}")
print(f"  weight tensors in rank-1 files: {len(w1)}")
print("  sample rank-0-only weights:", w0[:4])
