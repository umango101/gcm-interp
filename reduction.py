import torch, einops, glob, re
for d in sorted(glob.glob("results/gpt-oss-20b/pos-*__from_user-single_to_dev-single/atp")):
    fs = sorted(glob.glob(f"{d}/heads_*.pt"),
                key=lambda f: int(re.search(r"heads_(\d+)\.pt", f).group(1)))
    cols = [torch.load(f).squeeze().unsqueeze(-1) for f in fs]
    m = einops.reduce(torch.cat(cols, dim=-1), "l (n m) b -> l n b", "sum", n=64)
    torch.save(m, f"{d}/numerator_1_heads.pt")
    print(d.split("/")[2], tuple(m.shape), len(fs), "shards")
