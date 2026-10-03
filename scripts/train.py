"""Fine-tuning MobileNetV3-Small: python scripts/train.py --mode feature|partial|scratch"""
import argparse, json, random, time
from pathlib import Path
import numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms as T

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
N_UNFREEZE = 4  # blok terakhir backbone yang dibuka pada mode partial

def build(mode, n_cls):
    w = None if mode == "scratch" else models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
    m = models.mobilenet_v3_small(weights=w)
    m.classifier[3] = nn.Linear(m.classifier[3].in_features, n_cls)  # head baru
    if mode == "feature":                       # backbone beku, head dilatih
        for p in m.features.parameters(): p.requires_grad = False
    elif mode == "partial":                     # blok akhir + head dilatih
        for p in m.features[:-N_UNFREEZE].parameters(): p.requires_grad = False
    return m

def optimizer(m, mode):
    if mode == "partial":
        return torch.optim.Adam([
            {"params": m.features[-N_UNFREEZE:].parameters(), "lr": 1e-4},
            {"params": m.classifier.parameters(), "lr": 1e-3}])
    return torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=1e-3)

def set_train(m, mode):
    m.train()
    # BatchNorm pada lapisan beku tetap eval() agar statistik ImageNet tidak berubah
    if mode == "feature": m.features.eval()
    if mode == "partial": m.features[:-N_UNFREEZE].eval()

def run_epoch(m, loader, dev, opt=None, mode=None):
    train = opt is not None
    set_train(m, mode) if train else m.eval()
    loss_sum = correct = n = 0
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(dev), y.to(dev)
            out = m(x); loss = nn.functional.cross_entropy(out, y)
            if train:
                opt.zero_grad(); loss.backward(); opt.step()
            loss_sum += loss.item() * len(y); correct += (out.argmax(1) == y).sum().item(); n += len(y)
    return loss_sum / n, correct / n

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["feature", "partial", "scratch"])
    ap.add_argument("--data", default="data"); ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--bs", type=int, default=16); ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    tr_tf = T.Compose([T.RandomResizedCrop(224, scale=(0.6, 1.0)), T.RandomHorizontalFlip(),
                       T.ColorJitter(0.3, 0.3, 0.3), T.ToTensor(), T.Normalize(MEAN, STD)])
    va_tf = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize(MEAN, STD)])
    tr = datasets.ImageFolder(f"{a.data}/train", tr_tf); va = datasets.ImageFolder(f"{a.data}/val", va_tf)
    tl = DataLoader(tr, a.bs, shuffle=True); vl = DataLoader(va, a.bs)

    m = build(a.mode, len(tr.classes)).to(dev); opt = optimizer(m, a.mode)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    hist = {k: [] for k in ["train_loss", "train_acc", "val_loss", "val_acc"]}
    best, ep90, t0 = 0.0, None, time.time()
    Path("results").mkdir(exist_ok=True)
    for ep in range(1, a.epochs + 1):
        trl, tra = run_epoch(m, tl, dev, opt, a.mode); val, vaa = run_epoch(m, vl, dev)
        sched.step()
        for k, v in zip(hist, [trl, tra, val, vaa]): hist[k].append(v)
        if vaa > best:
            best = vaa; torch.save(m.state_dict(), f"results/{a.mode}_best.pt")
        if ep90 is None and vaa >= 0.9: ep90 = ep
        print(f"[{a.mode}] ep {ep:02d} train {tra:.3f} | val {vaa:.3f} (loss {val:.3f})")
    res = dict(mode=a.mode, model="mobilenet_v3_small", classes=tr.classes, best_val_acc=best,
               epoch_acc90=ep90, train_time_s=round(time.time() - t0, 1), history=hist)
    Path(f"results/{a.mode}.json").write_text(json.dumps(res, indent=2))

if __name__ == "__main__":
    main()
