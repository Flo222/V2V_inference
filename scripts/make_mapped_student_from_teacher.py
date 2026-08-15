import torch
import yaml
import importlib

from opencood.models.registry import resolve_model_module
import shutil
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]

TEACHER_DIR = ROOT / "opencood/logs/point_pillar_diffteacher_opv2v_e20_2026_05_19_11_46_49"
TEACHER_CKPT = TEACHER_DIR / "net_epoch20.pth"
TEACHER_CFG = TEACHER_DIR / "config.yaml"

NEW_DIR = ROOT / "opencood/logs/point_pillar_diffstudent_opv2v_e30_mapped_from_teacher_e20"

print("TEACHER_CKPT =", TEACHER_CKPT)
print("TEACHER_CFG  =", TEACHER_CFG)
print("NEW_DIR      =", NEW_DIR)

assert TEACHER_CKPT.exists(), f"missing teacher ckpt: {TEACHER_CKPT}"
assert TEACHER_CFG.exists(), f"missing teacher config: {TEACHER_CFG}"

if NEW_DIR.exists():
    shutil.rmtree(NEW_DIR)
NEW_DIR.mkdir(parents=True, exist_ok=True)

cfg_path = NEW_DIR / "config.yaml"
shutil.copy2(TEACHER_CFG, cfg_path)

with open(cfg_path, "r") as f:
    cfg = yaml.load(f, Loader=yaml.Loader)

# -------------------------
# Build student config from teacher config
# -------------------------
cfg["name"] = "point_pillar_diffstudent_opv2v_e30_mapped_from_teacher_e20"
cfg["root_dir"] = "/data/opv2v/train"
cfg["validate_dir"] = "/data/opv2v/validate"

cfg.setdefault("model", {})
cfg["model"]["core_method"] = "point_pillar_diff_stu"

cfg.setdefault("train_params", {})
cfg["train_params"]["batch_size"] = 2
cfg["train_params"]["epoches"] = 30

# Paper setting: 2 cooperating vehicles
if "max_cav" in cfg["train_params"]:
    cfg["train_params"]["max_cav"] = 2

try:
    cfg["model"]["args"]["max_cav"] = 2
except Exception:
    pass

# Paper setting: 30 epochs, Adam, initial lr=1e-3
if "lr_scheduler" in cfg and isinstance(cfg["lr_scheduler"], dict):
    if "epoches" in cfg["lr_scheduler"]:
        cfg["lr_scheduler"]["epoches"] = 30
    if "warmup_epoches" in cfg["lr_scheduler"]:
        cfg["lr_scheduler"]["warmup_epoches"] = 10
    if "args" in cfg["lr_scheduler"] and isinstance(cfg["lr_scheduler"]["args"], dict):
        cfg["lr_scheduler"]["args"]["epoches"] = 30
        cfg["lr_scheduler"]["args"]["warmup_epoches"] = 10

if "optimizer" in cfg and isinstance(cfg["optimizer"], dict):
    if "lr" in cfg["optimizer"]:
        cfg["optimizer"]["lr"] = 1e-3
    if "args" in cfg["optimizer"] and isinstance(cfg["optimizer"]["args"], dict):
        if "lr" in cfg["optimizer"]["args"]:
            cfg["optimizer"]["args"]["lr"] = 1e-3

with open(cfg_path, "w") as f:
    yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

print("\n[OK] wrote student config:", cfg_path)
print("name:", cfg["name"])
print("core_method:", cfg["model"]["core_method"])
print("root_dir:", cfg["root_dir"])
print("validate_dir:", cfg["validate_dir"])
print("epoches:", cfg["train_params"]["epoches"])
print("batch_size:", cfg["train_params"]["batch_size"])

# -------------------------
# Instantiate student model
# -------------------------
module = importlib.import_module(resolve_model_module(cfg["model"]["core_method"]))
model = getattr(module, "PointPillarDiffStu")(cfg["model"]["args"])
student_state = model.state_dict()

# -------------------------
# Load teacher checkpoint
# -------------------------
teacher_state = torch.load(TEACHER_CKPT, map_location="cpu")
if isinstance(teacher_state, dict):
    for k in ["state_dict", "model_state_dict", "model"]:
        if k in teacher_state and isinstance(teacher_state[k], dict):
            teacher_state = teacher_state[k]
            break

# Start from full student state so saved ckpt covers all student keys
new_state = {}
for k, v in student_state.items():
    if torch.is_tensor(v):
        new_state[k] = v.detach().clone()
    else:
        new_state[k] = v

copied_exact = []
copied_mapped = []
shape_mismatch = []
missing_target = []
copied_keys = set()

def can_copy(src, dst):
    return torch.is_tensor(src) and torch.is_tensor(dst) and tuple(src.shape) == tuple(dst.shape)

# 1. Exact copy: teacher branch remains teacher branch
for src_key, src_val in teacher_state.items():
    if src_key not in new_state:
        continue

    if can_copy(src_val, new_state[src_key]):
        new_state[src_key] = src_val.detach().clone()
        copied_exact.append(src_key)
        copied_keys.add(src_key)
    elif torch.is_tensor(src_val) and torch.is_tensor(new_state[src_key]):
        shape_mismatch.append((src_key, src_key, tuple(src_val.shape), tuple(new_state[src_key].shape), "exact"))

# 2. teacher branch -> student branch
# Example: backbone_teacher.xxx -> backbone.xxx
for src_key, src_val in teacher_state.items():
    if "." not in src_key:
        continue

    top, rest = src_key.split(".", 1)
    if not top.endswith("_teacher"):
        continue

    dst_top = top[:-len("_teacher")]
    dst_key = dst_top + "." + rest

    if dst_key not in new_state:
        missing_target.append((src_key, dst_key))
        continue

    if can_copy(src_val, new_state[dst_key]):
        new_state[dst_key] = src_val.detach().clone()
        copied_mapped.append((src_key, dst_key))
        copied_keys.add(dst_key)
    elif torch.is_tensor(src_val) and torch.is_tensor(new_state[dst_key]):
        shape_mismatch.append((src_key, dst_key, tuple(src_val.shape), tuple(new_state[dst_key].shape), "teacher_to_student"))

out_ckpt = NEW_DIR / "net_epoch0.pth"
torch.save(new_state, out_ckpt)

# -------------------------
# Report
# -------------------------
model_keys = set(student_state.keys())
ckpt_keys = set(new_state.keys())

missing_keys = sorted(model_keys - ckpt_keys)
extra_keys = sorted(ckpt_keys - model_keys)

prefix_total = defaultdict(int)
prefix_copied = defaultdict(int)

for k in student_state.keys():
    top = k.split(".", 1)[0]
    prefix_total[top] += 1
    if k in copied_keys:
        prefix_copied[top] += 1

critical_pairs = [
    ("backbone_teacher.deblocks.0.0.weight", "backbone.deblocks.0.0.weight"),
    ("cls_head_teacher.weight", "cls_head.weight"),
    ("reg_head_teacher.weight", "reg_head.weight"),
    ("diffuser_teacher.down1.0.weight", "diffuser.down1.0.weight"),
    ("diffuser_teacher.conv_last.weight", "diffuser.conv_last.weight"),
    ("diffuser_teacher.time_emb.1.weight", "diffuser.time_emb.1.weight"),
]

lines = []
lines.append(f"teacher_ckpt: {TEACHER_CKPT}")
lines.append(f"teacher_cfg: {TEACHER_CFG}")
lines.append(f"student_cfg: {cfg_path}")
lines.append(f"out_ckpt: {out_ckpt}")
lines.append("")
lines.append(f"student model keys: {len(student_state)}")
lines.append(f"teacher checkpoint keys: {len(teacher_state)}")
lines.append(f"checkpoint keys saved: {len(new_state)}")
lines.append(f"missing keys: {len(missing_keys)}")
lines.append(f"extra keys: {len(extra_keys)}")
lines.append("")
lines.append(f"copied exact: {len(copied_exact)}")
lines.append(f"copied teacher_to_student: {len(copied_mapped)}")
lines.append(f"shape mismatches: {len(shape_mismatch)}")
lines.append(f"missing mapping targets: {len(missing_target)}")
lines.append("")
lines.append("Top-level copied coverage:")
for top in sorted(prefix_total.keys()):
    lines.append(f"  {top}: copied {prefix_copied[top]} / total {prefix_total[top]}")

lines.append("")
lines.append("Critical pair max_abs_diff:")
for a, b in critical_pairs:
    if a in new_state and b in new_state and torch.is_tensor(new_state[a]) and torch.is_tensor(new_state[b]):
        if tuple(new_state[a].shape) == tuple(new_state[b].shape):
            diff = (new_state[a] - new_state[b]).abs().max().item()
            lines.append(f"  {a} vs {b}: max_abs_diff = {diff}")
        else:
            lines.append(f"  {a} vs {b}: shape mismatch {tuple(new_state[a].shape)} vs {tuple(new_state[b].shape)}")
    else:
        lines.append(f"  missing pair: {a} vs {b}")

lines.append("")
lines.append("First 30 shape mismatches:")
for item in shape_mismatch[:30]:
    lines.append(f"  {item}")

lines.append("")
lines.append("First 30 missing mapping targets:")
for item in missing_target[:30]:
    lines.append(f"  {item}")

report_path = NEW_DIR / "checkpoint_mapping_report.txt"
report_path.write_text("\n".join(lines))

print("\n" + "\n".join(lines[:100]))

if missing_keys:
    print("\n[ERROR] checkpoint does not cover all student keys.")
    for k in missing_keys[:50]:
        print("  MISS", k)
    raise SystemExit(1)

print("\n[OK] saved config:", cfg_path)
print("[OK] saved ckpt:", out_ckpt)
print("[OK] saved report:", report_path)
print("[OK] checkpoint covers all student model keys.")
