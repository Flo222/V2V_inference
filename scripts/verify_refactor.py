#!/usr/bin/env python3
"""Structural smoke test for the final-clean V2V_inference layout."""
from importlib import import_module
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

REQUIRED = [
    "opencood.models.registry",
    "opencood.models.baselines.where2comm.point_pillar_where2comm",
    "opencood.models.baselines.v2xvit.point_pillar_transformer_opv2v",
    "opencood.models.baselines.rocooper.models.point_pillar_rocooper",
    "opencood.communication.interface",
    "opencood.communication.channel.channel_manager",
    "opencood.communication.transport.quantization.feature_quantizer",
    "opencood.communication.transport.packetization.byte_stream_packetizer",
    "opencood.communication.transport.fec.fec_base",
    "opencood.communication.transport.recovery.partial_reconstruction",
    "opencood.methods.arce.arce_fixed_comm",
    "opencood.methods.arce.policies.action_space",
]
ENV_OPTIONAL = [
    "opencood.models.baselines.cosdh.models.point_pillar_cosdh_markov",
    "opencood.models.baselines.coopdiff.models.point_pillar_diff_stu_markov",
]

failed = []
for name in REQUIRED:
    try:
        import_module(name)
        print(f"[OK]  {name}")
    except Exception as exc:
        failed.append((name, exc))
        print(f"[FAIL] {name}: {type(exc).__name__}: {exc}")

for name in ENV_OPTIONAL:
    try:
        import_module(name)
        print(f"[OK]  {name}")
    except Exception as exc:
        print(f"[ENV] {name}: {type(exc).__name__}: {exc}")

forbidden = [ROOT / "opencood" / "comm", ROOT / "opencood" / "compression"]
for path in forbidden:
    if path.exists():
        failed.append((str(path), RuntimeError("obsolete compatibility path still exists")))
        print(f"[FAIL] obsolete path exists: {path}")
    else:
        print(f"[OK]  removed obsolete path: {path.relative_to(ROOT)}")

if failed:
    print("Structural smoke test: FAIL")
    raise SystemExit(1)
print("Structural smoke test: PASS")
