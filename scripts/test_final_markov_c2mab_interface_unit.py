#!/usr/bin/env python3
from __future__ import annotations

import ast
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml


def online_eval_flags(path: Path):
    tree = ast.parse(path.read_text())
    flags = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute) or func.attr != "add_argument":
            continue
        for arg in node.args:
            if isinstance(arg, ast.Str) and arg.s.startswith("--"):
                flags.add(arg.s)
            elif hasattr(ast, "Constant") and isinstance(arg, ast.Constant):
                if isinstance(arg.value, str) and arg.value.startswith("--"):
                    flags.add(arg.value)
    return flags


def main():
    repo = Path(__file__).resolve().parents[1]
    prepare = repo / "scripts" / "prepare_final_markov_c2mab_model.py"
    preflight = repo / "scripts" / "preflight_final_markov_c2mab_runtime.py"
    runner = repo / "scripts" / "run_final_markov_c2mab_audit.sh"
    online = repo / "opencood" / "tools" / "arce_online_eval.py"

    runner_text = runner.read_text()
    online_segment = runner_text.split(
        "python opencood/tools/arce_online_eval.py", 1
    )[1].split("python scripts/summarize_final_markov_c2mab_audit.py", 1)[0]
    assert "--reward-profile" not in online_segment
    assert "--reward-profile" in runner_text.split(
        "python scripts/prepare_final_markov_c2mab_model.py", 1
    )[1].split("CHECKPOINT=", 1)[0]

    flags = online_eval_flags(online)
    required = {
        "--model_dir", "--out_dir", "--method", "--scenario",
        "--fusion_method", "--max_frames", "--num_workers",
        "--progress_interval", "--window_size", "--window_stride",
        "--warmup_frames", "--seed",
    }
    assert required.issubset(flags), (required - flags)
    assert "--reward-profile" not in flags

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        source = root / "source"
        runtime = root / "runtime"
        test_dir = root / "test"
        source.mkdir()
        test_dir.mkdir()
        cfg = {
            "validate_dir": "/tmp/old",
            "wild_setting": {},
            "model": {"args": {"arce": {
                "enabled": True,
                "mode": "dc2mab",
                "action_space": {
                    "type": "final_36",
                    "send": [0, 1],
                    "quant": ["fp16", "int8", "int4"],
                    "rho": [0.0, 0.25, 0.5],
                    "cache": [0, 1],
                },
                "context": {
                    "dim": 6,
                    "normalize_bandwidth_by_mbps": 27.0,
                    "normalize_delay_by_ms": 400.0,
                },
                "c2mab": {"context_dim": 6},
                "reward": {
                    "type": "final_proxy",
                    "alpha_q": 0.5,
                    "alpha_cost": 0.3,
                    "alpha_delay": 0.2,
                    "alpha_violation": 1.0,
                    "tau_stale_ms": 300.0,
                    "stale_norm_ms": 400.0,
                },
                "recovery": {
                    "temporal_cache": True,
                    "spatial_interpolation": True,
                    "zero_fill": True,
                },
                "scheduler": {
                    "per_link_budget": True,
                },
            }}},
        }
        (source / "config.yaml").write_text(yaml.safe_dump(cfg))
        (source / "net_epoch20.pth").write_bytes(b"dummy")
        subprocess.run([
            sys.executable, str(prepare),
            "--source-model-dir", str(source),
            "--runtime-model-dir", str(runtime),
            "--test-dir", str(test_dir),
            "--seed", "2026",
            "--reward-profile", "r2b",
        ], check=True, stdout=subprocess.DEVNULL)

        prepared = yaml.safe_load((runtime / "config.yaml").read_text())
        arce = prepared["model"]["args"]["arce"]
        action = arce["action_space"]
        assert action["online_quant_modes"] == action["quant"]
        assert action["online_redundancy_ratios"] == [0.0, 0.10, 0.25, 0.60]
        assert action["rho"] == [0.0, 0.10, 0.25, 0.60]
        assert action["send_values"] == action["send"]
        assert action["cache_values"] == action["cache"]
        assert arce["context"]["include_cav_confidence"] is False
        assert arce["context"]["b_max_mbps"] == 27.0
        assert arce["context"]["stale_max_ms"] == 400.0
        assert arce["scheduler"]["per_link_budget"] is True

        subprocess.run([
            sys.executable, str(preflight),
            "--model-dir", str(runtime),
        ], check=True, stdout=subprocess.DEVNULL)

        manifest = json.loads((runtime / "final_markov_manifest.json").read_text())
        assert manifest["action_space_normalization"]["changed"] is True
        rho_migration = manifest["action_space_normalization"]["rho_schema_migration"]
        assert rho_migration["changed"] is True
        assert rho_migration["expected_action_count"] == 25
        assert manifest["context_normalization"]["effective_context_dim"] == 6

    print("Final Markov+C2MAB interface compatibility test passed.")


if __name__ == "__main__":
    main()
