#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml


def record(frame, state, action_id, quant, rho, cache, bw, plr, delay, source, parity, tx_source, tx_parity, fec, missing):
    return {
        "frame_id": frame,
        "ego_id": "0",
        "sender_id": "1",
        "agent_index": 1,
        "channel_state": state,
        "action_id": action_id,
        "action": {
            "send": 1,
            "quant_mode": quant,
            "redundancy_ratio": rho,
            "cache_enabled": cache,
            "fec_type": "raptor_sim" if rho > 0 else "none",
        },
        "pdf_action": {
            "send": 1,
            "quant_mode": quant,
            "redundancy_ratio": rho,
            "cache_enabled": cache,
            "fec_type": "raptor_sim" if rho > 0 else "none",
        },
        "dc2mab": {
            "selected": True,
            "proposal": {
                "action_id": action_id,
                "ucb": 0.7,
                "mean": 0.5,
                "bonus": 0.2,
                "channel_state": state,
                "complementarity": 0.3,
            },
            "selection_score": {"ratio": 0.001},
        },
        "channel": {"profile": {"bandwidth_mbps": bw, "plr": plr, "loss_rate": plr, "delay_ms": delay}},
        "system_budget": {"allocated_budget_bytes": 4096.0},
        "packet": {
            "num_source_packets": source,
            "num_parity_packets": parity,
            "num_transmitted_source_packets": tx_source,
            "num_transmitted_parity_packets": tx_parity,
            "num_source_dropped_by_budget": source - tx_source,
            "num_parity_dropped_by_budget": parity - tx_parity,
            "num_direct_received_source_packets": max(tx_source - 1, 0),
            "num_fec_recovered_source_packets": fec,
            "num_missing_source_packets": missing,
        },
        "quality": {"q_recv": (source - missing) / source},
        "size": {"actual_transmitted_bytes": float((tx_source + tx_parity) * 1024), "actual_received_bytes": float((tx_source + tx_parity - 1) * 1024)},
        "tx_bytes": float((tx_source + tx_parity) * 1024),
        "rx_bytes": float((tx_source + tx_parity - 1) * 1024),
    }


def eval_stat(tp, fp, gt, score):
    return {str(iou): {"tp": [tp], "fp": [fp], "gt": gt, "score": [score]} for iou in (0.3, 0.5, 0.7)}


def main():
    repo = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        source = root / "source"
        runtime = root / "runtime"
        out = root / "out"
        source.mkdir()
        out.mkdir()
        cfg = {
            "validate_dir": "/tmp/old",
            "wild_setting": {},
            "model": {"args": {
                "where2comm_fusion": {"communication": {"threshold": 0.01}},
                "arce": {
                    "enabled": True,
                    "mode": "dc2mab",
                    "action_space": {"quant": ["fp16", "int8", "int4"], "rho": [0.0, 0.25, 0.5]},
                    # Mirror the saved model's legacy context keys.  The
                    # preparation step must expose their current aliases.
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
                        "stale_norm_ms": 100.0,
                    },
                    "recovery": {
                        "temporal_cache": True,
                        "spatial_interpolation": True,
                        "zero_fill": True,
                        "temporal_fusion": {"enabled": True, "tau_stale_ms": 300.0},
                    },
                },
            }},
        }
        (source / "config.yaml").write_text(yaml.safe_dump(cfg))
        (source / "net_epoch20.pth").write_bytes(b"dummy")
        test_dir = root / "test"
        test_dir.mkdir()
        subprocess.run([
            sys.executable, str(repo / "scripts" / "prepare_final_markov_c2mab_model.py"),
            "--source-model-dir", str(source),
            "--runtime-model-dir", str(runtime),
            "--test-dir", str(test_dir),
            "--seed", "2026",
        ], check=True, stdout=subprocess.DEVNULL)
        assert (runtime / "net_epoch20.pth").is_symlink()
        prepared = yaml.safe_load((runtime / "config.yaml").read_text())
        arce = prepared["model"]["args"]["arce"]
        assert prepared["validate_dir"] == str(test_dir.resolve())
        assert arce["channel"]["profiles"]["good"]["bandwidth_mbps"] == 27.0
        assert arce["action_space"]["rho"] == [0.0, 0.10, 0.25, 0.60]
        assert arce["action_space"]["online_redundancy_ratios"] == [0.0, 0.10, 0.25, 0.60]
        assert arce["action_space"]["online_quant_modes"] == ["fp16", "int8", "int4"]
        assert arce["context"]["include_cav_confidence"] is False
        assert arce["context"]["b_max_mbps"] == 27.0
        assert arce["context"]["stale_max_ms"] == 400.0
        assert arce["recovery"] == "temporal_cache"
        assert arce["recovery_config"]["temporal_cache"] is True
        assert arce["recovery_config"]["spatial_interpolation"] is True
        assert arce["reward"]["mode"] == "ap_delta_cost"
        assert arce["reward"]["lambda_delta"] == 3.0
        assert arce["reward"]["lambda_abs"] == 0.1
        assert arce["reward"]["lambda_cost"] == 0.1
        assert not any(k.startswith("alpha_") for k in arce["reward"])
        manifest = json.loads((runtime / "final_markov_manifest.json").read_text())
        assert manifest["reward_normalization"]["changed"] is True
        assert manifest["reward_normalization"]["original"]["alpha_q"] == 0.5

        rows = [
            {"frame_index": 0, "record": record(0, "medium", "send1_int8_rho0_cache0_none", "int8", 0.0, 0, 5.0, 0.2, 50.0, 4, 0, 3, 0, 0, 2)},
            {"frame_index": 1, "record": record(1, "bad", "send1_int4_rho0p60_cache1_raptor_sim", "int4", 0.6, 1, 1.0, 0.35, 100.0, 4, 2, 3, 1, 1, 1)},
            {"frame_index": 2, "record": record(2, "good", "send1_fp16_rho0_cache0_none", "fp16", 0.0, 0, 27.0, 0.05, 10.0, 4, 0, 3, 0, 0, 2)},
        ]
        with (out / "runtime_records.jsonl").open("w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        frames = []
        for i, state in enumerate(("medium", "bad", "good")):
            rec = rows[i]["record"]
            frames.append({
                "frame_index": i,
                "frame_id": i,
                "tx_bytes": rec["tx_bytes"],
                "quality_mean_0357": 0.5 + i * 0.1,
                "mean_reward": 0.01 * i,
                "actions": [{
                    "action_id": rec["action_id"],
                    "channel_state": state,
                    "quant_mode": rec["action"]["quant_mode"],
                    "rho": rec["action"]["redundancy_ratio"],
                    "cache": rec["action"]["cache_enabled"],
                    "ucb_mean": 0.5,
                    "ucb_bonus": 0.2,
                }],
                "eval_stat": eval_stat(1, 0, 1, 0.9 - i * 0.1),
            })
        with (out / "online_trace.jsonl").open("w") as f:
            for frame in frames:
                f.write(json.dumps(frame) + "\n")
        (out / "final_summary.json").write_text(json.dumps({
            "AP@0.3-Markov": 1.0,
            "AP@0.5-Markov": 1.0,
            "AP@0.7-Markov": 1.0,
        }))
        subprocess.run([
            sys.executable, str(repo / "scripts" / "summarize_final_markov_c2mab_audit.py"),
            "--out-dir", str(out), "--warmup-frames", "1",
        ], check=True, stdout=subprocess.DEVNULL)
        summary = json.loads((out / "final_markov_c2mab_audit_summary.json").read_text())
        assert summary["checks"]["pass"] is True
        assert summary["overall"]["record_count"] == 3
        assert summary["overall"]["state_counter"] == {"medium": 1, "bad": 1, "good": 1}
        assert (out / "state_action_summary.csv").is_file()
        assert (out / "markov_transition.csv").is_file()
    print("Final Markov+C2MAB audit smoke test passed.")


if __name__ == "__main__":
    main()
