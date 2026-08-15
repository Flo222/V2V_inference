"""Read-only audit for FEC recovery and joint compression/redundancy experiments.

The auditor compares, for the same quantized source payload:

    F_quant   : quantized then dequantized payload before channel
    F_direct  : directly received systematic source packets; missing=0
    F_fec     : payload after the configured FEC decoder; remaining missing=0

It also separates packet loss into:

    generated -> budget transmitted -> channel received -> FEC recovered

The auditor never writes back into the inference path.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from typing import Any, Dict, Optional

import torch


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "y", "on")
    return bool(value)


def _safe_name(value: Any) -> str:
    text = re.sub(r"[^0-9A-Za-z_.-]+", "_", str(value))
    return text[:160] if text else "unknown"


def _safe_ratio(num: float, den: float) -> float:
    return float(num / den) if float(den) > 0.0 else 0.0


def _pair_metrics(reference: torch.Tensor, candidate: torch.Tensor, eps: float = 1e-12) -> Dict[str, Any]:
    if not torch.is_tensor(reference) or not torch.is_tensor(candidate):
        return {"available": False, "reason": "not_tensor"}
    if tuple(reference.shape) != tuple(candidate.shape):
        return {
            "available": False,
            "reason": "shape_mismatch",
            "reference_shape": [int(v) for v in reference.shape],
            "candidate_shape": [int(v) for v in candidate.shape],
        }

    a_raw = reference.detach()
    b_raw = candidate.detach()
    a = a_raw.float().reshape(-1)
    b = b_raw.float().reshape(-1)
    if a.numel() == 0:
        return {
            "available": True,
            "mse": 0.0,
            "nmse": 0.0,
            "mae": 0.0,
            "max_abs_error": 0.0,
            "cosine_similarity": 1.0,
            "exact_equal": True,
            "allclose": True,
        }

    diff = a - b
    mse = torch.mean(diff * diff)
    energy = torch.mean(a * a).clamp_min(eps)
    denom = torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b)
    if float(denom.item()) <= eps:
        cosine = 1.0 if torch.equal(a_raw, b_raw) else 0.0
    else:
        cosine = float(torch.dot(a, b).div(denom).item())
    return {
        "available": True,
        "mse": float(mse.item()),
        "nmse": float((mse / energy).item()),
        "mae": float(diff.abs().mean().item()),
        "max_abs_error": float(diff.abs().max().item()),
        "cosine_similarity": float(cosine),
        "exact_equal": bool(torch.equal(a_raw, b_raw)),
        "allclose": bool(torch.allclose(a_raw, b_raw, rtol=1e-5, atol=1e-6)),
    }


def _mask_to_cpu_bool(mask: Any) -> torch.Tensor:
    if torch.is_tensor(mask):
        return mask.detach().to(device="cpu", dtype=torch.bool).flatten()
    return torch.as_tensor(mask, dtype=torch.bool).flatten().cpu()


def _mask_fingerprint(mask: Any) -> str:
    m = _mask_to_cpu_bool(mask)
    payload = bytes(m.to(dtype=torch.uint8).tolist())
    return hashlib.sha256(payload).hexdigest()


def _mask_ranges(mask: Any, true_value: bool = True) -> list:
    m = _mask_to_cpu_bool(mask)
    if not true_value:
        m = ~m
    indices = torch.nonzero(m, as_tuple=False).flatten().tolist()
    if not indices:
        return []
    ranges = []
    start = prev = int(indices[0])
    for idx in indices[1:]:
        idx = int(idx)
        if idx != prev + 1:
            ranges.append([start, prev + 1])
            start = idx
        prev = idx
    ranges.append([start, prev + 1])
    return ranges


class FECRecoveryAuditor:
    """Record budget/channel/FEC recovery without changing inference."""

    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        cfg = cfg or {}
        self.cfg = dict(cfg)
        self.enabled = _as_bool(cfg.get("enabled", False))
        self.strict = _as_bool(cfg.get("strict", False))
        self.experiment_name = str(cfg.get("experiment_name", "experiment3_pure_fec_recovery"))
        self.output_dir = os.path.abspath(os.path.expanduser(str(cfg.get("output_dir", "audit_runs/fec"))))
        self.file_name = str(cfg.get("file_name", "fec_recovery_audit.jsonl"))
        self.save_tensors = _as_bool(cfg.get("save_tensors", False))
        self.save_first_n_links = max(0, int(cfg.get("save_first_n_links", 0)))
        self.require_no_budget_drop = _as_bool(cfg.get("require_no_budget_drop", True))
        self.require_all_encoded_transmitted = _as_bool(cfg.get("require_all_encoded_transmitted", True))
        self.require_budget_not_exceeded = _as_bool(cfg.get("require_budget_not_exceeded", True))
        self._record_count = 0
        self._snapshot_count = 0
        self._jsonl_path = os.path.join(self.output_dir, self.file_name)
        self._snapshot_dir = os.path.join(self.output_dir, "tensor_snapshots")
        if self.enabled:
            os.makedirs(self.output_dir, exist_ok=True)
            if self.save_tensors:
                os.makedirs(self._snapshot_dir, exist_ok=True)
            with open(self._jsonl_path, "w", encoding="utf-8") as f:
                f.write("")

    def reset(self) -> None:
        self._record_count = 0
        self._snapshot_count = 0

    def _write(self, record: Dict[str, Any]) -> None:
        with open(self._jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            f.write("\n")

    def _save_snapshot(
        self,
        *,
        frame_id: Any,
        ego_index: int,
        agent_index: int,
        plr: float,
        rho: float,
        quant_dequantized: torch.Tensor,
        direct_recovered: torch.Tensor,
        fec_recovered: torch.Tensor,
        source_tx_mask: torch.Tensor,
        parity_tx_mask: torch.Tensor,
        source_receive_mask: torch.Tensor,
        parity_receive_mask: torch.Tensor,
    ) -> Optional[str]:
        if not self.save_tensors or self._snapshot_count >= self.save_first_n_links:
            return None
        name = "frame_%s_ego_%d_sender_%d_plr_%s_rho_%s_%04d.pt" % (
            _safe_name(frame_id), int(ego_index), int(agent_index),
            _safe_name("%.4f" % plr), _safe_name("%.4f" % rho), self._snapshot_count,
        )
        path = os.path.join(self._snapshot_dir, name)
        torch.save(
            {
                "frame_id": frame_id,
                "ego_index": int(ego_index),
                "agent_index": int(agent_index),
                "plr": float(plr),
                "rho": float(rho),
                "quantized_then_dequantized": quant_dequantized.detach().cpu().clone(),
                "direct_only_recovered": direct_recovered.detach().cpu().clone(),
                "fec_recovered": fec_recovered.detach().cpu().clone(),
                "source_tx_mask": source_tx_mask.detach().cpu().clone(),
                "parity_tx_mask": parity_tx_mask.detach().cpu().clone(),
                "source_receive_mask": source_receive_mask.detach().cpu().clone(),
                "parity_receive_mask": parity_receive_mask.detach().cpu().clone(),
            },
            path,
        )
        self._snapshot_count += 1
        return path

    def record(
        self,
        *,
        frame_id: Any,
        link_id: Any,
        ego_index: int,
        agent_index: int,
        quant_mode: str,
        fec_type: str,
        redundancy_ratio: float,
        plr: float,
        quant_dequantized: torch.Tensor,
        direct_recovered_compact: torch.Tensor,
        fec_recovered_compact: torch.Tensor,
        source_tx_mask: torch.Tensor,
        parity_tx_mask: torch.Tensor,
        source_receive_mask: torch.Tensor,
        parity_receive_mask: torch.Tensor,
        num_source_packets: int,
        num_parity_packets: int,
        num_encoded_packets: int,
        num_tx_source_packets: int,
        num_tx_parity_packets: int,
        num_source_dropped_by_budget: int,
        num_parity_dropped_by_budget: int,
        num_direct_received_source_packets: int,
        num_fec_recovered_source_packets: int,
        num_missing_source_packets: int,
        actual_transmitted_bytes: float,
        actual_received_bytes: float,
        packet_size_bytes: int,
        num_admitted_source_packets: Optional[int] = None,
        bandwidth_budget_bytes: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        if not self.enabled:
            return None
        try:
            source_tx_mask = _mask_to_cpu_bool(source_tx_mask)
            parity_tx_mask = _mask_to_cpu_bool(parity_tx_mask)
            source_receive_mask = _mask_to_cpu_bool(source_receive_mask)
            parity_receive_mask = _mask_to_cpu_bool(parity_receive_mask)

            k = int(num_source_packets)
            admitted_source = (
                int(num_admitted_source_packets)
                if num_admitted_source_packets is not None
                else int(num_tx_source_packets)
            )
            p = int(num_parity_packets)
            encoded = int(num_encoded_packets)
            tx_source = int(num_tx_source_packets)
            tx_parity = int(num_tx_parity_packets)
            source_budget_drop = int(num_source_dropped_by_budget)
            parity_budget_drop = int(num_parity_dropped_by_budget)
            direct = int(num_direct_received_source_packets)
            fec_rec = int(num_fec_recovered_source_packets)
            missing = int(num_missing_source_packets)
            final_recovered = int(direct + fec_rec)
            direct_missing = int(k - direct)
            packet_bytes = int(packet_size_bytes)

            direct_metrics = _pair_metrics(quant_dequantized, direct_recovered_compact)
            fec_metrics = _pair_metrics(quant_dequantized, fec_recovered_compact)
            direct_nmse = float(direct_metrics.get("nmse", math.nan))
            fec_nmse = float(fec_metrics.get("nmse", math.nan))
            nmse_reduction = direct_nmse - fec_nmse
            relative_nmse_reduction = nmse_reduction / max(abs(direct_nmse), 1e-12)

            source_channel_lost_mask = source_tx_mask & (~source_receive_mask)
            parity_channel_lost_mask = parity_tx_mask & (~parity_receive_mask)
            source_channel_lost = int(source_channel_lost_mask.sum().item())
            parity_channel_lost = int(parity_channel_lost_mask.sum().item())
            source_received = int(source_receive_mask.sum().item())
            parity_received = int(parity_receive_mask.sum().item())

            expected_tx_bytes = float((tx_source + tx_parity) * packet_bytes)
            expected_rx_bytes = float((source_received + parity_received) * packet_bytes)
            budget_bytes = None if bandwidth_budget_bytes is None else float(bandwidth_budget_bytes)
            budget_utilization = None if not budget_bytes or budget_bytes <= 0 else float(actual_transmitted_bytes) / budget_bytes

            sanity = {
                "source_packet_accounting_valid": bool(direct + fec_rec + missing == k),
                "encoded_packet_accounting_valid": bool(
                    k + p == encoded
                ),
                "admitted_source_count_valid": bool(
                    0 <= admitted_source <= k
                    and admitted_source == tx_source
                ),
                "source_budget_accounting_valid": bool(tx_source + source_budget_drop == k),
                "parity_budget_accounting_valid": bool(tx_parity + parity_budget_drop == p),
                "source_channel_accounting_valid": bool(source_received + source_channel_lost == tx_source),
                "parity_channel_accounting_valid": bool(parity_received + parity_channel_lost == tx_parity),
                "source_tx_mask_matches_count": bool(int(source_tx_mask.sum().item()) == tx_source),
                "parity_tx_mask_matches_count": bool(int(parity_tx_mask.sum().item()) == tx_parity),
                "source_receive_matches_direct_count": bool(source_received == direct),
                "actual_tx_bytes_matches_count": bool(abs(float(actual_transmitted_bytes) - expected_tx_bytes) <= 1e-6),
                "actual_rx_bytes_matches_count": bool(abs(float(actual_received_bytes) - expected_rx_bytes) <= 1e-6),
                "fec_never_reduces_recovery": bool(final_recovered >= direct),
                "fec_feature_not_worse_than_direct": bool(fec_nmse <= direct_nmse + 1e-10),
                "no_budget_drop": bool(source_budget_drop == 0 and parity_budget_drop == 0),
                "all_encoded_transmitted": bool(tx_source + tx_parity == encoded),
                "budget_not_exceeded": bool(budget_bytes is None or float(actual_transmitted_bytes) <= budget_bytes + 1e-6),
            }
            required = [
                sanity["source_packet_accounting_valid"],
                sanity["encoded_packet_accounting_valid"],
                sanity["admitted_source_count_valid"],
                sanity["source_budget_accounting_valid"],
                sanity["parity_budget_accounting_valid"],
                sanity["source_channel_accounting_valid"],
                sanity["parity_channel_accounting_valid"],
                sanity["source_tx_mask_matches_count"],
                sanity["parity_tx_mask_matches_count"],
                sanity["source_receive_matches_direct_count"],
                sanity["actual_tx_bytes_matches_count"],
                sanity["actual_rx_bytes_matches_count"],
                sanity["fec_never_reduces_recovery"],
                sanity["fec_feature_not_worse_than_direct"],
            ]
            if self.require_no_budget_drop:
                required.append(sanity["no_budget_drop"])
            if self.require_all_encoded_transmitted:
                required.append(sanity["all_encoded_transmitted"])
            if self.require_budget_not_exceeded:
                required.append(sanity["budget_not_exceeded"])
            sanity["passed"] = bool(all(required))

            snapshot_path = self._save_snapshot(
                frame_id=frame_id,
                ego_index=ego_index,
                agent_index=agent_index,
                plr=float(plr),
                rho=float(redundancy_ratio),
                quant_dequantized=quant_dequantized,
                direct_recovered=direct_recovered_compact,
                fec_recovered=fec_recovered_compact,
                source_tx_mask=source_tx_mask,
                parity_tx_mask=parity_tx_mask,
                source_receive_mask=source_receive_mask,
                parity_receive_mask=parity_receive_mask,
            )

            record = {
                "experiment": self.experiment_name,
                "frame_id": frame_id,
                "link_id": str(link_id),
                "ego_index": int(ego_index),
                "agent_index": int(agent_index),
                "quant_mode": str(quant_mode),
                "fec_type": str(fec_type),
                "redundancy_ratio": float(redundancy_ratio),
                "plr": float(plr),
                "budget": {
                    "bandwidth_budget_bytes": budget_bytes,
                    "actual_transmitted_bytes": float(actual_transmitted_bytes),
                    "actual_received_bytes": float(actual_received_bytes),
                    "actual_transmitted_source_bytes": float(tx_source * packet_bytes),
                    "actual_transmitted_parity_bytes": float(tx_parity * packet_bytes),
                    "budget_utilization": budget_utilization,
                },
                "packet": {
                    "packet_size_bytes": packet_bytes,
                    "num_source_packets": k,
                    "num_parity_packets": p,
                    "num_encoded_packets": encoded,
                    "num_transmitted_source_packets": tx_source,
                    "num_transmitted_parity_packets": tx_parity,
                    "num_transmitted_packets": int(tx_source + tx_parity),
                    "num_source_dropped_by_budget": source_budget_drop,
                    "num_parity_dropped_by_budget": parity_budget_drop,
                    "num_source_lost_by_channel": source_channel_lost,
                    "num_parity_lost_by_channel": parity_channel_lost,
                    "num_direct_received_source_packets": direct,
                    "num_received_parity_packets": parity_received,
                    "num_fec_recovered_source_packets": fec_rec,
                    "num_missing_source_packets": missing,
                    "num_direct_missing_source_packets": direct_missing,
                    "num_final_recovered_source_packets": final_recovered,
                    "source_tx_ratio": _safe_ratio(tx_source, k),
                    "parity_tx_ratio": _safe_ratio(tx_parity, p),
                    "encoded_tx_ratio": _safe_ratio(tx_source + tx_parity, encoded),
                    "source_budget_drop_ratio": _safe_ratio(source_budget_drop, k),
                    "parity_budget_drop_ratio": _safe_ratio(parity_budget_drop, p),
                    "source_channel_loss_ratio_of_transmitted": _safe_ratio(source_channel_lost, tx_source),
                    "parity_channel_loss_ratio_of_transmitted": _safe_ratio(parity_channel_lost, tx_parity),
                    "source_channel_loss_ratio_of_generated": _safe_ratio(source_channel_lost, k),
                    "parity_channel_loss_ratio_of_generated": _safe_ratio(parity_channel_lost, p),
                    "source_direct_recovery_ratio": _safe_ratio(direct, k),
                    "source_final_recovery_ratio": _safe_ratio(final_recovered, k),
                    "fec_recovery_fraction_of_direct_missing": _safe_ratio(fec_rec, direct_missing),
                    # Kept for Experiment-3 summary compatibility. Under finite
                    # budget this is explicitly loss among transmitted packets.
                    "source_empirical_loss_ratio": _safe_ratio(source_channel_lost, tx_source),
                    "parity_empirical_loss_ratio": _safe_ratio(parity_channel_lost, tx_parity),
                    "source_tx_fingerprint": _mask_fingerprint(source_tx_mask),
                    "source_loss_fingerprint": _mask_fingerprint(source_channel_lost_mask),
                    "source_receive_fingerprint": _mask_fingerprint(source_receive_mask),
                    "source_budget_dropped_ranges": _mask_ranges(~source_tx_mask, true_value=True),
                    "source_lost_ranges": _mask_ranges(source_channel_lost_mask, true_value=True),
                    "parity_lost_ranges": _mask_ranges(parity_channel_lost_mask, true_value=True),
                },
                # Backward-compatible alias used by the Experiment-3 summary.
                "bytes": {
                    "bandwidth_budget_bytes": budget_bytes,
                    "actual_transmitted_bytes": float(actual_transmitted_bytes),
                    "actual_received_bytes": float(actual_received_bytes),
                },
                "direct_feature_error": direct_metrics,
                "fec_feature_error": fec_metrics,
                "fec_gain": {
                    "nmse_reduction": float(nmse_reduction),
                    "relative_nmse_reduction": float(relative_nmse_reduction),
                    "cosine_gain": float(fec_metrics.get("cosine_similarity", math.nan) - direct_metrics.get("cosine_similarity", math.nan)),
                },
                "sanity": sanity,
                "snapshot_path": snapshot_path,
            }
            self._write(record)
            self._record_count += 1
            return {
                "source_tx_fingerprint": record["packet"]["source_tx_fingerprint"],
                "source_loss_fingerprint": record["packet"]["source_loss_fingerprint"],
                "num_fec_recovered_source_packets": fec_rec,
                "num_missing_source_packets": missing,
                "direct_nmse": direct_nmse,
                "fec_nmse": fec_nmse,
                "nmse_reduction": float(nmse_reduction),
                "sanity_passed": bool(sanity["passed"]),
            }
        except Exception as exc:
            error = {
                "experiment": self.experiment_name,
                "frame_id": frame_id,
                "link_id": str(link_id),
                "ego_index": int(ego_index),
                "agent_index": int(agent_index),
                "error": "%s: %s" % (type(exc).__name__, str(exc)),
            }
            try:
                self._write(error)
            except Exception:
                pass
            if self.strict:
                raise
            return {"error": error["error"], "sanity_passed": False}
