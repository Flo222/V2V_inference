from __future__ import annotations

from typing import Any, Dict, Iterable, Optional


MB = 1_000_000.0


def _as_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return float(default)
        return float(x)
    except Exception:
        return float(default)


def _as_bool_send_value(x: Any, default: bool = True) -> bool:
    if x is None:
        return bool(default)
    if isinstance(x, str):
        return x.strip().lower() not in ("0", "false", "no", "none", "null")
    return bool(x)


def _get(d: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k, default)
    return cur


def _is_comm_record(r: Dict[str, Any]) -> bool:
    if not isinstance(r, dict):
        return False
    return (
        "budget_consistency" in r
        or "packetization" in r
        or "byte_stream_packetization" in r
        or "actual_tx_bytes" in r
        or "tx_bytes" in r
        or r.get("no_send", False)
        or r.get("is_no_send", False)
    )


def _extract_tx_bytes(r: Dict[str, Any]) -> float:
    size = r.get("size", {}) or {}
    bc = r.get("budget_consistency", {}) or {}
    pkt = r.get("packetization", {}) or r.get("byte_stream_packetization", {}) or {}

    # Keep the same canonical priority as arce_bw_breakdown_utils.extract_tx_bytes.
    return _as_float(
        size.get(
            "actual_transmitted_bytes",
            size.get(
                "transmitted_bytes",
                size.get(
                    "tx_bytes",
                    r.get(
                        "actual_transmitted_bytes",
                        r.get(
                            "tx_bytes",
                            bc.get(
                                "actual_tx_bytes",
                                bc.get(
                                    "executor_actual_tx_bytes",
                                    pkt.get("actual_tx_bytes", pkt.get("transmitted_num_bytes", 0.0)),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        )
    )


def _extract_no_send(r: Dict[str, Any]) -> bool:
    act = r.get("pdf_action", None) or r.get("action", None) or {}
    action_send = _as_bool_send_value(_get(act, "send", default=1), default=True)

    return bool(
        r.get("no_send", False)
        or r.get("is_no_send", False)
        or _get(act, "is_no_send", default=False)
        or not action_send
    )


def _extract_quant_mode(r: Dict[str, Any]) -> str:
    act = r.get("pdf_action", None) or r.get("action", None) or {}
    return str(_get(act, "quant_mode", default=r.get("quant_mode", "unknown"))).strip().lower()


def _extract_source_tensor_kind(r: Dict[str, Any]) -> str:
    pkt = r.get("packetization", {}) or r.get("byte_stream_packetization", {}) or {}
    return str(
        pkt.get(
            "source_tensor_kind",
            pkt.get(
                "quantized_tensor_kind",
                r.get("source_tensor_kind", "unknown"),
            ),
        )
    ).strip().lower()


def summarize_bw_records(
    records: Iterable[Dict[str, Any]],
    method: str = "unknown",
    scenario: str = "unknown",
    num_frames: Optional[int] = None,
) -> Dict[str, Any]:
    total_tx_bytes = 0.0
    record_count = 0
    transmitted_link_count = 0
    no_send_count = 0
    frame_ids = set()

    int4_count = 0
    packed_int4_count = 0

    for r in records:
        if not _is_comm_record(r):
            continue

        record_count += 1

        frame_id = r.get("frame_id", r.get("sample_id", None))
        if frame_id is not None:
            frame_ids.add(frame_id)

        tx_bytes = _extract_tx_bytes(r)
        no_send = _extract_no_send(r)

        total_tx_bytes += tx_bytes

        if no_send:
            no_send_count += 1

        if (not no_send) and tx_bytes > 0.0:
            transmitted_link_count += 1

        quant_mode = _extract_quant_mode(r)
        source_tensor_kind = _extract_source_tensor_kind(r)

        if quant_mode == "int4":
            int4_count += 1
            if source_tensor_kind == "packed_int4":
                packed_int4_count += 1

    frame_count = int(num_frames) if num_frames is not None else len(frame_ids)
    frame_count = max(int(frame_count), 1)

    bw_mb_per_frame = total_tx_bytes / frame_count / MB

    return {
        "method": method,
        "scenario": scenario,
        "frame_count": int(frame_count),
        "record_count": int(record_count),
        "transmitted_link_count": int(transmitted_link_count),
        "no_send_count": int(no_send_count),
        "BW": float(bw_mb_per_frame),
        "bw_MB_per_frame": float(bw_mb_per_frame),
        "total_tx_MB": float(total_tx_bytes / MB),
        "int4_count": int(int4_count),
        "packed_int4_count": int(packed_int4_count),
        "all_int4_packed": (
            None if int4_count == 0 else bool(int4_count == packed_int4_count)
        ),
    }


def build_main_table_row(
    method: str,
    ap03_ideal: Optional[float] = None,
    ap03_markov: Optional[float] = None,
    ap05_ideal: Optional[float] = None,
    ap05_markov: Optional[float] = None,
    ap07_ideal: Optional[float] = None,
    ap07_markov: Optional[float] = None,
    bw_ideal: Optional[float] = None,
    bw_markov: Optional[float] = None,
    ndigits: int = 3,
) -> Dict[str, Any]:
    def _r(x: Optional[float]) -> Optional[float]:
        return None if x is None else round(float(x), ndigits)

    return {
        "Method": method,
        "ap0.3-Ideal": _r(ap03_ideal),
        "ap0.3-Markov": _r(ap03_markov),
        "ap0.5-Ideal": _r(ap05_ideal),
        "ap0.5-Markov": _r(ap05_markov),
        "ap0.7-Ideal": _r(ap07_ideal),
        "ap0.7-Markov": _r(ap07_markov),
        "BW-Ideal": _r(bw_ideal),
        "BW-Markov": _r(bw_markov),
    }
