from __future__ import annotations

import json
from typing import Any, List


TARGET_NO_SEND_ID = "send0_none_rho0_cache0_none"
ALLOWED_SEND_QUANTS = set(["fp16", "int8", "int4"])
ALLOWED_FEC_RHO = set([
    ("none", 0.0),
    ("raptor_sim", 0.10),
    ("raptor_sim", 0.25),
    ("raptor_sim", 0.60),
])


def _get(a: Any, key: str, default: Any = None) -> Any:
    if isinstance(a, dict):
        if key in a:
            return a.get(key)
        extra = a.get("extra", None)
        if isinstance(extra, dict) and key in extra:
            return extra.get(key)
        return default

    if hasattr(a, key):
        return getattr(a, key)

    if hasattr(a, "extra") and isinstance(getattr(a, "extra"), dict):
        extra = getattr(a, "extra")
        if key in extra:
            return extra.get(key)

    if hasattr(a, "to_dict"):
        d = a.to_dict()
        if key in d:
            return d.get(key)
        extra = d.get("extra", None)
        if isinstance(extra, dict) and key in extra:
            return extra.get(key)

    return default


def _as_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0


def _is_send(a: Any) -> bool:
    send = _get(a, "send", None)
    if send is None:
        send = _get(a, "send_flag", None)
    if send is None:
        send = _get(a, "send_enabled", 1)

    if isinstance(send, str):
        return send.strip().lower() not in ("0", "false", "no", "none", "null")
    return bool(send)


def _norm_quant(x: Any) -> str:
    if x is None:
        return "none"
    return str(x).strip().lower()


def _norm_fec(x: Any) -> str:
    if x is None:
        return "none"
    return str(x).strip().lower()


def _round_rho(x: Any) -> float:
    return round(_as_float(x), 2)


def _load_actions() -> List[Any]:
    from opencood.methods.arce.policies.action_space import build_pdf_action_space

    try:
        actions = build_pdf_action_space()
    except TypeError:
        actions = build_pdf_action_space({})

    if isinstance(actions, tuple):
        actions = actions[0]

    return list(actions)


def main() -> None:
    actions = _load_actions()
    illegal = []

    action_type = type(actions[0]).__name__ if actions else "none"
    if action_type == "dict":
        illegal.append(
            "action type changed to dict; expected PDFARCEAction-like object"
        )

    ids = [str(_get(a, "action_id", _get(a, "id", ""))) for a in actions]
    if len(ids) != len(set(ids)):
        illegal.append("duplicated action ids")

    no_send = []
    send_actions = []

    for a in actions:
        action_id = str(_get(a, "action_id", _get(a, "id", "")))
        send = _is_send(a)
        q = _norm_quant(_get(a, "quant_mode", _get(a, "quant", "none")))
        fec = _norm_fec(_get(a, "fec_type", "none"))
        rho = _round_rho(_get(a, "redundancy_ratio", _get(a, "rho", 0.0)))
        cache = int(_as_float(_get(a, "cache_enabled", _get(a, "cache", 0))))

        if not send:
            no_send.append(a)
            if action_id != TARGET_NO_SEND_ID:
                illegal.append("bad no-send id: %s" % action_id)
            if q != "none" or fec != "none" or rho != 0.0 or cache != 0:
                illegal.append("bad no-send fields: %s" % action_id)
            continue

        send_actions.append(a)

        if q == "fp32":
            illegal.append("online send action contains fp32: %s" % action_id)
        if q not in ALLOWED_SEND_QUANTS:
            illegal.append("bad send quant %s: %s" % (q, action_id))
        if (fec, rho) not in ALLOWED_FEC_RHO:
            illegal.append("illegal fec-rho pair %s %.2f: %s" % (fec, rho, action_id))

    send_quant_modes = sorted(set([
        _norm_quant(_get(a, "quant_mode", _get(a, "quant", "none")))
        for a in send_actions
    ]))
    rho_values = sorted(set([
        _round_rho(_get(a, "redundancy_ratio", _get(a, "rho", 0.0)))
        for a in send_actions
    ]))

    quant_quality_status = "missing"
    quant_quality_default = {}
    try:
        from opencood.methods.arce.policies.quant_quality import (
            DEFAULT_QUANT_QUALITY,
            get_quant_quality,
            get_quant_loss,
        )

        quant_quality_default = dict(DEFAULT_QUANT_QUALITY)
        expected = {
            "fp32": 1.0,
            "fp16": 0.99,
            "int8": 0.90,
            "int4": 0.65,
            "none": 1.0,
        }

        for k, v in expected.items():
            got = get_quant_quality(k)
            if abs(got - v) > 1e-9:
                illegal.append("quant quality %s expected %.3f got %.3f" % (k, v, got))

        if abs(get_quant_loss("int4") - 0.35) > 1e-9:
            illegal.append("quant loss int4 expected 0.35")

        unknown_quality = get_quant_quality("unknown_quant")
        if abs(unknown_quality - 1.0) < 1e-9:
            illegal.append("unknown quant defaults to perfect quality")

        quant_quality_status = "ok"
    except Exception as e:
        quant_quality_status = "error: %s" % str(e)

    result = {
        "action_type": action_type,
        "num_actions": len(actions),
        "num_no_send": len(no_send),
        "num_send": len(send_actions),
        "send_quant_modes": send_quant_modes,
        "rho_values": rho_values,
        "no_send_ids": [
            str(_get(a, "action_id", _get(a, "id", ""))) for a in no_send
        ],
        "illegal_actions": illegal,
        "quant_quality_status": quant_quality_status,
        "quant_quality_default": quant_quality_default,
        "status": "PASS"
        if (
            len(actions) == 25
            and len(no_send) == 1
            and len(send_actions) == 24
            and send_quant_modes == ["fp16", "int4", "int8"]
            and rho_values == [0.0, 0.1, 0.25, 0.6]
            and not illegal
        )
        else "FAIL",
    }

    print(json.dumps(result, indent=2, sort_keys=True))

    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
