"""Late-message channel helper for CoSDH Markov evaluation.

The existing CosDHMarkovByteChannel damages CoSDH-selected intermediate
feature messages.  Intermediate-late CoSDH inference also sends non-ego CAV
single-agent dense detection maps to ego.  This helper applies the same
Markov byte-budget / packet-loss / delay logic to those late dense maps.
"""

from typing import Dict, Iterable, List, Tuple

import torch


LATE_TENSOR_KEYS: Tuple[str, ...] = (
    "psm",
    "rm",
    "dm",
    "cls_preds",
    "reg_preds",
    "dir_preds",
)


def _collect_late_tensors(output_dict: Dict) -> Tuple[List[str], List[torch.Tensor]]:
    """Collect dense prediction maps with compatible [1,C,H,W] shapes."""
    keys: List[str] = []
    tensors: List[torch.Tensor] = []
    H = W = None

    for key in LATE_TENSOR_KEYS:
        value = output_dict.get(key, None)
        if not torch.is_tensor(value):
            continue
        if value.dim() != 4 or value.shape[0] != 1:
            continue
        if H is None:
            H, W = value.shape[-2], value.shape[-1]
        if value.shape[-2:] != (H, W):
            continue
        keys.append(key)
        tensors.append(value)

    return keys, tensors


def apply_late_markov_to_output_dict(
    output_dict: Dict,
    channel,
    link_key: str,
    verbose_prefix: str = "CoSDH-Markov-Late",
) -> Dict:
    """Apply a CosDHMarkovByteChannel to non-ego late dense predictions.

    Parameters
    ----------
    output_dict:
        Model output for one non-ego CAV.  Expected keys are psm/rm/dm or
        cls_preds/reg_preds/dir_preds.
    channel:
        Existing CosDHMarkovByteChannel instance.
    link_key:
        Stable link id for the non-ego CAV, e.g. ``late_641``.
    verbose_prefix:
        Prefix printed when channel.verbose is enabled.

    Returns
    -------
    output_dict:
        The same dict, with dense maps replaced by channel-impaired maps.
    """
    if channel is None or not bool(getattr(channel, "enabled", False)):
        return output_dict

    keys, tensors = _collect_late_tensors(output_dict)
    if not tensors:
        return output_dict

    # Concatenate classification/regression/direction maps so the late
    # message is budgeted as one dense packet stream.
    cat = torch.cat(tensors, dim=1)  # [1, C_total, H, W]
    cur_msg = cat[0]

    # Late dense maps do not have CoSDH communication masks.  Treat all
    # non-zero spatial locations as transmitted cells.  For dense heads this
    # usually means nearly the whole map, which is exactly why the late branch
    # should be bandwidth-sensitive.
    spatial_mask = cur_msg.abs().sum(dim=0) > 0

    # ``inference_late_fusion`` starts a new late frame before iterating CAVs.
    # This fallback keeps the helper safe for direct calls.
    if not getattr(channel, "_frame_sessions", None):
        if hasattr(channel, "start_frame"):
            channel.start_frame()

    session = channel._get_or_create_session(link_key, cur_msg.device)

    # Use a reserved scale id to keep the delay cache separate from CoSDH's
    # multiscale intermediate feature cache.
    scale_idx = 999
    delayed_msg, delayed_mask = channel._select_delayed_message(
        link_key,
        scale_idx,
        cur_msg,
        spatial_mask,
        session["delay_slots"],
    )
    impaired_msg, stat = channel._apply_channel_to_cells(
        delayed_msg,
        delayed_mask,
        session,
        scale_idx=scale_idx,
        num_scales=1,
    )

    impaired_cat = impaired_msg.unsqueeze(0)
    split_sizes = [t.shape[1] for t in tensors]
    split_tensors = torch.split(impaired_cat, split_sizes, dim=1)
    for key, tensor in zip(keys, split_tensors):
        output_dict[key] = tensor

    info = {
        "link_key": link_key,
        "state": session["state"],
        "bandwidth_mbps": session["bandwidth_mbps"],
        "packet_loss_rate": session["packet_loss_rate"],
        "delay_slots": session["delay_slots"],
        "late_message": True,
    }
    info.update(stat)
    output_dict["late_comm_info"] = info

    if hasattr(channel, "latest_info"):
        channel.latest_info.append(info)

    if bool(getattr(channel, "verbose", False)):
        print(
            "[{}] link={} state={} bw={}Mbps plr={} delay={} "
            "cells recv/sent/selected={}/{}/{} budget_bytes {}/{}".format(
                verbose_prefix,
                link_key,
                session["state"],
                session["bandwidth_mbps"],
                session["packet_loss_rate"],
                session["delay_slots"],
                stat.get("received_units", 0),
                stat.get("sent_units", 0),
                stat.get("selected_cells", 0),
                stat.get("remaining_budget_bytes_after", 0),
                stat.get("initial_budget_bytes", 0),
            )
        )

    return output_dict
