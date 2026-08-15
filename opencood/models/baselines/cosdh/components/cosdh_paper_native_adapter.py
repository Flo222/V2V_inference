from __future__ import print_function

import copy
import struct

import numpy as np
import torch

from opencood.models.baselines.cosdh.transport.cosdh_paper_native_byte_channel import \
    CosDHPaperNativeByteChannel


_LATE_KEY_GROUPS = (
    ("classification", ("psm", "cls_preds")),
    ("regression", ("rm", "reg_preds")),
    ("direction", ("dm", "dir_preds")),
)

_HEADER_STRUCT = struct.Struct("<4sBBBBIIIIII")
_HEADER_BYTES = _HEADER_STRUCT.size
_MAGIC = b"CSDH"
_VERSION = 1
_KIND_INTERMEDIATE = 1
_KIND_LATE = 2
_DTYPE_FP16 = 1
_DTYPE_FP32 = 2


def _record_len_list(record_len):
    if torch.is_tensor(record_len):
        return [
            int(value)
            for value in record_len.detach().cpu().reshape(-1).tolist()
        ]
    return [int(value) for value in record_len]


def _canonical_late_items(output_dict):
    items = []
    for canonical, aliases in _LATE_KEY_GROUPS:
        for key in aliases:
            value = output_dict.get(key, None)
            if torch.is_tensor(value):
                if value.dim() != 4 or int(value.shape[0]) != 1:
                    raise ValueError(
                        "CoSDH late tensor {} must be [1,C,H,W], got {}".format(
                            key, tuple(value.shape)
                        )
                    )
                items.append((canonical, key, value))
                break
    return items


def _pack_header(kind, dtype_code, scale_idx, c, h, w, units, record_bytes, payload_bytes):
    return _HEADER_STRUCT.pack(
        _MAGIC,
        _VERSION,
        int(kind),
        int(dtype_code),
        int(scale_idx) & 0xFF,
        int(c),
        int(h),
        int(w),
        int(units),
        int(record_bytes),
        int(payload_bytes),
    )


def _tensor_to_fp16_bytes(tensor):
    array = tensor.detach().cpu().numpy().astype(np.float16, copy=False)
    return np.frombuffer(array.tobytes(order="C"), dtype=np.uint8).copy()


def _tensor_to_fp32_bytes(tensor):
    array = tensor.detach().float().cpu().numpy().astype(np.float32, copy=False)
    return np.frombuffer(array.tobytes(order="C"), dtype=np.uint8).copy()


class CosDHPaperNativeFrameTransport(object):
    """Paper-native CoSDH serializer plus ideal/Markov byte channel.

    Public class/function names intentionally match the previous adapter so the
    already validated CoSDH model and inference hooks do not change.
    """

    def __init__(self, arce_cfg, paper_cfg, dataset_name):
        del arce_cfg  # No UCB/ARCE is used by this no-policy byte baseline.
        self.paper_cfg = copy.deepcopy(paper_cfg or {})
        self.dataset_name = str(dataset_name)
        self.enabled = bool(self.paper_cfg.get("enabled", False))
        self.identity_transport = bool(
            self.paper_cfg.get("identity_transport", False)
        )
        requested_mode = str(
            self.paper_cfg.get("channel_mode", "ideal")
        ).lower()
        if self.identity_transport:
            requested_mode = "ideal"

        byte_cfg = copy.deepcopy(
            self.paper_cfg.get("byte_channel", {}) or {}
        )
        byte_cfg["mode"] = requested_mode
        self.byte_channel = CosDHPaperNativeByteChannel(byte_cfg)
        self.executor = self.byte_channel
        self.executor_type = (
            "ideal_byte_stream"
            if requested_mode == "ideal"
            else "markov_fixed_byte_stream"
        )
        self.latest_info = {}
        self.nonzero_epsilon = float(
            self.paper_cfg.get("nonzero_epsilon", 0.0)
        )

    def _intermediate_segment(self, encoded, sender_idx, scale_idx):
        feature = encoded[sender_idx].detach().float()
        c, h, w = [int(value) for value in feature.shape]
        cell_mask = feature.abs().sum(dim=0) > self.nonzero_epsilon
        flat_indices = torch.nonzero(
            cell_mask.reshape(-1), as_tuple=False
        ).flatten()
        token_count = int(flat_indices.numel())
        record_bytes = int(4 + c * 2)

        records = np.zeros(token_count * record_bytes, dtype=np.uint8)
        for token_idx, flat_index in enumerate(
            flat_indices.detach().cpu().tolist()
        ):
            y = int(flat_index) // w
            x = int(flat_index) % w
            start = token_idx * record_bytes
            records[start:start + 4] = np.frombuffer(
                struct.pack("<HH", y, x), dtype=np.uint8
            )
            values = _tensor_to_fp16_bytes(feature[:, y, x])
            records[start + 4:start + record_bytes] = values

        header = np.frombuffer(
            _pack_header(
                _KIND_INTERMEDIATE,
                _DTYPE_FP16,
                scale_idx,
                c,
                h,
                w,
                token_count,
                record_bytes,
                int(records.shape[0]),
            ),
            dtype=np.uint8,
        ).copy()
        data = np.concatenate([header, records])
        return data, {
            "kind": "intermediate",
            "name": "scale{}".format(scale_idx),
            "scale_idx": int(scale_idx),
            "dtype": "float16",
            "shape": [c, h, w],
            "header_bytes": _HEADER_BYTES,
            "token_count": token_count,
            "record_bytes": record_bytes,
            "coordinate_bytes": int(token_count * 4),
            "value_bytes": int(token_count * c * 2),
            "byte_length": int(data.shape[0]),
        }

    def _late_segment(self, tensor, canonical, source_key):
        value = tensor[0].detach().float()
        c, h, w = [int(item) for item in value.shape]
        scalar_count = int(value.numel())
        payload = _tensor_to_fp32_bytes(value)
        header = np.frombuffer(
            _pack_header(
                _KIND_LATE,
                _DTYPE_FP32,
                255,
                c,
                h,
                w,
                scalar_count,
                4,
                int(payload.shape[0]),
            ),
            dtype=np.uint8,
        ).copy()
        data = np.concatenate([header, payload])
        return data, {
            "kind": "late",
            "name": str(canonical),
            "canonical": str(canonical),
            "source_key": str(source_key),
            "dtype": "float32",
            "shape": [c, h, w],
            "header_bytes": _HEADER_BYTES,
            "scalar_count": scalar_count,
            "record_bytes": 4,
            "coordinate_bytes": 0,
            "value_bytes": int(payload.shape[0]),
            "byte_length": int(data.shape[0]),
        }

    def _build_sender_frame(self, encoded_scales, late_output, sender_idx):
        chunks = []
        segments = []
        offset = 0

        for scale_idx, encoded in enumerate(encoded_scales):
            chunk, segment = self._intermediate_segment(
                encoded, sender_idx, scale_idx
            )
            segment["stream_offset"] = int(offset)
            chunks.append(chunk)
            segments.append(segment)
            offset += int(chunk.shape[0])

        for canonical, source_key, tensor in _canonical_late_items(late_output):
            chunk, segment = self._late_segment(
                tensor, canonical, source_key
            )
            segment["stream_offset"] = int(offset)
            chunks.append(chunk)
            segments.append(segment)
            offset += int(chunk.shape[0])

        stream = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.uint8)
        return {
            "stream": stream,
            "segments": segments,
            "sender_idx": int(sender_idx),
            "source_bytes": int(stream.shape[0]),
        }

    @staticmethod
    def _link_keys(data_dict, cav_num):
        cav_ids = data_dict.get("cav_id_list", None) if isinstance(data_dict, dict) else None
        if torch.is_tensor(cav_ids):
            cav_ids = cav_ids.detach().cpu().reshape(-1).tolist()
        keys = []
        for sender_idx in range(1, cav_num):
            if isinstance(cav_ids, (list, tuple)) and sender_idx < len(cav_ids):
                keys.append(str(cav_ids[sender_idx]))
            else:
                keys.append("b0_cav{}".format(sender_idx))
        return keys

    @staticmethod
    def _decode_intermediate(received, segment, device):
        c, h, w = [int(value) for value in segment["shape"]]
        output = torch.zeros((c, h, w), dtype=torch.float32, device=device)
        record_bytes = int(segment["record_bytes"])
        token_count = int(segment["token_count"])
        data_start = int(segment["stream_offset"]) + int(segment["header_bytes"])
        stream = received["stream"]
        valid = received["valid"]
        recovered_tokens = 0

        for token_idx in range(token_count):
            start = data_start + token_idx * record_bytes
            end = start + record_bytes
            if end > stream.shape[0] or not bool(valid[start:end].all()):
                continue
            y, x = struct.unpack("<HH", stream[start:start + 4].tobytes())
            if int(y) >= h or int(x) >= w:
                continue
            values = np.frombuffer(
                stream[start + 4:end].tobytes(), dtype=np.float16
            ).astype(np.float32)
            output[:, int(y), int(x)] = torch.from_numpy(
                values.copy()
            ).to(device=device)
            recovered_tokens += 1
        return output, recovered_tokens

    @staticmethod
    def _decode_late(received, segment, device):
        c, h, w = [int(value) for value in segment["shape"]]
        scalar_count = int(segment["scalar_count"])
        data_start = int(segment["stream_offset"]) + int(segment["header_bytes"])
        data_end = data_start + scalar_count * 4
        stream = received["stream"]
        valid = received["valid"]

        raw = np.zeros(scalar_count * 4, dtype=np.uint8)
        available_end = min(data_end, stream.shape[0])
        if available_end > data_start:
            raw[:available_end - data_start] = stream[data_start:available_end]
        recovered_scalars = 0
        for scalar_idx in range(scalar_count):
            start = data_start + scalar_idx * 4
            end = start + 4
            local_start = scalar_idx * 4
            local_end = local_start + 4
            if end <= valid.shape[0] and bool(valid[start:end].all()):
                recovered_scalars += 1
            else:
                raw[local_start:local_end] = 0

        values = np.frombuffer(raw.tobytes(), dtype=np.float32).copy()
        tensor = torch.from_numpy(values).to(device=device).reshape(1, c, h, w)
        return tensor, recovered_scalars

    def _deserialize(
        self,
        received_results,
        sender_frames,
        encoded_scales,
        late_outputs,
        cav_num,
    ):
        recovered_scales = [torch.zeros_like(value) for value in encoded_scales]
        for scale_idx, encoded in enumerate(encoded_scales):
            recovered_scales[scale_idx][0] = encoded[0]

        recovered_late = [dict(output) for output in late_outputs]
        decode_stats = []

        for result_idx, (received, frame) in enumerate(
            zip(received_results, sender_frames)
        ):
            sender_idx = result_idx + 1
            sender_stats = {
                "sender_idx": int(sender_idx),
                "segments": [],
            }
            for segment in frame["segments"]:
                if segment["kind"] == "intermediate":
                    scale_idx = int(segment["scale_idx"])
                    tensor, recovered_units = self._decode_intermediate(
                        received,
                        segment,
                        encoded_scales[scale_idx].device,
                    )
                    recovered_scales[scale_idx][sender_idx] = tensor
                    source_units = int(segment["token_count"])
                else:
                    tensor, recovered_units = self._decode_late(
                        received,
                        segment,
                        next(iter(recovered_late[sender_idx - 1].values())).device
                        if recovered_late[sender_idx - 1]
                        else encoded_scales[0].device,
                    )
                    output = recovered_late[sender_idx - 1]
                    source_key = str(segment["source_key"])
                    canonical = str(segment["canonical"])
                    output[source_key] = tensor
                    for group_name, aliases in _LATE_KEY_GROUPS:
                        if group_name == canonical:
                            for alias in aliases:
                                if alias in output:
                                    output[alias] = tensor
                            break
                    source_units = int(segment["scalar_count"])

                sender_stats["segments"].append(
                    {
                        "kind": str(segment["kind"]),
                        "name": str(segment["name"]),
                        "source_units": source_units,
                        "recovered_units": int(recovered_units),
                        "recovery_ratio": float(
                            recovered_units / float(max(source_units, 1))
                        ),
                    }
                )
            decode_stats.append(sender_stats)

        return recovered_scales, recovered_late, decode_stats

    @staticmethod
    def _roundtrip_exact(encoded_scales, recovered_scales, late_outputs, recovered_late):
        for original, recovered in zip(encoded_scales, recovered_scales):
            if not torch.equal(original.detach().float(), recovered.detach().float()):
                return False
        for original_output, recovered_output in zip(late_outputs, recovered_late):
            for canonical, source_key, tensor in _canonical_late_items(original_output):
                del canonical
                recovered = recovered_output.get(source_key, None)
                if not torch.is_tensor(recovered):
                    return False
                if not torch.equal(tensor.detach().float(), recovered.detach().float()):
                    return False
        return True

    def communicate_joint_frame(
        self,
        encoded_scales,
        encoded_scalar_masks,
        late_outputs,
        record_len,
        data_dict,
        local_cav_confidences=None,
        force_identity=False,
    ):
        del encoded_scalar_masks
        del local_cav_confidences
        lengths = _record_len_list(record_len)
        if len(lengths) != 1:
            raise ValueError(
                "CoSDH byte-stream inference currently requires batch_size=1"
            )
        cav_num = int(lengths[0])
        expected_collaborators = max(0, cav_num - 1)

        if len(late_outputs) != expected_collaborators:
            raise ValueError(
                "late output count {} does not match collaborators {}".format(
                    len(late_outputs), expected_collaborators
                )
            )

        sender_frames = [
            self._build_sender_frame(
                encoded_scales,
                late_outputs[sender_idx - 1],
                sender_idx,
            )
            for sender_idx in range(1, cav_num)
        ]
        link_keys = self._link_keys(data_dict or {}, cav_num)

        original_mode = self.byte_channel.mode
        if bool(force_identity) or bool(self.identity_transport) or not self.enabled:
            self.byte_channel.mode = "ideal"
        try:
            received_results, channel_infos = self.byte_channel.transmit_frame(
                sender_frames, link_keys
            )
        finally:
            self.byte_channel.mode = original_mode

        recovered_scales, recovered_late, decode_stats = self._deserialize(
            received_results,
            sender_frames,
            encoded_scales,
            late_outputs,
            cav_num,
        )
        effective_mode = (
            "ideal"
            if bool(force_identity) or bool(self.identity_transport) or not self.enabled
            else original_mode
        )
        exact = None
        if effective_mode == "ideal":
            exact = self._roundtrip_exact(
                encoded_scales,
                recovered_scales,
                late_outputs,
                recovered_late,
            )

        source_bytes = int(sum(frame["source_bytes"] for frame in sender_frames))
        sent_bytes = int(sum(info["sent_bytes_before_loss"] for info in channel_infos))
        received_bytes = int(sum(info["received_valid_bytes"] for info in channel_infos))
        coordinate_bytes = int(
            sum(
                segment.get("coordinate_bytes", 0)
                for frame in sender_frames
                for segment in frame["segments"]
            )
        )
        header_bytes = int(
            sum(
                segment.get("header_bytes", 0)
                for frame in sender_frames
                for segment in frame["segments"]
            )
        )
        intermediate_bytes = int(
            sum(
                segment.get("byte_length", 0)
                for frame in sender_frames
                for segment in frame["segments"]
                if segment["kind"] == "intermediate"
            )
        )
        late_bytes = int(
            sum(
                segment.get("byte_length", 0)
                for frame in sender_frames
                for segment in frame["segments"]
                if segment["kind"] == "late"
            )
        )

        info = {
            "enabled": effective_mode == "markov",
            "mode": effective_mode,
            "executor_type": (
                "ideal_byte_stream"
                if effective_mode == "ideal"
                else "markov_fixed_byte_stream"
            ),
            "joint_transport_calls_this_frame": 1,
            "share_intermediate_late_budget": True,
            "no_policy": True,
            "extra_quantization": "none",
            "extra_redundancy": "none",
            "extra_selection": "none",
            "segment_order": [
                "scale0_fp16_sparse",
                "scale1_fp16_sparse",
                "scale2_fp16_sparse",
                "late_classification_fp32_dense",
                "late_regression_fp32_dense",
                "late_direction_fp32_dense_if_present",
            ],
            "source_bytes": source_bytes,
            "sent_bytes_before_loss": sent_bytes,
            "received_valid_bytes": received_bytes,
            "intermediate_total_bytes": intermediate_bytes,
            "late_total_bytes": late_bytes,
            "coordinate_bytes": coordinate_bytes,
            "header_bytes": header_bytes,
            "ideal_roundtrip_exact": exact,
            "channel_links": channel_infos,
            "decode": decode_stats,
            "native_payload": {
                "interface": "cosdh_paper_native_byte_stream_v1",
                "payload_type": "cosdh_joint_mixed_dtype_byte_stream",
                "stage": (
                    "post_selection_encoder_fp16_and_"
                    "post_detector_pre_nms"
                ),
                "metadata": {
                    "dataset": self.dataset_name,
                    "paper_consistent": True,
                    "intermediate_scale_count": len(encoded_scales),
                    "late_segment_count": (
                        len(sender_frames[0]["segments"]) - len(encoded_scales)
                        if sender_frames else 0
                    ),
                    "intermediate_dtype": "float16",
                    "late_dtype": "float32",
                    "coordinates_in_byte_stream": True,
                    "headers_in_byte_stream": True,
                    "share_intermediate_late_budget": True,
                    "ucb_arce_used": False,
                },
            },
        }
        self.latest_info = copy.deepcopy(info)
        return recovered_scales, recovered_late, info


def run_cosdh_paper_native_ego(
    model,
    data_dict,
    spatial_features,
    psm_single,
    record_len,
    normalized_affine_matrix,
    req_mask,
):
    if not bool(getattr(model, "compression", False)):
        raise RuntimeError(
            "Paper-native CoSDH requires compression > 0"
        )

    feature_list = model.backbone.get_multiscale_feature(spatial_features)
    encoded_scales = []
    encoded_scalar_masks = []
    raw_scale_features = []
    communication_rates = []

    cfg = getattr(model, "cosdh_paper_native_cfg", {}) or {}
    fp16_wire = bool(cfg.get("fp16_wire", True))
    fp16_in_train = bool(cfg.get("fp16_in_train", False))
    apply_fp16 = fp16_wire and (not model.training or fp16_in_train)

    for scale_idx, fuse_module in enumerate(model.fusion_net):
        raw_feature = feature_list[scale_idx]
        encoded, scalar_mask, rate, _ = \
            fuse_module.prepare_paper_native_encoded(
                x=raw_feature,
                psm_single=psm_single,
                record_len=record_len,
                normalized_affine_matrix=normalized_affine_matrix,
                compressor=model.naive_compressor_list[scale_idx],
                req_mask=req_mask,
                fp16_wire=apply_fp16,
                nonzero_epsilon=float(cfg.get("nonzero_epsilon", 0.0)),
            )
        raw_scale_features.append(raw_feature)
        encoded_scales.append(encoded)
        encoded_scalar_masks.append(scalar_mask)
        communication_rates.append(float(rate))

    late_outputs = data_dict.get("_cosdh_paper_late_outputs", []) or []
    sanitized_data_dict = {
        key: value
        for key, value in data_dict.items()
        if not str(key).startswith("_cosdh_paper_")
    }
    force_identity = bool(model.training) and not bool(
        cfg.get("apply_transport_in_train", False)
    )

    recovered_scales, recovered_late, comm_info = \
        model.cosdh_paper_transport.communicate_joint_frame(
            encoded_scales=encoded_scales,
            encoded_scalar_masks=encoded_scalar_masks,
            late_outputs=late_outputs,
            record_len=record_len,
            data_dict=sanitized_data_dict,
            local_cav_confidences=None,
            force_identity=force_identity,
        )

    fused_feature_list = []
    for scale_idx, fuse_module in enumerate(model.fusion_net):
        fused = fuse_module.fuse_paper_native_received(
            local_raw_features=raw_scale_features[scale_idx],
            recovered_encoded=recovered_scales[scale_idx],
            record_len=record_len,
            normalized_affine_matrix=normalized_affine_matrix,
            compressor=model.naive_compressor_list[scale_idx],
        )
        fused_feature_list.append(fused)

    fused_feature = model.backbone.decode_multiscale_feature(
        fused_feature_list
    )
    if model.shrink_flag:
        fused_feature = model.shrink_conv(fused_feature)

    psm = model.cls_head(fused_feature)
    rm = model.reg_head(fused_feature)
    style = str(getattr(model, "cosdh_output_style", "opv2v"))
    if style == "v2xreal":
        output_dict = {"psm": psm, "rm": rm}
    else:
        output_dict = {"cls_preds": psm, "reg_preds": rm}

    if model.use_dir:
        output_dict["dir_preds"] = model.dir_head(fused_feature)

    output_dict["comm_info"] = {
        "paper_native_byte_stream": comm_info,
        "communication_rates": communication_rates,
    }
    output_dict["_cosdh_recovered_late_outputs"] = recovered_late
    model.latest_paper_native_info = copy.deepcopy(comm_info)
    return output_dict


__all__ = [
    "CosDHPaperNativeFrameTransport",
    "run_cosdh_paper_native_ego",
]
