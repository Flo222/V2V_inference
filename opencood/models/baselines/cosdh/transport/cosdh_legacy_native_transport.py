# -*- coding: utf-8 -*-
"""Transparent CoSDH legacy-native Ideal transport.

This transport preserves the checkpoint's original execution graph:

Intermediate:
    encoder -> FP16 bytes -> exact restore -> decoder

Late:
    non-ego dense detection heads -> original-dtype bytes -> exact restore
    -> original dataset.post_process

It performs no Markov damage, scheduling, sparsification, extra quantization,
redundancy, ARCE, or UCB.
"""

from __future__ import print_function

import copy

import numpy as np
import torch


class CosDHLegacyNativeTransport(object):
    """Exact byte round-trips for legacy-native CoSDH payload boundaries."""

    LATE_FIELDS = ("cls_preds", "reg_preds", "dir_preds")

    def __init__(self, cfg=None):
        self.cfg = copy.deepcopy(cfg) if isinstance(cfg, dict) else {}
        self.enabled = bool(self.cfg.get("enabled", False))
        self.mode = str(self.cfg.get("mode", "disabled")).lower()
        self.intermediate_enabled = bool(
            self.cfg.get("intermediate_enabled", True)
        )
        self.late_enabled = bool(self.cfg.get("late_enabled", False))
        self.late_payload_type = str(
            self.cfg.get("late_payload_type", "dense_heads")
        ).lower()
        self.require_exact_roundtrip = bool(
            self.cfg.get("require_exact_roundtrip", True)
        )
        self.latest_info = {}
        self.start_frame()

    def configure(self, cfg):
        """Update runtime switches without reconstructing the model."""
        if not isinstance(cfg, dict):
            cfg = {}
        self.cfg = copy.deepcopy(cfg)
        self.enabled = bool(cfg.get("enabled", False))
        self.mode = str(cfg.get("mode", "disabled")).lower()
        self.intermediate_enabled = bool(
            cfg.get("intermediate_enabled", True)
        )
        self.late_enabled = bool(cfg.get("late_enabled", False))
        self.late_payload_type = str(
            cfg.get("late_payload_type", "dense_heads")
        ).lower()
        self.require_exact_roundtrip = bool(
            cfg.get("require_exact_roundtrip", True)
        )
        self.start_frame()

    def start_frame(self, record_len=None, link_key_aliases=None):
        """Reset accounting once per ego inference frame."""
        self._scale_records = []
        self._late_records = []
        self._intermediate_source_bytes = 0
        self._intermediate_received_bytes = 0
        self._late_source_bytes = 0
        self._late_received_bytes = 0
        self._intermediate_roundtrip_calls = 0
        self._late_roundtrip_calls = 0
        self._record_len = self._normalize_record_len(record_len)
        self._link_key_aliases = copy.deepcopy(link_key_aliases)
        self.latest_info = self._build_latest_info()

    @staticmethod
    def _normalize_record_len(record_len):
        if record_len is None:
            return []
        if torch.is_tensor(record_len):
            return [int(v) for v in record_len.detach().cpu().reshape(-1)]
        return [int(v) for v in np.asarray(record_len).reshape(-1)]

    @staticmethod
    def _serialize_tensor(source):
        if not torch.is_tensor(source):
            raise TypeError("wire source must be a torch.Tensor")
        contiguous = source.detach().contiguous()
        try:
            array = contiguous.cpu().numpy()
        except TypeError as exc:
            raise TypeError(
                "Unsupported wire dtype {} for NumPy byte serialization"
                .format(contiguous.dtype)
            ) from exc
        payload = array.tobytes(order="C")
        return contiguous, array.dtype, payload

    @staticmethod
    def _deserialize_tensor(payload, shape, numpy_dtype, device, torch_dtype):
        restored_array = np.frombuffer(
            payload,
            dtype=numpy_dtype,
        ).copy()
        expected_values = int(np.prod(shape))
        if int(restored_array.size) != expected_values:
            raise RuntimeError(
                "Payload value count {} does not match shape {} "
                "({} values)".format(
                    int(restored_array.size), list(shape), expected_values
                )
            )
        restored = torch.from_numpy(restored_array).reshape(tuple(shape))
        return restored.to(device=device, dtype=torch_dtype)

    def _link_key(self, batch_idx, local_idx):
        aliases = self._link_key_aliases
        try:
            if isinstance(aliases, (list, tuple)):
                if aliases and isinstance(aliases[0], (list, tuple)):
                    return str(aliases[batch_idx][local_idx])
                return str(aliases[local_idx])
        except (IndexError, TypeError):
            pass
        return "b{}_cav{}".format(int(batch_idx), int(local_idx))

    def _build_latest_info(self):
        intermediate_equal = all(
            bool(item.get("bytes_equal", False))
            for item in self._scale_records
        ) if self._scale_records else True
        late_equal = all(
            bool(item.get("bytes_equal", False))
            for item in self._late_records
        ) if self._late_records else True
        total_source = (
            int(self._intermediate_source_bytes)
            + int(self._late_source_bytes)
        )
        total_received = (
            int(self._intermediate_received_bytes)
            + int(self._late_received_bytes)
        )
        executor = "legacy_native_ideal_byte_roundtrip"
        if self.late_enabled and self.late_payload_type == "candidate_records":
            executor = "legacy_native_candidate_ideal_byte_roundtrip"
        elif self.late_enabled:
            executor = "legacy_native_full_ideal_byte_roundtrip"
        return {
            "enabled": bool(self.enabled),
            "mode": str(self.mode),
            "executor_type": executor,
            "intermediate_enabled": bool(self.intermediate_enabled),
            "late_enabled": bool(self.late_enabled),
            "late_payload_type": str(self.late_payload_type),
            "roundtrip_calls_this_frame": int(
                self._intermediate_roundtrip_calls
                + self._late_roundtrip_calls
            ),
            "intermediate_roundtrip_calls": int(
                self._intermediate_roundtrip_calls
            ),
            "late_roundtrip_calls": int(self._late_roundtrip_calls),
            "scale_records": copy.deepcopy(self._scale_records),
            "late_records": copy.deepcopy(self._late_records),
            "intermediate_source_bytes": int(
                self._intermediate_source_bytes
            ),
            "intermediate_received_bytes": int(
                self._intermediate_received_bytes
            ),
            "late_source_bytes": int(self._late_source_bytes),
            "late_received_bytes": int(self._late_received_bytes),
            "source_bytes": int(total_source),
            "received_bytes": int(total_received),
            "intermediate_all_bytes_equal": bool(intermediate_equal),
            "late_all_bytes_equal": bool(late_equal),
            "all_bytes_equal": bool(intermediate_equal and late_equal),
            "record_len": list(self._record_len),
            "ucb_arce_used": False,
        }

    def roundtrip_intermediate(
        self,
        encoded,
        record_len,
        scale_idx,
        link_key_aliases=None,
    ):
        """Round-trip collaborator FP16 encoded rows; ego remains local."""
        if not self.enabled or not self.intermediate_enabled:
            return encoded
        if self.mode != "ideal":
            raise RuntimeError(
                "cosdh_legacy_native supports only mode=ideal here; got {!r}"
                .format(self.mode)
            )
        if not torch.is_tensor(encoded) or encoded.dim() != 4:
            raise TypeError(
                "encoded must be a 4-D torch.Tensor, got {}".format(
                    type(encoded)
                )
            )
        if encoded.dtype != torch.float16:
            raise TypeError(
                "Intermediate wire tensor must be FP16, got {}".format(
                    encoded.dtype
                )
            )

        lengths = self._normalize_record_len(record_len)
        if sum(lengths) != int(encoded.shape[0]):
            raise ValueError(
                "sum(record_len)={} does not match encoded batch {}".format(
                    sum(lengths), int(encoded.shape[0])
                )
            )
        if link_key_aliases is not None:
            self._link_key_aliases = copy.deepcopy(link_key_aliases)
        self._record_len = list(lengths)

        restored_groups = []
        offset = 0
        scale_source_bytes = 0
        scale_received_bytes = 0
        collaborator_records = []

        for batch_idx, cav_num in enumerate(lengths):
            group = encoded[offset:offset + cav_num]
            offset += cav_num
            if cav_num <= 0:
                raise ValueError("record_len contains non-positive value")

            restored_rows = [group[0:1]]
            for local_idx in range(1, cav_num):
                source, numpy_dtype, payload = self._serialize_tensor(
                    group[local_idx:local_idx + 1]
                )
                restored = self._deserialize_tensor(
                    payload,
                    source.shape,
                    numpy_dtype,
                    source.device,
                    source.dtype,
                )
                bytes_equal = bool(torch.equal(source, restored))
                if self.require_exact_roundtrip and not bytes_equal:
                    raise RuntimeError(
                        "Ideal Intermediate round-trip changed scale {} link {}"
                        .format(
                            int(scale_idx),
                            self._link_key(batch_idx, local_idx),
                        )
                    )
                nbytes = len(payload)
                scale_source_bytes += nbytes
                scale_received_bytes += nbytes
                collaborator_records.append({
                    "batch_idx": int(batch_idx),
                    "local_idx": int(local_idx),
                    "link_key": self._link_key(batch_idx, local_idx),
                    "shape": [int(v) for v in source.shape],
                    "dtype": str(source.dtype).replace("torch.", ""),
                    "source_bytes": int(nbytes),
                    "received_bytes": int(nbytes),
                    "bytes_equal": bool(bytes_equal),
                })
                restored_rows.append(restored)
            restored_groups.append(torch.cat(restored_rows, dim=0))

        restored_encoded = torch.cat(restored_groups, dim=0)
        tensor_equal = bool(torch.equal(encoded, restored_encoded))
        if self.require_exact_roundtrip and not tensor_equal:
            raise RuntimeError(
                "Ideal restored encoded tensor differs at scale {}".format(
                    int(scale_idx)
                )
            )

        self._scale_records.append({
            "scale_idx": int(scale_idx),
            "shape": [int(v) for v in encoded.shape],
            "dtype": str(encoded.dtype).replace("torch.", ""),
            "collaborator_count": int(
                sum(max(0, n - 1) for n in lengths)
            ),
            "source_bytes": int(scale_source_bytes),
            "received_bytes": int(scale_received_bytes),
            "bytes_equal": bool(tensor_equal),
            "collaborators": collaborator_records,
        })
        self._intermediate_source_bytes += scale_source_bytes
        self._intermediate_received_bytes += scale_received_bytes
        self._intermediate_roundtrip_calls += 1
        self.latest_info = self._build_latest_info()
        return restored_encoded

    def roundtrip_late_output(self, output_dict, cav_id):
        """Round-trip one non-ego dense Late output without changing fields.

        Only cls_preds, reg_preds, and optional dir_preds cross this boundary.
        Confidence filtering, beta suppression, projection, and NMS remain in
        the original dataset.post_process implementation.
        """
        if not self.enabled or not self.late_enabled:
            return output_dict
        if self.late_payload_type == "candidate_records":
            return output_dict
        if self.mode != "ideal":
            raise RuntimeError(
                "cosdh_legacy_native supports only mode=ideal here; got {!r}"
                .format(self.mode)
            )
        if not isinstance(output_dict, dict):
            raise TypeError(
                "Late output must be a mapping, got {}".format(
                    type(output_dict)
                )
            )
        for required in ("cls_preds", "reg_preds"):
            if required not in output_dict or not torch.is_tensor(
                output_dict[required]
            ):
                raise KeyError(
                    "Late output for CAV {} is missing tensor field {}"
                    .format(cav_id, required)
                )

        restored_output = output_dict.__class__()
        field_records = []
        source_bytes = 0
        received_bytes = 0

        for key, value in output_dict.items():
            if key not in self.LATE_FIELDS or not torch.is_tensor(value):
                restored_output[key] = value
                continue

            source, numpy_dtype, payload = self._serialize_tensor(value)
            restored = self._deserialize_tensor(
                payload,
                source.shape,
                numpy_dtype,
                source.device,
                source.dtype,
            )
            bytes_equal = bool(torch.equal(source, restored))
            if self.require_exact_roundtrip and not bytes_equal:
                raise RuntimeError(
                    "Ideal Late round-trip changed CAV {} field {}".format(
                        cav_id, key
                    )
                )
            nbytes = len(payload)
            source_bytes += nbytes
            received_bytes += nbytes
            field_records.append({
                "field": str(key),
                "shape": [int(v) for v in source.shape],
                "dtype": str(source.dtype).replace("torch.", ""),
                "source_bytes": int(nbytes),
                "received_bytes": int(nbytes),
                "bytes_equal": bool(bytes_equal),
            })
            restored_output[key] = restored

        if not field_records:
            raise RuntimeError(
                "No dense Late tensor fields were serialized for CAV {}"
                .format(cav_id)
            )
        record_equal = all(
            bool(item.get("bytes_equal", False)) for item in field_records
        )
        self._late_records.append({
            "cav_id": str(cav_id),
            "field_count": int(len(field_records)),
            "fields": field_records,
            "source_bytes": int(source_bytes),
            "received_bytes": int(received_bytes),
            "bytes_equal": bool(record_equal),
        })
        self._late_source_bytes += source_bytes
        self._late_received_bytes += received_bytes
        self._late_roundtrip_calls += 1
        self.latest_info = self._build_latest_info()
        return restored_output

    def roundtrip_late_candidates(self, boxes_local, scores, cav_id):
        """Round-trip confidence-filtered, pre-NMS Late candidates.

        Each candidate is represented by seven local-frame box parameters and
        one score.  Projection to ego coordinates and global NMS remain on the
        receiver side.  The two tensors preserve their original dtype exactly.
        """
        if not self.enabled or not self.late_enabled:
            return boxes_local, scores
        if self.late_payload_type != "candidate_records":
            return boxes_local, scores
        if self.mode != "ideal":
            raise RuntimeError(
                "candidate Late transport supports only mode=ideal here; "
                "got {!r}".format(self.mode)
            )
        if not torch.is_tensor(boxes_local) or not torch.is_tensor(scores):
            raise TypeError("Late candidates must be torch tensors")
        if boxes_local.dim() != 2 or int(boxes_local.shape[1]) != 7:
            raise ValueError(
                "boxes_local must have shape [N, 7], got {}".format(
                    list(boxes_local.shape)
                )
            )
        if scores.dim() != 1 or int(scores.shape[0]) != int(boxes_local.shape[0]):
            raise ValueError(
                "scores must have shape [N] matching boxes, got {} vs {}"
                .format(list(scores.shape), list(boxes_local.shape))
            )

        restored = []
        fields = []
        source_bytes = 0
        received_bytes = 0
        for name, tensor in (("boxes_local", boxes_local), ("scores", scores)):
            source, numpy_dtype, payload = self._serialize_tensor(tensor)
            value = self._deserialize_tensor(
                payload,
                source.shape,
                numpy_dtype,
                source.device,
                source.dtype,
            )
            equal = bool(torch.equal(source, value))
            if self.require_exact_roundtrip and not equal:
                raise RuntimeError(
                    "Ideal candidate round-trip changed CAV {} field {}"
                    .format(cav_id, name)
                )
            nbytes = len(payload)
            source_bytes += nbytes
            received_bytes += nbytes
            fields.append({
                "field": name,
                "shape": [int(v) for v in source.shape],
                "dtype": str(source.dtype).replace("torch.", ""),
                "source_bytes": int(nbytes),
                "received_bytes": int(nbytes),
                "bytes_equal": bool(equal),
            })
            restored.append(value)

        record_equal = all(item["bytes_equal"] for item in fields)
        self._late_records.append({
            "cav_id": str(cav_id),
            "payload_type": "candidate_records",
            "candidate_count": int(boxes_local.shape[0]),
            "bytes_per_candidate": 32 if boxes_local.dtype == torch.float32
                and scores.dtype == torch.float32 else None,
            "field_count": int(len(fields)),
            "fields": fields,
            "source_bytes": int(source_bytes),
            "received_bytes": int(received_bytes),
            "bytes_equal": bool(record_equal),
        })
        self._late_source_bytes += source_bytes
        self._late_received_bytes += received_bytes
        self._late_roundtrip_calls += 1
        self.latest_info = self._build_latest_info()
        return restored[0], restored[1]

