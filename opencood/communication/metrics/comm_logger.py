"""
Communication logger for ARCE / OpenCOOD experiments.

This module writes communication records produced by:

    opencood.methods.arce.executors.fixed_executor.ARCEFixedComm

to disk.

Supported outputs:
    1. raw jsonl:
        Full nested ARCE records, one communication link per line.

    2. flat jsonl:
        Flattened standard communication metrics, one link per line.

    3. csv:
        Flat metric table for Excel / pandas plotting.

    4. summary json:
        Overall communication summary.

    5. full summary json:
        Overall + by-frame + by-channel + by-quant + by-FEC summary.

This module does NOT:
    - run ARCE communication;
    - run detection evaluation;
    - modify feature tensors;
    - require pandas.

Typical usage:

    from opencood.communication.metrics.comm_logger import CommLogger

    logger = CommLogger(log_dir)

    recovered_feature, record = arce_comm.communicate_feature(...)
    logger.log_record(record)

    logger.finalize()

Or after inference:

    logger = CommLogger(log_dir)
    logger.log_from_arce_comm(arce_comm)
    logger.finalize()
"""

from __future__ import annotations

import csv
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

from opencood.communication.metrics import (
    extract_record_metrics,
    flatten_dict,
)

from opencood.communication.metrics.comm_stats import (
    CommStats,
    extract_flat_comm_records,
    summarize_comm_records,
)


DEFAULT_RAW_JSONL_NAME = "arce_comm_records.jsonl"
DEFAULT_FLAT_JSONL_NAME = "arce_comm_flat.jsonl"
DEFAULT_FLAT_CSV_NAME = "arce_comm_flat.csv"
DEFAULT_SUMMARY_JSON_NAME = "arce_comm_summary.json"
DEFAULT_FULL_SUMMARY_JSON_NAME = "arce_comm_summary_full.json"
DEFAULT_CONFIG_JSON_NAME = "arce_comm_logger_config.json"


def _ensure_dir(path: Union[str, os.PathLike]) -> Path:
    """
    Create directory if it does not exist.

    Parameters
    ----------
    path : str or PathLike
        Directory path.

    Returns
    -------
    pathlib.Path
        Created / existing directory.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _is_number(value: Any) -> bool:
    """
    Check whether value is finite number-like.
    """
    if isinstance(value, bool):
        return False

    try:
        value = float(value)
    except (TypeError, ValueError):
        return False

    return math.isfinite(value)


def _json_safe(value: Any) -> Any:
    """
    Convert common objects into JSON-safe values.

    This handles:
        - dict
        - list / tuple
        - int / float / str / bool / None
        - numpy scalar / array if numpy is installed
        - torch tensor if torch is installed
        - pathlib.Path
        - other objects by str(value)

    Notes
    -----
    ARCE records should normally be tensor-free. This function is defensive
    in case a debug field accidentally contains tensor / numpy values.
    """
    if value is None:
        return None

    if isinstance(value, (str, int, bool)):
        return value

    if isinstance(value, float):
        return value if math.isfinite(value) else None

    if isinstance(value, Path):
        return str(value)

    try:
        import numpy as np

        if isinstance(value, np.generic):
            return _json_safe(value.item())

        if isinstance(value, np.ndarray):
            return _json_safe(value.tolist())
    except Exception:
        pass

    try:
        import torch

        if torch.is_tensor(value):
            if value.numel() == 1:
                return _json_safe(value.detach().cpu().item())
            return _json_safe(value.detach().cpu().tolist())
    except Exception:
        pass

    if isinstance(value, dict):
        return {
            str(_json_safe(k)): _json_safe(v)
            for k, v in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]

    if hasattr(value, "as_dict") and callable(value.as_dict):
        try:
            return _json_safe(value.as_dict())
        except Exception:
            pass

    return str(value)


def _write_json(path: Union[str, os.PathLike], data: Any, indent: int = 2) -> None:
    """
    Write JSON file.
    """
    path = Path(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(
            _json_safe(data),
            f,
            indent=indent,
            ensure_ascii=False,
        )


def _append_jsonl(path: Union[str, os.PathLike], data: Any) -> None:
    """
    Append one JSON object as one jsonl line.
    """
    path = Path(path)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(_json_safe(data), ensure_ascii=False))
        f.write("\n")


def _write_jsonl(path: Union[str, os.PathLike], records: Iterable[Any]) -> None:
    """
    Rewrite a jsonl file from records.
    """
    path = Path(path)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(_json_safe(record), ensure_ascii=False))
            f.write("\n")


def _collect_csv_fieldnames(records: Sequence[Dict[str, Any]]) -> List[str]:
    """
    Collect stable CSV fieldnames from flat records.

    The first record's key order is preserved, and newly appearing keys are
    appended later.
    """
    fieldnames: List[str] = []
    seen = set()

    for record in records:
        for key in record.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    return fieldnames


def _write_csv(path: Union[str, os.PathLike], records: Sequence[Dict[str, Any]]) -> None:
    """
    Write flat records to CSV.

    Complex values are JSON-encoded as strings.
    """
    path = Path(path)

    records = [_json_safe(record) for record in records]

    fieldnames = _collect_csv_fieldnames(records)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()

        for record in records:
            row = {}

            for key in fieldnames:
                value = record.get(key, "")

                if isinstance(value, (dict, list, tuple)):
                    row[key] = json.dumps(
                        _json_safe(value),
                        ensure_ascii=False,
                    )
                else:
                    row[key] = value

            writer.writerow(row)


class CommLogger:
    """
    ARCE communication record logger.

    Parameters
    ----------
    log_dir : str or PathLike
        Directory for communication log files.

    prefix : str
        Prefix for output files. If prefix="arce_comm", files are:
            arce_comm_records.jsonl
            arce_comm_flat.jsonl
            arce_comm_flat.csv
            arce_comm_summary.json
            arce_comm_summary_full.json

    reset_on_init : bool
        If True, existing output files are removed on initialization.

    keep_in_memory : bool
        If True, raw records and flat records are kept in memory.
        Needed for CSV and summary writing.

    write_raw_jsonl : bool
        Whether to write full nested records.

    write_flat_jsonl : bool
        Whether to write extracted flat metric records.

    write_csv : bool
        Whether to write flat CSV.

    write_summary : bool
        Whether to write summary JSON files.

    skip_bypassed : bool
        If True, bypassed records are skipped in statistics / flat outputs.

    flatten_raw_for_csv : bool
        If True, CSV contains flattened raw record fields plus standard metrics.
        If False, CSV contains only standard flat metrics.

    flush_every : int
        If >0, rewrite CSV and summary every N logged records.
        JSONL files are appended immediately.
    """

    def __init__(
        self,
        log_dir: Union[str, os.PathLike],
        prefix: str = "arce_comm",
        reset_on_init: bool = True,
        keep_in_memory: bool = True,
        write_raw_jsonl: bool = True,
        write_flat_jsonl: bool = True,
        write_csv: bool = True,
        write_summary: bool = True,
        skip_bypassed: bool = False,
        flatten_raw_for_csv: bool = False,
        csv_flatten_sep: str = ".",
        csv_flatten_max_depth: Optional[int] = None,
        flush_every: int = 0,
    ):
        self.log_dir = _ensure_dir(log_dir)
        self.prefix = str(prefix)

        self.reset_on_init = bool(reset_on_init)
        self.keep_in_memory = bool(keep_in_memory)

        self.write_raw_jsonl = bool(write_raw_jsonl)
        self.write_flat_jsonl = bool(write_flat_jsonl)
        self.write_csv = bool(write_csv)
        self.write_summary = bool(write_summary)

        self.skip_bypassed = bool(skip_bypassed)
        self.flatten_raw_for_csv = bool(flatten_raw_for_csv)
        self.csv_flatten_sep = str(csv_flatten_sep)
        self.csv_flatten_max_depth = csv_flatten_max_depth

        self.flush_every = int(flush_every)

        if self.flush_every < 0:
            raise ValueError(f"flush_every should be non-negative, got {flush_every}.")

        self.raw_jsonl_path = self.log_dir / f"{self.prefix}_records.jsonl"
        self.flat_jsonl_path = self.log_dir / f"{self.prefix}_flat.jsonl"
        self.csv_path = self.log_dir / f"{self.prefix}_flat.csv"
        self.summary_json_path = self.log_dir / f"{self.prefix}_summary.json"
        self.full_summary_json_path = self.log_dir / f"{self.prefix}_summary_full.json"
        self.config_json_path = self.log_dir / f"{self.prefix}_logger_config.json"

        self.records: List[Dict[str, Any]] = []
        self.flat_records: List[Dict[str, Any]] = []

        self.stats = CommStats(
            records=None,
            skip_bypassed=self.skip_bypassed,
            keep_records=False,
            keep_flat_metrics=False,
        )

        self.num_logged = 0
        self.num_skipped = 0
        self.created_time = time.strftime("%Y-%m-%d %H:%M:%S")

        if self.reset_on_init:
            self.reset_files()

        self.write_config()

    # ------------------------------------------------------------------
    # file helpers
    # ------------------------------------------------------------------

    def reset_files(self) -> None:
        """
        Remove existing output files generated by this logger.
        """
        for path in (
            self.raw_jsonl_path,
            self.flat_jsonl_path,
            self.csv_path,
            self.summary_json_path,
            self.full_summary_json_path,
            self.config_json_path,
        ):
            if path.exists():
                path.unlink()

    def write_config(self) -> None:
        """
        Write logger config JSON.
        """
        _write_json(
            self.config_json_path,
            self.get_config(),
            indent=2,
        )

    def get_config(self) -> Dict[str, Any]:
        """
        Return JSON-friendly logger config.
        """
        return {
            "log_dir": str(self.log_dir),
            "prefix": self.prefix,
            "created_time": self.created_time,
            "reset_on_init": bool(self.reset_on_init),
            "keep_in_memory": bool(self.keep_in_memory),
            "write_raw_jsonl": bool(self.write_raw_jsonl),
            "write_flat_jsonl": bool(self.write_flat_jsonl),
            "write_csv": bool(self.write_csv),
            "write_summary": bool(self.write_summary),
            "skip_bypassed": bool(self.skip_bypassed),
            "flatten_raw_for_csv": bool(self.flatten_raw_for_csv),
            "csv_flatten_sep": self.csv_flatten_sep,
            "csv_flatten_max_depth": self.csv_flatten_max_depth,
            "flush_every": int(self.flush_every),
            "paths": {
                "raw_jsonl": str(self.raw_jsonl_path),
                "flat_jsonl": str(self.flat_jsonl_path),
                "csv": str(self.csv_path),
                "summary_json": str(self.summary_json_path),
                "full_summary_json": str(self.full_summary_json_path),
                "config_json": str(self.config_json_path),
            },
        }

    # ------------------------------------------------------------------
    # record conversion
    # ------------------------------------------------------------------

    def record_to_flat(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert one raw ARCE record to flat metrics.
        """
        return _json_safe(extract_record_metrics(record))

    def record_to_csv_row(self, record: Dict[str, Any], flat_record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build one CSV row.

        If flatten_raw_for_csv=True, flatten raw nested fields and merge them
        with extracted standard metrics. Standard metrics take priority.
        """
        if not self.flatten_raw_for_csv:
            return _json_safe(flat_record)

        raw_flat = flatten_dict(
            _json_safe(record),
            sep=self.csv_flatten_sep,
            max_depth=self.csv_flatten_max_depth,
        )

        row = dict(raw_flat)
        row.update(_json_safe(flat_record))
        return row

    # ------------------------------------------------------------------
    # logging APIs
    # ------------------------------------------------------------------

    def log_record(self, record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Log one ARCE communication record.

        Parameters
        ----------
        record : dict
            Raw record generated by ARCEFixedComm.

        Returns
        -------
        dict or None
            Flat metric record, or None if skipped.
        """
        record = _json_safe(record or {})
        flat_record = self.record_to_flat(record)

        if self.skip_bypassed and bool(flat_record.get("bypassed", False)):
            self.num_skipped += 1
            return None

        self.num_logged += 1

        if self.keep_in_memory:
            self.records.append(record)
            self.flat_records.append(flat_record)

        self.stats.add_metrics(flat_record)

        if self.write_raw_jsonl:
            _append_jsonl(self.raw_jsonl_path, record)

        if self.write_flat_jsonl:
            _append_jsonl(self.flat_jsonl_path, flat_record)

        if self.flush_every > 0 and self.num_logged % self.flush_every == 0:
            self.flush()

        return flat_record

    def log_records(self, records: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Log multiple ARCE records.

        Returns
        -------
        list
            Flat metric records that were not skipped.
        """
        flat = []

        for record in records:
            item = self.log_record(record)
            if item is not None:
                flat.append(item)

        return flat

    def log_from_arce_comm(self, arce_comm: Any) -> List[Dict[str, Any]]:
        """
        Log all records from an ARCEFixedComm-like object.

        Expected method:
            arce_comm.get_records()
        """
        if not hasattr(arce_comm, "get_records"):
            raise AttributeError("arce_comm should have get_records() method.")

        return self.log_records(arce_comm.get_records())

    def log_result(self, result: Any) -> Optional[Dict[str, Any]]:
        """
        Log an ARCECommResult-like object.

        Expected field:
            result.record
        """
        if not hasattr(result, "record"):
            raise AttributeError("result should have record attribute.")

        return self.log_record(result.record)

    # ------------------------------------------------------------------
    # writing summaries / tables
    # ------------------------------------------------------------------

    def _get_csv_records(self) -> List[Dict[str, Any]]:
        """
        Return records to write into CSV.
        """
        if not self.keep_in_memory:
            return []

        if not self.flatten_raw_for_csv:
            return [_json_safe(item) for item in self.flat_records]

        rows = []

        for record, flat_record in zip(self.records, self.flat_records):
            rows.append(self.record_to_csv_row(record, flat_record))

        return rows

    def write_csv_file(self) -> Optional[Path]:
        """
        Write flat CSV file.

        Returns
        -------
        pathlib.Path or None
            CSV path, or None if CSV writing is disabled / impossible.
        """
        if not self.write_csv:
            return None

        rows = self._get_csv_records()

        if not rows:
            return None

        _write_csv(self.csv_path, rows)
        return self.csv_path

    def write_summary_files(self) -> Dict[str, Path]:
        """
        Write summary JSON files.

        Returns
        -------
        dict
            Written summary file paths.
        """
        written = {}

        if not self.write_summary:
            return written

        compact = self.stats.get_compact_summary()
        overall = self.stats.summarize()
        full = self.stats.full_summary().as_dict()

        summary = {
            "compact": compact,
            "overall": overall,
            "logger": {
                "num_logged": int(self.num_logged),
                "num_skipped": int(self.num_skipped),
                "log_dir": str(self.log_dir),
                "prefix": self.prefix,
            },
        }

        _write_json(self.summary_json_path, summary, indent=2)
        _write_json(self.full_summary_json_path, full, indent=2)

        written["summary_json"] = self.summary_json_path
        written["full_summary_json"] = self.full_summary_json_path

        return written

    def rewrite_jsonl_files(self) -> Dict[str, Path]:
        """
        Rewrite JSONL files from in-memory records.

        This is useful when records were modified or when JSONL writing was
        disabled during logging and should be produced at finalize time.
        """
        written = {}

        if not self.keep_in_memory:
            return written

        if self.write_raw_jsonl:
            _write_jsonl(self.raw_jsonl_path, self.records)
            written["raw_jsonl"] = self.raw_jsonl_path

        if self.write_flat_jsonl:
            _write_jsonl(self.flat_jsonl_path, self.flat_records)
            written["flat_jsonl"] = self.flat_jsonl_path

        return written

    def flush(self) -> Dict[str, Any]:
        """
        Flush CSV and summary files.

        JSONL records are appended immediately in log_record().
        If keep_in_memory=True, this also rewrites JSONL files to ensure
        consistency.

        Returns
        -------
        dict
            Flush status and output paths.
        """
        outputs: Dict[str, Any] = {
            "num_logged": int(self.num_logged),
            "num_skipped": int(self.num_skipped),
            "paths": {},
        }

        if self.keep_in_memory:
            outputs["paths"].update(
                {
                    key: str(path)
                    for key, path in self.rewrite_jsonl_files().items()
                }
            )

        csv_path = self.write_csv_file()
        if csv_path is not None:
            outputs["paths"]["csv"] = str(csv_path)

        summary_paths = self.write_summary_files()
        outputs["paths"].update(
            {
                key: str(path)
                for key, path in summary_paths.items()
            }
        )

        return outputs

    def finalize(self) -> Dict[str, Any]:
        """
        Finalize logger and write all outputs.

        Returns
        -------
        dict
            Output paths and compact summary.
        """
        outputs = self.flush()
        outputs["compact_summary"] = self.stats.get_compact_summary()
        outputs["config"] = self.get_config()
        return outputs

    close = finalize

    # ------------------------------------------------------------------
    # accessors
    # ------------------------------------------------------------------

    def get_records(self) -> List[Dict[str, Any]]:
        """
        Return stored raw records.
        """
        return list(self.records)

    def get_flat_records(self) -> List[Dict[str, Any]]:
        """
        Return stored flat metric records.
        """
        return list(self.flat_records)

    def get_summary(self) -> Dict[str, Any]:
        """
        Return compact summary.
        """
        return self.stats.get_compact_summary()

    def get_full_summary(self) -> Dict[str, Any]:
        """
        Return full summary.
        """
        return self.stats.full_summary().as_dict()

    def print_summary(self, prefix: str = "[ARCE CommLogger]") -> Dict[str, Any]:
        """
        Print compact summary and return it.
        """
        summary = self.get_summary()

        msg = (
            f"{prefix} "
            f"records={summary['num_records']} "
            f"non_bypassed={summary['num_non_bypassed_records']} "
            f"tx={summary['total_transmitted_mb']:.4f}MB "
            f"rx={summary['total_received_mb']:.4f}MB "
            f"loss={summary['overall_packet_loss_ratio']:.4f} "
            f"fec_rec={summary['total_fec_recovered_packets']} "
            f"tmp={summary['total_temporal_filled_packets']} "
            f"spatial={summary['total_spatial_filled_packets']} "
            f"zero={summary['total_zero_filled_packets']}"
        )

        print(msg)
        return summary

    def __len__(self) -> int:
        """
        Number of logged records.
        """
        return int(self.num_logged)

    def __repr__(self) -> str:
        return (
            "CommLogger("
            f"log_dir={str(self.log_dir)!r}, "
            f"prefix={self.prefix!r}, "
            f"num_logged={self.num_logged}, "
            f"num_skipped={self.num_skipped})"
        )


def save_comm_records(
    records: Iterable[Dict[str, Any]],
    log_dir: Union[str, os.PathLike],
    prefix: str = "arce_comm",
    skip_bypassed: bool = False,
    flatten_raw_for_csv: bool = False,
) -> Dict[str, Any]:
    """
    Convenience function to save communication records in one call.

    Parameters
    ----------
    records : iterable of dict
        ARCE records.

    log_dir : str or PathLike
        Output directory.

    prefix : str
        Output file prefix.

    skip_bypassed : bool
        Whether to skip bypassed records.

    flatten_raw_for_csv : bool
        Whether CSV should include flattened raw record fields.

    Returns
    -------
    dict
        Finalize output.
    """
    logger = CommLogger(
        log_dir=log_dir,
        prefix=prefix,
        reset_on_init=True,
        keep_in_memory=True,
        write_raw_jsonl=True,
        write_flat_jsonl=True,
        write_csv=True,
        write_summary=True,
        skip_bypassed=skip_bypassed,
        flatten_raw_for_csv=flatten_raw_for_csv,
    )

    logger.log_records(records)
    return logger.finalize()


def save_from_arce_comm(
    arce_comm: Any,
    log_dir: Union[str, os.PathLike],
    prefix: str = "arce_comm",
    skip_bypassed: bool = False,
    flatten_raw_for_csv: bool = False,
) -> Dict[str, Any]:
    """
    Convenience function to save records from ARCEFixedComm-like object.

    Expected method:
        arce_comm.get_records()
    """
    if not hasattr(arce_comm, "get_records"):
        raise AttributeError("arce_comm should have get_records() method.")

    return save_comm_records(
        records=arce_comm.get_records(),
        log_dir=log_dir,
        prefix=prefix,
        skip_bypassed=skip_bypassed,
        flatten_raw_for_csv=flatten_raw_for_csv,
    )


def load_jsonl(path: Union[str, os.PathLike]) -> List[Dict[str, Any]]:
    """
    Load JSONL records from disk.

    Parameters
    ----------
    path : str or PathLike
        JSONL file path.

    Returns
    -------
    list of dict
        Loaded records.
    """
    path = Path(path)

    records = []

    if not path.exists():
        return records

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            records.append(json.loads(line))

    return records


def load_comm_records(log_dir: Union[str, os.PathLike], prefix: str = "arce_comm") -> List[Dict[str, Any]]:
    """
    Load raw communication records from logger output directory.
    """
    path = Path(log_dir) / f"{prefix}_records.jsonl"
    return load_jsonl(path)


def load_flat_records(log_dir: Union[str, os.PathLike], prefix: str = "arce_comm") -> List[Dict[str, Any]]:
    """
    Load flat communication records from logger output directory.
    """
    path = Path(log_dir) / f"{prefix}_flat.jsonl"
    return load_jsonl(path)


__all__ = [
    "DEFAULT_RAW_JSONL_NAME",
    "DEFAULT_FLAT_JSONL_NAME",
    "DEFAULT_FLAT_CSV_NAME",
    "DEFAULT_SUMMARY_JSON_NAME",
    "DEFAULT_FULL_SUMMARY_JSON_NAME",
    "DEFAULT_CONFIG_JSON_NAME",
    "CommLogger",
    "save_comm_records",
    "save_from_arce_comm",
    "load_jsonl",
    "load_comm_records",
    "load_flat_records",
]