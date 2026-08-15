#!/usr/bin/env python
"""Static/unit checks for the unified GRACE evaluation entry points."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from opencood.tools.arce_reward_audit import save_reward_runtime_audit
from opencood.tools.build_arce_eval_summary import build_summary


AP_LINE = (
    "The Average Precision at IOU 0.3 is 0.701, "
    "The Average Precision at IOU 0.5 is 0.689, "
    "The Average Precision at IOU 0.7 is 0.501\n"
)


class EvalSummaryModeTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.out_dir = Path(self.temp_dir.name)
        (self.out_dir / "ap.log").write_text(AP_LINE, encoding="utf-8")
        (self.out_dir / "bw.json").write_text(
            json.dumps(
                {
                    "BW": 0.0123,
                    "total_tx_MB": 12.3,
                    "frame_count": 1000,
                    "record_count": 1500,
                    "transmitted_link_count": 1400,
                    "no_send_count": 100,
                    "int4_count": 500,
                    "packed_int4_count": 500,
                    "all_int4_packed": True,
                }
            ),
            encoding="utf-8",
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def _check_method(self, method):
        ap_only = build_summary(
            self.out_dir, method, "Markov", include_ap=True, include_bw=False
        )
        self.assertEqual(ap_only["evaluation_protocol"], "ap_only")
        self.assertEqual(ap_only["AP@0.3-Markov"], 0.701)
        self.assertIsNone(ap_only["BW-Markov"])

        bw_only = build_summary(
            self.out_dir, method, "Markov", include_ap=False, include_bw=True
        )
        self.assertEqual(bw_only["evaluation_protocol"], "bw_only")
        self.assertIsNone(bw_only["AP@0.3-Markov"])
        self.assertEqual(bw_only["BW-Markov"], 0.0123)

        both = build_summary(
            self.out_dir, method, "Markov", include_ap=True, include_bw=True
        )
        self.assertEqual(both["evaluation_protocol"], "separate_pass_ap_bw")
        self.assertEqual(both["AP@0.5-Markov"], 0.689)
        self.assertEqual(both["BW-Markov"], 0.0123)

    def test_arce_three_metric_modes(self):
        self._check_method("ARCE-C2MAB")

    def test_fixed_three_metric_modes(self):
        self._check_method("Where2Comm-ARCE-Fixed")

    def test_rejects_empty_metric_selection(self):
        with self.assertRaises(ValueError):
            build_summary(
                self.out_dir,
                "ARCE-C2MAB",
                "Markov",
                include_ap=False,
                include_bw=False,
            )


class RewardAuditTest(unittest.TestCase):
    def test_shared_reward_audit_export(self):
        records = [
            {
                "reward_update": {
                    "delta_confidence": 0.2,
                    "mean_reward": 0.1,
                    "num_updated": 1,
                    "num_send_updated": 1,
                    "num_no_send_updated": 0,
                }
            },
            {"not_a_reward_update": True},
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            result = save_reward_runtime_audit(
                records, Path(temp_dir), frame_count=2
            )
            audit = json.loads(
                Path(result["reward_runtime_audit_json"]).read_text(
                    encoding="utf-8"
                )
            )

        self.assertEqual(result["reward_update_count"], 1)
        self.assertEqual(audit["summary"]["mean_reward"]["mean"], 0.1)
        self.assertEqual(audit["summary"]["num_updated"]["pos"], 1)


class EntrypointWiringTest(unittest.TestCase):
    def test_single_runner_uses_summary_module_and_seed(self):
        script = (REPO_ROOT / "scripts" / "run_arce_single_eval.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("build_arce_eval_summary.py", script)
        self.assertIn('--include_ap "$RUN_AP"', script)
        self.assertIn('--include_bw "$RUN_BW"', script)
        self.assertIn('--seed "$SEED"', script)
        self.assertNotIn("ap_re = re.compile", script)

    def test_arce_runner_uses_static_preflight(self):
        script = (REPO_ROOT / "scripts" / "run_arce_c2mab.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("preflight_final_markov_c2mab_runtime.py", script)
        self.assertIn('RUN_PREFLIGHT="${RUN_PREFLIGHT:-1}"', script)
        self.assertNotIn("--build-model", script)

    def test_pair_runner_exposes_both_methods(self):
        script = (REPO_ROOT / "scripts" / "run_arce_pair_eval.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("run_arce()", script)
        self.assertIn("run_baseline()", script)
        self.assertIn("NUM_WORKERS", script)
        self.assertIn("SEED", script)


if __name__ == "__main__":
    unittest.main()
