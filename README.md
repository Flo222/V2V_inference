# V2V_inference

Modular cooperative-perception inference and communication experiments built on OpenCOOD.

## Final-clean layout

```text
opencood/
├── models/
│   ├── baselines/        # Where2Comm / V2X-ViT / CoSDH / RoCooper / CoopDiff
│   ├── common/           # shared backbone/submodule/fusion building blocks
│   ├── upstream/         # other retained OpenCOOD models
│   └── registry.py       # YAML core_method -> canonical model module
├── communication/        # shared communication mechanisms and channel background
│   ├── interface/
│   ├── transport/
│   │   ├── quantization/
│   │   ├── packetization/
│   │   ├── fec/
│   │   └── recovery/
│   ├── channel/
│   ├── pipeline/
│   └── metrics/
└── methods/
    └── arce/             # ARCE/C2MAB strategy, context, importance, action, reward
```

There is intentionally **no** `opencood/comm/` compatibility package and no
`opencood/compression/` package. Repository-owned imports use canonical paths.
Existing YAML `model.core_method` values are preserved through
`opencood/models/registry.py`, so current experiment YAML files do not need to
be renamed just because the Python files were reorganized.

## Responsibility boundary

- **Baseline (`models/baselines`)**: defines what the original method sends and how it fuses received information.
- **ARCE (`methods/arce`)**: decides how to communicate under the current context (importance, action, C2MAB, reward, redundancy policy).
- **Communication (`communication`)**: executes generic mechanisms such as quantization, byte packetization, FEC/RaptorQ, Markov loss/latency and recovery.

The intended data flow is:

```text
Baseline native feature/message
        ↓
NativePayload / NativeMessage
        ↓
ARCE controller / policy
        ↓ CommunicationAction
Shared communication transport
        ↓
Channel (ideal / Markov / bandwidth / loss / latency)
        ↓
Recovery
        ↓
Baseline-native fusion
        ↓
Detection / AP
```

## Active baselines

```text
opencood/models/baselines/
├── where2comm/
├── v2xvit/
├── cosdh/
├── rocooper/
└── coopdiff/
```

Baseline-specific legacy/native transport code that is still required to
reproduce previously validated experiments is kept inside that baseline's own
`transport/` directory instead of a global `models/comm_modules/` directory.

## Communication code lookup

| Function | Canonical path |
|---|---|
| FP32/FP16/INT8/INT4 quantization | `opencood/communication/transport/quantization/` |
| byte-stream packetization | `opencood/communication/transport/packetization/` |
| XOR / simulated FEC / RaptorQ | `opencood/communication/transport/fec/` |
| Markov state / packet loss / bandwidth / latency | `opencood/communication/channel/` |
| temporal cache / zero fill / interpolation | `opencood/communication/transport/recovery/` |
| communication statistics | `opencood/communication/metrics/` |
| ARCE/C2MAB action/context/reward | `opencood/methods/arce/` |
| priority-aware redundancy planning | `opencood/methods/arce/priority_block_fec_transport.py` |

## Environment

Reuse the existing validated `opencood` conda environment:

```bash
cd /home/server/v2x_projects/V2V_inference
conda activate opencood
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
bash scripts/setup_v2v_inference.sh
```

The setup script installs this source tree in editable mode without forcing
CUDA/spconv dependency upgrades, builds `box_overlaps` when needed, and runs the
structural smoke test.

## Existing checkpoints

Checkpoint files may remain in the old OPV2V log directories. The model registry
changes Python import locations only; it does not rename module attributes or
state-dict parameter keys.

For example, run the new code while pointing `--model_dir` at an existing log
folder.

## Verification

```bash
python scripts/verify_refactor.py
```

See `STRUCTURE.md`, `MIGRATION.md`, and `FINAL_CLEAN_NOTES.md` for details.
