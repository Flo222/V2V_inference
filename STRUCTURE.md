# V2V_inference — Final Clean Structure

## Dependency rule

- `opencood/models/baselines/`: baseline-native model/fusion/payload semantics — **what to send**.
- `opencood/methods/arce/`: ARCE context, importance, C2MAB/action/reward — **how to choose**.
- `opencood/communication/`: shared quantization, packetization, FEC, channel and recovery mechanisms — **how to execute**.

There is intentionally **no** `opencood/comm/` compatibility package and no
`opencood/compression/` package in the final-clean tree.

## Active baselines

```text
opencood/models/baselines/
├── where2comm/
├── v2xvit/
├── cosdh/
├── rocooper/
└── coopdiff/
```

Existing YAML `model.core_method` values are resolved by
`opencood/models/registry.py`; flat `models/point_pillar_<baseline>.py`
compatibility wrappers are therefore not needed.

## Shared communication

```text
opencood/communication/
├── interface/
├── transport/
│   ├── quantization/
│   ├── packetization/
│   ├── fec/
│   └── recovery/
├── channel/
├── pipeline/
└── metrics/
```

## ARCE

```text
opencood/methods/arce/
├── arce_fixed_comm.py
├── arce_c2mab_comm.py
├── controller.py
├── priority_block_fec_transport.py
├── policies/
└── audit/
```

ARCE may select quantization/FEC/recovery settings, but the generic mechanism
implementations remain under `opencood/communication/`.

## Model directory

```text
opencood/models/
├── baselines/   # five active experiment baselines
├── common/      # shared submodules and generic fusion blocks
├── upstream/    # other retained OpenCOOD models
└── registry.py  # YAML core_method -> canonical module resolver
```
