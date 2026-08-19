# ARCE module layout

- `controller.py`: public facade for ARCE executors and policies.
- `executors/`: fixed and C2MAB communication execution.
- `policy/`: fixed/random policies and C2MAB action, UCB, proposal and bank logic.
- `context/`, `reward/`, `cost/`, `runtime/`: independent decision-support layers.
- `transport_policy/`: payload transport actions and priority FEC scheduling.
- `audit/`: compression and FEC recovery auditors.

`common.py` contains shared ARCE/C2MAB helpers.  Import implementations from
the role-specific packages above; use `controller.py` for the public facade.
