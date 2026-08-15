# CoSDH

CoSDH is grouped into `models`, `fusion`, `components`, and `transport` because its original implementation contains multiple intermediate/late message forms and baseline-specific transport helpers. Those helpers are retained for faithful reproduction; new common-channel experiments should migrate their physical channel mechanics toward `opencood.communication` while preserving CoSDH's native payload semantics.
