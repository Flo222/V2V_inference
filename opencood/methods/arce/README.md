# ARCE method layer

This directory contains **method-specific decision logic**. It answers:

> Given the current payload, channel context, importance and feedback, **how should we communicate this frame/link?**

It may choose quantization mode, redundancy/FEC level, sending decision, priority order and recovery policy. The implementations of INT4/INT8, byte packetization, RaptorQ/XOR, Markov channel behavior and generic recovery operators live in `opencood.communication`.

The current validated executors (`ARCEFixedComm`, `ARCEC2MABComm`) are retained to avoid changing numerical experiment semantics during this structural refactor. New refactoring should progressively delegate mechanics to `CommunicationPipeline` rather than re-implement them here.
