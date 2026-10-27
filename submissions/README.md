# Model submissions

| Model | Edition | Decision Index | Complete results | Engine / commit | Hardware | Declared limits |
|---|---|---:|---|---|---|---|
| [Sieve-31B-v3](https://huggingface.co/sthanika-ai/Sieve-31B-v3/tree/481c43fb8ab910e6c9ca83d229ae6a32a9f18f5c) | 0.3 | 71.68 | [scores.json](https://huggingface.co/datasets/sthanika-ai/Sieve-31B-v3-decision-index-results/blob/188e02fbf1c8f0a16f52f38236cd3f966a66f07c/runs/Sieve-31B-v3/scores.json) | [`sieve.decision_index_engine:SieveEngine`](https://github.com/sthanika-ai/Sieve/blob/0fb2c5fe13b948fd67ec013478dcda99cfc68091/sieve/decision_index_engine.py) @ `0fb2c5f`; kit `f5a6e4f` (0.3) | 8× NVIDIA RTX PRO 6000 Blackwell Server Edition, suite in 8 shards, one engine process per GPU; bf16 eager, one request at a time per process | no token limit; up to 255 choice options per question; no truncation (0 unsupported) |
