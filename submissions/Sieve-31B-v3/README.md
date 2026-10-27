# Sieve-31B-v3 evaluation harness

Sieve-31B-v3 (`sthanika-ai/Sieve-31B-v3` at `481c43fb8ab910e6c9ca83d229ae6a32a9f18f5c`) is a full fine-tune of
`google/gemma-4-31B-it` at `842da379` (every linear projection of its 60 decoder layers, 29.29B trained parameters; the
token embeddings, tied with the output layer, and every norm are frozen). It is stored as a full bf16 model plus
`head.pt`. It answers through its own output layer, from the logits of the option labels at an `Answer: (` slot; it
generates nothing. The engine is `sieve.decision_index_engine:SieveEngine` from
[github.com/sthanika-ai/Sieve](https://github.com/sthanika-ai/Sieve) at `0fb2c5fe13b948fd67ec013478dcda99cfc68091`.

## Run

```sh
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install transformers==5.17.0 tokenizers==0.23.2 peft==0.21.0 huggingface_hub==1.32.0 safetensors==0.8.0 accelerate==1.15.0 jinja2 numpy pydantic httpx
pip install --no-deps "sieve-decisions @ git+https://github.com/sthanika-ai/Sieve@0fb2c5fe13b948fd67ec013478dcda99cfc68091"
pip install --no-deps "decision-index @ git+https://github.com/apolinario/decision-index@f5a6e4f"
hf download sthanika-ai/Sieve-31B-v3 --revision 481c43fb8ab910e6c9ca83d229ae6a32a9f18f5c --local-dir Sieve-31B-v3
python -m decision_index pipeline --engine sieve.decision_index_engine:SieveEngine --option model=Sieve-31B-v3 --out runs/Sieve-31B-v3
```

The weights take 58.3 GiB of GPU memory in bf16. The largest public request (53,703 tokens, 17 questions) peaked at
81.4 GiB allocated, so one RTX PRO 6000 (96 GB) runs every public request.

## What the engine does

- **Serving:** the model runs exactly as it is served, with everything read from `head.pt`.
  - The prompt is Gemma's own chat template with thinking off: a shared block (system turn, the list of questions, the
    state), then one block per question with its labelled options (A, B, …, up to 255) and an `Answer: (` slot.
  - The answer is the softmax of the model's output-layer logits of the question's option labels at its slot, divided
    by the calibrated temperature of the question's category (question type, number of options and kind of state),
    else the global one (1.27). This changes no answer.
  - Every choice question whose options may be reordered is read twice, with its options as given and rotated by
    half, and the two log-probability vectors are averaged.
- **Attention:** in the 10 global layers the shared block attends only to itself and each question's block attends to
  the shared block and to itself, in both directions; the 50 sliding-window layers are causal. Questions never see
  each other. The model was trained with this attention.
- **Several questions:** with a shared block of 2,048 tokens or more, the shared block is computed once and its keys
  and values serve every question and both option orders; this gives the same result as computing it with each
  question. Shorter requests run the shared block with the questions in packs.
- **What it never does:** truncate, drop options, tune prompts per benchmark or see gold labels.
- **Declared limits:** no token limit and up to 255 choice options per question. A request beyond 255 options is
  `Unsupported`; none occurred.
- **Out of memory:** raised as an ordinary error, so a resumed run retries it. None occurred.
- **Latency:** the engine times the eager path, which is the model's fastest path; there is no CUDA-graph path.
