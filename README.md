# LLM-Evidence-Graph

Cluster short texts using pairwise judgments from an LLM.

Each experiment seed runs its own query round and its own training run. The
reported number is the mean over the seeds in the config.

Banking77 is bundled as a runnable example.

## Layout

```
leg.py                     CLI entry point
configs/default.yaml       configuration template
configs/smoke.yaml         end-to-end check against a local mock judge
data/bank77/               texts.json (3,080 utterances), labels.npy (77 classes)
prompts/bank77.txt         the Banking77 judge prompt
src/llm_evidence_graph/    the package
scripts/mock_judge.py      local stand-in judge, for a zero-cost run
```

## Install

Python 3.10 or newer.

```
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

A GPU is used if present; everything also runs on CPU.

## Configure

```
cp .env.example .env
cp configs/default.yaml configs/mine.yaml
```

**1. The judge endpoint** (`api`). Any OpenAI-compatible server exposing
`/chat/completions`.

```yaml
api:
  base_url: "https://your-endpoint.example.com/v1"
  api_key_env: "LLM_API_KEY"
  model: "your-model-id"
  rpm_limit: 500
  tpm_limit: 2000000
  concurrency: 64
  max_tokens: 900
```

The key is read from the environment variable named by `api_key_env`. Put it
in `.env`:

```
LLM_API_KEY=your-key-here
```

`rpm_limit` and `tpm_limit` should be your provider's real limits; the client
throttles to them. `concurrency` is the number of in-flight requests.

The judge must support structured outputs (`response_format` with a
`json_schema`).

The client ignores system proxy settings. If your endpoint is only reachable
through a proxy, remove `trust_env=False` from
`src/llm_evidence_graph/llm/scheduler.py`.

**2. The text embedding** (`embedding`).

```yaml
embedding:
  provider: sentence-transformers
  model: "sentence-transformers/all-MiniLM-L6-v2"
  dimension: 384
  batch_size: 256
  cache: cache/embeddings.npy
  instruction: ""
```

Two providers:

- `placeholder` — random unit vectors, for a dry run.
- `sentence-transformers` — any model loadable by that package. Set `instruction`
  to a prefix if the model expects one (E5 uses `"query: "`; Instructor uses a
  task instruction). Leave it empty otherwise.

Embeddings are cached at `cache`; delete the file to recompute. An existing
`.npy` is loaded directly, so you can pair `data/bank77/` with any embedding
file. Rows must line up with `data/bank77/texts.json` in file order,
L2-normalized.

**3. Retrieval** (`retrieval`). For each text, the top
`pool_depth * candidates_per_block` neighbours by cosine similarity are split
into `pool_depth` blocks of `candidates_per_block`. The defaults (6 x 10 = 60)
issue `pool_depth` requests per text.

**4. Method hyper-parameters** (`graph`, `train`, `cluster`). The defaults are
`tau = 0.4`, no per-target edge cap, six experiment seeds, and one KMeans run
per trained representation.

`api.temperature` and `api.pass_seed` control the judge's sampling. At the
defaults (`0.0`, `false`) every round sends the same request, so the six seeds
differ only through training randomness; raise the temperature or set
`pass_seed: true` if your server supports a sampling seed.

## Use

```
python leg.py --config configs/mine.yaml embed
python leg.py --config configs/mine.yaml manifest
python leg.py --config configs/mine.yaml request
python leg.py --config configs/mine.yaml train
python leg.py --config configs/mine.yaml summary
```

- `embed` computes or loads the text embedding.
- `manifest` writes the retrieval list, one block per request.
- `request` runs one query round per seed and appends responses to
  `raw_seed<N>.jsonl`. Rerunning skips anything already answered successfully,
  so an interrupted round resumes where it stopped. `--seed N` limits it to one seed.
- `train` builds the graph, trains the encoder, clusters, and reports ACC and
  NMI. `--seed N` limits it to one seed.
- `summary` aggregates the per-seed results.

Or everything at once:

```
python leg.py --config configs/mine.yaml run
```

Total query volume is `seeds * pool_depth * N` requests. For the bundled
Banking77 with the defaults that is 110,880.

Outputs land in `cache_dir`:

```
manifests.jsonl       the request list
raw_seed<N>.jsonl     one JSON line per request, keyed by request_id
z_seed<N>.npy         the learned representation for seed N
results_seed<N>.json  that seed's ACC and NMI
summary.json          the mean and spread over seeds
```

## A zero-cost run

```
python scripts/mock_judge.py &
LLM_API_KEY=mock python leg.py --config configs/smoke.yaml run
```

The mock scores each candidate by trigram overlap with the target and adds a
little noise, so the rounds differ from one another. The embedding is random.
