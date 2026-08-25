# Layer-level residual-stream localization and steering

Tests whether the head-agnostic / layer-informative finding survives when both
the localization and the intervention move to layer granularity. Same estimator,
same datasets, same cross-steer matrix as the head pipeline — only the hook
point changes, from `layer.self_attn.o_proj.output` to `layer.output`.

## Running

```bash
sbatch scripts/verify_layers_determinism.sh                                   # run this first
sbatch scripts/localize_layers.sh                                            # both localizations
sbatch --export=ALL,LOC_PAIR=female-long_male-long     scripts/eval_layers.sh
sbatch --export=ALL,LOC_PAIR=female-single_male-single scripts/eval_layers.sh
for s in 42 43 44 45 46; do                                                  # baseline band
  sbatch --export=ALL,SEED=$s,LOC_PAIR=female-long_male-long     scripts/eval_layers_random.sh
  sbatch --export=ALL,SEED=$s,LOC_PAIR=female-single_male-single scripts/eval_layers_random.sh
done
```

## The matrix

Cross-steering is expressed with the same flags as the head pipeline, so no cell
has its own code path:

| axis | controlled by |
|---|---|
| which layers are steered | `--source` / `--base` (the localization) |
| which prompts are evaluated | `--eval_test` |
| which steering vector is applied | `--steering_add_path` / `--steering_sub_path` |

2 localizations × 2 eval sets × 2 steering vectors = 8 cells, each swept over
k ∈ {1,2,3,5,7,9,10} layers × 7 coefficients. Cross-localization is the pairing
of a cell under one `LOC_PAIR` with its counterpart under the other.

## Per-layer sweep (`--sweep_mode per_layer`)

Instead of selecting the top k layers and steering them together, steer each
layer individually and measure all of them. The x-axis becomes layer index.

```bash
sbatch --export=ALL,LOC_PAIR=female-long_male-long     scripts/eval_layers_per_layer.sh
sbatch --export=ALL,LOC_PAIR=female-single_male-single scripts/eval_layers_per_layer.sh
```

This changes what the experiment tests, for the better. Top-k asks whether ATP's
chosen layers beat random — seven coarse points, and it confounds "the ranking is
good" with "steering more layers does more". Per-layer measures every layer, so
the attribution score becomes a *prediction* to correlate against the
*measurement*. That correlation is the localization claim stated directly, and it
makes the random arm redundant: there is no selection left to randomize.

Because the map is no longer a selector, the sweep runs without one — a missing
localization warns rather than aborting.

**Cost.** Per cell it is `n_layers x n_vals` runs instead of `7 x n_vals`, i.e.
5.7x at 40 layers. `eval_layers_per_layer.sh` trims `N_VALS` to four values, giving
`40 x 4 x 8 cells = 1280` generation runs against the top-k sweep's 392. Judge
cost scales the same way. Widen `N_VALS` only if you need the N axis at layer
resolution.

**Output.** Results go to a `{algo}-per-layer` method directory, so the two
sweeps coexist and never overwrite each other. Gen filenames keep their shape —
the third numeric field just holds a layer index rather than a count, which is
why the scorer needs no code change:

```python
METHOD  = "atp-per-layer"
TOP_KS  = list(range(40))    # layer indices
NS      = [2, 5, 8, 10]
```

`steering_meta.json` records `sweep_mode` and `sweep_axis` so a directory can be
read correctly later without inferring which sweep produced it.

### Plotting

The scorer's `{metric}_dataset.csv` is already rows=N x cols=layer, so no new
scoring stage is needed:

```bash
python -m layers.plot_layer_sweep \
  --dataset_csv results_pipeline/.../plots/judge_0.5_dataset.csv \
  --layer_effects results_layers/Qwen1.5-14B-Chat/from_female-long_to_male-long/atp/eval/layer_effects.csv \
  --baseline 0.05 --out per_layer_effect.png
```

Effect vs layer index, one line per N, with the ATP profile overlaid on a second
axis and the Spearman correlation in the title. It also writes a tidy
`*_by_layer.csv` with measured and predicted columns side by side.

Read the correlation rather than the peak agreement. A method that ranks the
single best layer correctly but is otherwise uninformative is a different finding
from one that gets the whole profile right, and only the correlation separates
them. Near-zero would be the head-agnostic result extended to layers.

## Steering scale

The head pipeline normalizes each head-slice vector to unit norm and scales by
N ∈ [1..20]. That convention does not transfer: a unit vector is negligible
against a full residual stream (‖resid‖ ~ 10²–10³ in Qwen1.5-14B), so unit-norm
layer steering at N=10 is close to a no-op and would read as a false null.

Default is `--steering_scale relative`: the direction is rescaled to
`alpha × mean ‖resid‖` at that layer, with `alpha = N × n_scale`. N stays an
integer because the scorer parses it out of the gen filename; `n_scale=0.1`
turns N=1..10 into 0.1×..1.0× of a typical residual. The norm reference is
measured on the prompts *being steered*, not on the ones the vector came from,
so alpha means the same thing in the matched and cross-steer cells.

`raw` (CAA convention, vector untouched) and `unit` (head parity) are also
available. The effective alphas and the measured norms are written to
`steering_meta.json` in every cell.

## Cumulative vs marginal ranking

Residual-stream attribution at layer L includes every upstream contribution, so
late layers dominate by construction and a top-k list under `--rank_by
cumulative` will skew late for reasons that have nothing to do with the task.
`--rank_by marginal` ranks on `effect[L] − effect[L−1]` instead. Both are
computed from the same shards, so switching costs a reduction, not a re-run.
`layer_effects.png` plots both.

`steering_meta.json` also records `pair_spread`, the across-pair std of the
last-token difference. If the mean direction is small relative to that spread,
the steering vector is noise and a flat sweep says nothing about localization.
Check it before reading a null.

## Speed

The dominant cost was generation mode. The head pipeline offered two options and
they are a false dichotomy: `--kv_caching` steered the prefill only (fast, but
generated tokens never steered), and its default steered every position by
*disabling the cache* and re-forwarding the whole sequence at each decode step.
The second is what you want semantically and is quadratic in `max_new_tokens`.

Both cached modes cost the same; they differ only in semantics:

| mode | token-forwards, max_new=256 | max_new=24 | steers generated tokens |
|---|---|---|---|
| `recompute` (old default) | 61,056 | 2,940 | yes |
| `prefill` (**pipeline default**) | 366 | 134 | no |
| `all_steps` | 366 | 134 | yes |

That is ~167x less compute on the long-form cells and ~22x on single-token.
Wall-clock gain is smaller — decode steps are memory-bound — but it is the
difference between a sweep that finishes and one that does not.

**Why `prefill` is the default despite `all_steps` being semantically fuller.**
The head pipeline's `gemma_*` and `olmo_*` scripts all pass `--kv_caching`, which
is prefill-only. Since the entire point of the layer arm is to be read against
the head arm, the steering regime has to match; `all_steps` would make the two
non-comparable for a reason that has nothing to do with layers versus heads.
`--gen_mode all_steps` is one flag away as a sensitivity check.

Note what `prefill` does and does not mean. It is not "steering that stops":
generated tokens still attend to steered prompt keys and values in the KV cache,
so they are influenced throughout — just indirectly, through a modified context,
rather than by having their own residual stream displaced. The practical
consequence is that the steering pressure is fixed at the context while the
model's own generation dynamics compete with it, so the effect tends to decay
with output length.

That decay is a **pre-existing property of the head experiment's design, and it
sits on the axis this project measures**: at 24 tokens it is negligible, at 256
it is not, so the free-form arm is under-steered relative to the single-token arm
by an amount that is a function of output length rather than of the tasks. It
applies equally to the head and layer arms, so the layer-vs-head comparison is
fair — but the single-token-vs-free-form comparison inherits it in both. A single
`--gen_mode all_steps` cell on the free-form eval is the cheapest way to bound
how much of the format difference is really a length artefact.

`recompute` is kept as the reference `all_steps` is checked against:

```bash
sbatch scripts/verify_gen_mode.sh    # runs both, diffs generations, times each
```

The equivalence argument is sound, but "should match" is a reason to check rather
than a reason not to. If they diverge, `all_steps` is not free and the default
should go back.

Three smaller wins, all in the scripts already:

- **`--sdp_backend default` for eval.** Math SDPA is only needed for the
  localization *backward*; the eval sweep is forward-only, where flash SDPA is
  both faster and deterministic. Localization keeps `math`. Keep this consistent
  across all eval cells — the backends differ numerically, so a half-and-half
  sweep is not internally comparable.
- **`--gen_batch_size 25`** instead of the auto-chosen 16, halving the number of
  generate calls. Fix it once for the whole sweep: under left padding, batch size
  changes results, so cached baselines record it and rebuild on mismatch.
- **Shared unsteered baseline.** It depends only on the eval set and generation
  settings, not on localization or steering vector, so it now lives beside the
  steering cache. Across the 8-cell matrix that is 2 baseline generations instead
  of 8.

Not done, but available if the sweep is still too slow: multiple (layer, N)
configurations could share one batch by steering row-subsets of a replicated
batch, cutting generate calls by the number of configs packed. It changes
batching — and therefore results — so it would need its own equivalence check.

## Determinism

`determinism.py` covers the generation path, but it is not sufficient here. Two
gaps mattered enough to close:

**SDPA backward.** The localization pass differentiates through attention, and
`ModelHandler` only forces `attn_implementation="eager"` on the pyreft path — so
everywhere else transformers picks SDPA, whose flash and mem-efficient backward
kernels accumulate with atomics. That is nondeterministic by construction, and it
lands directly on the gradients, hence on every attribution value, hence on the
layer ranking. `--sdp_backend math` (the default in all three scripts) pins SDPA
to the math backend, whose backward is deterministic. Slower and heavier on
memory; at batch 1 that is affordable.

**`warn_only=True`.** Fail-open was the right call for generation. For a ranking
derived from gradients it is the wrong default, because the failure produces
plausible numbers rather than an error. `--strict_determinism` flips it to
fail-closed, so an offending op raises and names itself. On in the scripts.

Two smaller reuse hazards, both closed by recording build conditions and
rebuilding on mismatch rather than silently reusing:

- The steering cache is shared across cells and accumulated batch by batch, so
  one built at batch 8 differs in the last bits from one built at batch 16.
- With left padding, batched generation is batch-size dependent. The steered and
  unsteered arms are compared item by item, so a baseline reused from a run with
  a different batch size yields a difference that is an artefact of batching.

Top-k layer selection uses an explicit sort with ties broken by layer index,
rather than `torch.topk`, whose tie-breaking is not documented as stable — and
ties are exactly what a layer-agnostic attribution map would produce.

### What each change since the determinism pass does to it

| change | determinism impact |
|---|---|
| `--gen_mode prefill` | none — no RNG, `do_sample=False`, fixed intervention order |
| `--gen_batch_size 25` | reproducible, but **changes results** vs 16 under left padding; fix it once per sweep |
| shared baseline cache | safe only because the build key gates reuse; atomic write handles concurrent jobs |
| per-layer sweep | none — sweep order is `range(n_layers)`, selections are sorted |
| `--sdp_backend default` for eval | the one to actually verify (see below) |

Two gaps this audit found and closed. Neither cache key recorded `sdp_backend`,
so a cache built under math SDPA would have been silently reused under flash —
both caches come from forward passes, so the backend is part of their provenance.
And the verification harness pinned `--sdp_backend math` in its eval leg, meaning
it verified a configuration nothing actually runs; it now mirrors
`eval_layers_per_layer.sh` flag for flag.

**On flash SDPA in the eval sweep.** Flash attention's nondeterminism is in the
*backward* (atomic accumulation); the forward has a fixed reduction order. The
eval sweep is forward-only, so `default` should be safe — and `--strict_determinism`
is the backstop: with `warn_only=False`, an op without a deterministic
implementation raises and names itself rather than quietly varying. Localization,
which does differentiate through attention, keeps `--sdp_backend math`.

That is the argument, not a measurement. Run the harness.

### Verifying it

Assertions are cheap; run the check.

```bash
sbatch scripts/verify_layers_determinism.sh          # two runs into two roots, then diff
python -m layers.verify_determinism ./results_layers_verifyA ./results_layers_verifyB
```

Exit 0 means every compared artifact is bit-identical. The report is ordered
causally — shards, map, steering cache, layer selections, generations — so the
first failure is the one to debug and later ones are usually downstream of it. A
mismatch in `determinism.json` is reported separately as env drift: it explains a
diff rather than being one, and does not set the exit code.

Do this before the full sweep. A nondeterministic backward yields a
plausible-looking ranking, so the failure is invisible in the results themselves
and only ever shows up as a diff between two runs.

Every output directory gets a `determinism.json` recording torch/transformers/
nnsight versions, CUDA and device, the SDP and cuDNN flags, seed, and batch size.
`steering_meta.json` additionally carries content hashes of the attribution map,
steering cache, and norm reference — so two runs that agree there but disagree
downstream have a generation-path problem, while two that disagree there never
had a chance to match.

## Running the two localizations concurrently

The two `eval_layers.sh` jobs share `--results_root ./results_layers`, and their
output prefixes differ by `from_{source}_to_{base}`, so gen files, configs, and
attribution maps never collide.

One thing is shared on purpose: the steering cache under
`{root}/{model}/_steering_cache/`. Reuse across localizations is the point — the
same steering set should not be re-traced four times — but it means two
concurrent jobs write the same file. `torch.save` is not atomic, so writes go
through a temp file in the same directory followed by `os.replace`, and reads
that fail are treated as a miss and rebuilt rather than raising. Two jobs racing
to build it wastes a pass and nothing else: the computation is deterministic, so
whichever write lands last is byte-identical to the one it replaced.

If you scale past two concurrent jobs, note this is the only shared mutable
state; everything else is partitioned by output prefix.

## Verifying the deployment

`ImportError: cannot import name X from layers.Y` when `X` plainly exists in the
source is a deployment problem, and it has two causes that produce identical
tracebacks: a partial copy (some files updated, others not), or stale bytecode
(Python reuses a cached `.pyc` when the source's recorded mtime and size still
match, so `cp -p` / `scp -p` / `rsync -t` can make a new source look like the old
one — the file reads correctly while Python keeps running the previous version).

```bash
bash scripts/check_layers_install.sh
```

Clears `__pycache__`, checksums every file against `layers/MANIFEST.sha256`, and
imports each module to confirm the symbols `run_layers.py` needs are actually
present — naming which symbol is missing from which module rather than failing on
whichever import comes first. A missing third-party dependency is reported
separately as an environment problem, since recopying files would not fix it.

Run it after any copy to the cluster and before resubmitting.

## When a job dies at model load

`RuntimeError: ... Error 802: system not yet initialized` from
`torch._C._cuda_init()` is the NVSwitch fabric manager / IMEX layer not being up
on the allocated node. It is a node fault, not a job fault: nothing in this
pipeline has run yet, and retrying on the same node fails identically. On a
preemptable partition with `--requeue` that can burn a long series of
allocations, each dying at model load having produced nothing.

`scripts/_preflight.sh` is sourced by all four scripts and probes CUDA with a
real allocation before any work starts. On failure it records the node in
`logs/bad_gpu_nodes.txt`, prints the `--exclude=` list to resubmit with, and
requeues (up to three restarts — past that the likelier explanation is a
partition-wide problem or a bad `--gres` request, and looping hides it). It
distinguishes a broken node (exit 75, requeue) from a wrong conda env (exit 70,
no requeue), since requeueing the latter would loop forever.

Confirm by hand with `nvidia-smi` on the node, and
`nvidia-smi -q | grep -i -A2 fabric`.

## Scoring

Output layout and gen filenames are identical to the head pipeline
(`{N}_{targeted|random}_{steer}_{k}_{eval_source}_gen.json`), so
`eval_pipeline_bias.py` scores these runs after three edits:

```python
RESULTS_DIR = os.path.join(BASE_DIR, "results_layers")
TOP_KS      = [1, 2, 3, 5, 7, 9, 10]   # layer counts, not fractions
NS          = [1, 2, 4, 5, 6, 8, 10]
```

`MODEL_ID`, `METHOD`, `LOCALIZATIONS`, `EVAL_SUB_DIRS`, `STEER_SUB_DIRS` are
unchanged. Note `TOP_KS` values are now counts — anything downstream that reads
them as fractions of the search space needs the same adjustment.

## Notes

- `layer.output` is a tuple on transformers ≤4.53.x and a bare tensor on several
  4.54+ decoders. Since `syc` has drifted between 4.53.3 and 4.57.1 more than
  once, the shape is detected at startup with a real 2-token trace rather than
  inferred from a version string.
- The reduction enumerates shards from disk and records the count in a sidecar,
  rebuilding when the two disagree — the failure mode that produced half-shard
  attribution maps in the conflicts work.
- Random layer draws are seeded on `(seed, k)`, so the k=3 draw is not a prefix
  of the k=5 draw. Otherwise the baseline's own k-curve is autocorrelated and
  cannot be read against the targeted one.
- nnsight stays at the pinned 0.4.11; none of these scripts upgrade it.
