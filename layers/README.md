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
