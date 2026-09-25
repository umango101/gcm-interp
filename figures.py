#!/usr/bin/env python3
"""Paper figures for the free-form vs single-token localization comparison.

Self-contained port for umango101/gcm-interp (branch `cleanup`). It reads the
judged accuracy files directly from this repo's `results_pipeline/` tree, so
it no longer needs figdata.py, cellgrid.py, eval/patch_site.py or
eval/response_span.py from the ayushimehrotra checkout.

    python figures.py                          # levels + bytask, long-form eval
    python figures.py --which bytask           # just the by-task figure
    python figures.py --eval single --which levels,grid --controls
    python figures.py --font_scale 1.4         # larger text everywhere

EXPECTED LAYOUT
---------------
results_pipeline/<model>/from_<task>-<arm>_to_<base>/<algo>/
    <task>-<ev>_eval/<task>-<ev>_steer/accuracy/
        <N>_<targeted|random>_steer_topk_<k>_gen_accuracy_<metric>.json.accuracy.json

  <algo>    "atp" for the localizations, "random" for the uniform-random control
  <arm>     long (free-form) / single (single-token localization)
  <ev>      long / single evaluation; only files with steer mode == eval mode
            are used, matching the original figure

Each file holds {"q1": <0-1 accuracy>}. At every budget k the curve takes the
best steering factor N, for every arm including random, so the curves stay
comparable.

RANDOM BASELINE
---------------
The uniform-random control is written under both from_<task>-long and
from_<task>-single, but the two copies are the same intervention (fixed seed;
identical values in every file on this branch), so they are pooled into one
"random" curve per (model, task). On this branch the control exists only for
bias and summarization; the verse and factual-recall panels simply have no
random line.

FIGURES
-------
  levels    both arms' mean accuracy against budget, pooled over model-task
            pairs (--controls adds the random arm)
  bytask    one panel per task, both evaluation modes, with the random baseline
  grid      per-cell supplement: one panel per (task, model)
"""
import argparse
import json
import os
import re
import statistics as st
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D


# ------------------------------------------------------------------ font scale
# Every font size in this file goes through fs(), so one number controls all
# text. Margins and header offsets that have to make room for text are scaled
# by the same factor (see margin()/from_top()). Override with --font_scale.
FONT_SCALE = 1.25


def fs(size):
    """Scale a base font size (in points) by FONT_SCALE."""
    return size * FONT_SCALE


def margin(frac):
    """Scale a figure-fraction margin that holds text (left/bottom)."""
    return frac * FONT_SCALE


def from_top(y):
    """Scale a figure-fraction position measured down from the top edge."""
    return 1 - (1 - y) * FONT_SCALE


# ------------------------------------------------------------------ locations
HERE = Path(__file__).resolve().parent          # figures are written here


def find_results_root(start=HERE, name="results_pipeline"):
    """Walk up from this file until a results_pipeline/ directory turns up,
    so the script works from the repo root or from a subfolder like analysis/."""
    for d in (start, *start.parents):
        if (d / name).is_dir():
            return d / name
    return start / name


# $GCM_RESULTS or --results_root override the search
RESULTS_ROOT = Path(os.environ.get("GCM_RESULTS") or find_results_root())


# ------------------------------------------------------------------ scope
GRID_MODELS = ["gemma-3-12b-it", "Qwen1.5-14B-Chat", "Qwen1.5-32B-Chat",
               "OLMo-2-1124-13B-DPO", "Falcon3-10B-Instruct"]
MODELS = list(GRID_MODELS)
SHORT = {"gemma-3-12b-it": "Gemma-3-12B", "Qwen1.5-14B-Chat": "Qwen1.5-14B",
         "Qwen1.5-32B-Chat": "Qwen1.5-32B", "OLMo-2-1124-13B-DPO": "OLMo-2-13B",
         "Falcon3-10B-Instruct": "Falcon3-10B"}

# directory prefix -> task name used in titles
TASKNAME = {"verse": "verse", "paragraph": "summarization", "female": "bias",
            "lying": "factual recall", "extraversion": "persona"}
TASKS = ["verse", "summarization", "factual recall", "bias"]
KS = [0.01, 0.03, 0.05, 0.07, 0.09, 0.1, 0.5, 1.0]

# Which accuracy tag each (eval mode, task) is scored under on this branch.
# Long-form eval: "comb" everywhere. Single-token eval: "w_rf" for bias,
# "mcqa" for the rest. ("mcqa_pre" sits next to mcqa and is deliberately NOT
# used.) Exactly one tag per cell, so two spellings of the same number can
# never both enter the max-over-N.
METRIC = {"long": "comb", "single": "mcqa"}
METRIC_TASK = {("single", "bias"): "w_rf"}


def want_metric(eval_mode, task):
    return METRIC_TASK.get((eval_mode, task), METRIC[eval_mode])


FN_RE = re.compile(
    r"^(?P<N>\d+)_(?P<reps>random|targeted)_(?P<method>steer|mean)_topk_"
    r"(?P<topk>[\d.]+)_gen_accuracy_(?P<metric>[a-z_0-9]+)"
    r"\.json\.accuracy\.json$")
FROM_RE = re.compile(r"^from_(?P<src>.+?)-(?P<arm>long|single)_to_(?P<base>.+)$")
MODE_RE = re.compile(r"^(?P<task>.+)-(?P<mode>long|single)$")


# ------------------------------------------------------------------ palette
INK = "#2b2f36"
LF = "#0680a4"      # free-form localization
ST = "#c25a34"      # single-token localization
HAIR = "#b9bec6"
BAND = "#7fb3c4"
CTRL = "#8d939c"    # random baseline


def strip_mode(s):
    m = MODE_RE.match(s)
    return (m.group("task"), m.group("mode")) if m else (s, None)


# ------------------------------------------------------------------ loader
def harvest(eval_mode="long", include_random=False, algo_dir="atp",
            root=None):
    """(model, task, arm) -> {k: best accuracy over steering factors N}.

    arm is "long" / "single" for the two localizations and, with
    include_random, "random" for the uniform control.
    """
    root = Path(root or RESULTS_ROOT)
    if not root.exists():
        raise SystemExit(f"results root not found: {root}  "
                         f"(pass --results_root or set $GCM_RESULTS)")
    g = defaultdict(lambda: defaultdict(list))
    for p in root.rglob("*.accuracy.json"):
        parts = p.relative_to(root).parts
        # model / from_* / algo / *_eval / *_steer / accuracy / file
        if len(parts) != 7 or parts[5] != "accuracy":
            continue
        model, frm, algo, evald, steerd, _, fname = parts
        if any(c.endswith("_old") for c in parts):
            continue
        m = FN_RE.match(fname)
        if not m or m.group("method") != "steer":
            continue
        is_ctrl = algo == "random"
        if algo != algo_dir and not (is_ctrl and include_random):
            continue
        if m.group("reps") != ("random" if is_ctrl else "targeted"):
            continue
        fm_ = FROM_RE.match(frm)
        if not fm_ or not evald.endswith("_eval") or not steerd.endswith("_steer"):
            continue
        task = TASKNAME.get(fm_.group("src"))
        _, ev = strip_mode(evald[:-5])
        _, stm = strip_mode(steerd[:-6])
        if task is None or ev != eval_mode or stm != ev:
            continue
        if m.group("metric") != want_metric(ev, task):
            continue
        try:
            v = json.load(open(p)).get("q1")
        except Exception:
            continue
        if v is None:
            continue
        # uniform random is arm-independent: pool the -long and -single copies
        key = "random" if is_ctrl else fm_.group("arm")
        g[(model, task, key)][float(m.group("topk"))].append(float(v))
    # best steering factor at each budget, identically for every arm
    return {key: {k: max(vs) for k, vs in d.items()} for key, d in g.items()}


# ------------------------------------------------------------------ helpers
def boot_ci(rows, n=10000, seed=0):
    if len(rows) < 2:
        return float("nan"), float("nan")
    r = np.random.default_rng(seed)
    a = np.asarray(rows, float)
    m = np.sort(r.choice(a, (n, len(a))).mean(1))
    return float(m[int(.025 * n)]), float(m[int(.975 * n)])


def style():
    """Palatino (TeX Gyre Pagella) if its .otf files are in ./fonts next to
    this script, otherwise DejaVu Serif."""
    for f in sorted((HERE / "fonts").glob("*.otf")):
        fm.fontManager.addfont(str(f))
    names = {f.name for f in fm.fontManager.ttflist}
    fam = ("TeX Gyre Pagella" if any("Pagella" in n for n in names)
           else "Palatino" if "Palatino" in names else "DejaVu Serif")
    plt.rcParams.update({
        "font.family": "serif", "font.serif": [fam, "DejaVu Serif"],
        "mathtext.fontset": "custom", "mathtext.rm": fam,
        "mathtext.it": f"{fam}:italic", "mathtext.bf": f"{fam}:bold",
        "mathtext.cal": fam, "mathtext.sf": fam, "mathtext.tt": "DejaVu Sans Mono",
        # base size for everything without an explicit fontsize (axis labels,
        # default tick labels in the levels figure)
        "font.size": fs(9),
        "axes.linewidth": 0.7, "axes.edgecolor": INK, "axes.labelcolor": INK,
        "text.color": INK, "xtick.color": INK, "ytick.color": INK,
        "xtick.direction": "out", "ytick.direction": "out",
        "xtick.major.width": 0.7, "ytick.major.width": 0.7,
        "xtick.major.size": 3, "ytick.major.size": 3,
        "xtick.minor.size": 0, "ytick.minor.size": 0,
        "legend.frameon": False, "figure.dpi": 200,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
    })
    return fam


def titled(ax, title, subtitle=None, fontsize=10.5):
    # fontsize is a base size; scaled here rather than in the signature so that
    # --font_scale (set after import) still applies
    lines = 0 if not subtitle else subtitle.count("\n") + 1
    ax.set_title(title, fontsize=fs(fontsize), pad=(9 + 9 * lines) * FONT_SCALE)
    if subtitle:
        ax.text(0.5, 1.015, subtitle, transform=ax.transAxes, ha="center",
                va="bottom", fontsize=fs(8.5), color=INK, alpha=0.62,
                linespacing=1.45)


def bare(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(False)
    ax.set_axisbelow(True)


EVALS = (("long", "Free-form evaluation", "judged accuracy"),
         ("single", "Single-token evaluation", "letter-match accuracy"))

MIN_MODELS = 3


def curve(acc, models, task, arm):
    """(budgets, mean, lo, hi, n_models) across models at each budget."""
    xs, mu, lo, hi, ns = [], [], [], [], []
    for k in KS:
        v = [acc[(m, task, arm)][k] for m in models
             if acc.get((m, task, arm)) and k in acc[(m, task, arm)]]
        if len(v) < MIN_MODELS:
            continue
        l, h = boot_ci(v)
        xs.append(k)
        mu.append(st.mean(v))
        lo.append(l)
        hi.append(h)
        ns.append(len(v))
    return xs, mu, lo, hi, (max(ns) if ns else 0)

def verdict(xs, lo, hi):
    """Budgets where each arm's advantage is resolved (CI excludes zero).

    The two evaluation modes give opposite answers, so a hardcoded title would
    mislabel one of them. Every title below is derived from these two lists.
    """
    ahead_long = [xs[z] for z in range(len(xs)) if lo[z] > 0]
    ahead_single = [xs[z] for z in range(len(xs)) if hi[z] < 0]
    return ahead_long, ahead_single

def headline(ahead_long, ahead_single, evalmode):
    where = "free-form" if evalmode == "long" else "single-token"
    if ahead_long and not ahead_single:
        return ("Free-form localization's advantage is confined to "
                "small budgets")
    if ahead_single and not ahead_long:
        return (f"On {where} evaluation, single-token localization is the "
                f"one that leads")
    if ahead_long and ahead_single:
        return "Which localization leads depends on the budget"
    return "The two localizations are indistinguishable at every budget"

# Only the uniform-random control exists in results_pipeline on this branch.
# (The ayushimehrotra tree also had depth-matched "randomlayer" arms.)
CTRL_ARMS = (("random", "uniform random"),)

def levels_figure(acc, out, evalmode, with_controls=False, models=None):
    """Both arms' mean curves. Companion to the gap figure, own takeaway.

    Direct labels on the curves instead of a legend: a legend makes the reader
    look away from the line to decode a colour, which is the same
    cross-referencing cost as a second panel, just smaller.

    with_controls adds the uniform and depth-matched random arms. Each curve is
    averaged over the cells where that arm exists: the localizations over all
    19, the controls over the 12 that have them (verse, summarization, persona
    -- bias and factual recall were never run with controls). The cell counts
    are printed in the subtitle because the curves therefore rest on different
    cell sets, and part of any vertical gap between a localization and a control
    is that difference rather than the arm.
    """
    # GRID_MODELS, not MODELS: the pooled figure and the per-cell grid must
    # rest on the same cell set, or "18 cells averaged" and "23 paired cells"
    # describe different populations and the reader cannot reconcile them. The
    # only reason on record for holding Falcon3-10B out was CLAUDE.md 2 -- the
    # two cells breaching the 0.10 provenance floor were both Falcon -- and that
    # is an atp-vs-random-control statement, about arms this figure does not
    # draw unless --controls is passed.
    #
    # Caveat kept in view rather than hidden: Falcon's atp sweep also covers
    # N=15,20 where the others stop at 10, and every arm here is a max over N,
    # so a wider sweep can only raise its level. That cancels in the LF-ST
    # difference this figure is about (both arms share the sweep) but does
    # slightly lift Falcon's contribution to the absolute mean.
    paired = [(m, t) for m in (models or GRID_MODELS) for t in TASKS
              if acc.get((m, t, "long")) and acc.get((m, t, "single"))]
    fig, ax = plt.subplots(figsize=(5.4, 3.5))
    bare(ax)
    ax.set_xscale("log")

    curves = {}
    for arm, colour in (("long", LF), ("single", ST)):
        xs, mu, lo, hi = [], [], [], []
        for k in KS:
            v = [acc[(m, t, arm)][k] for m, t in paired if k in acc[(m, t, arm)]]
            if len(v) < 3:
                continue
            xs.append(k)
            mu.append(st.mean(v))
            l, h = boot_ci(v)
            lo.append(l)
            hi.append(h)
        curves[arm] = (xs, mu, lo, hi)
        ax.fill_between(xs, lo, hi, color=colour, alpha=0.13, lw=0, zorder=1)
        ax.plot(xs, mu, color=colour, lw=2.0, zorder=3,
                ls="-" if arm == "long" else (0, (4, 2.2)),
                solid_capstyle="round")
        ax.plot(xs, mu, "o", color=colour, ms=3.6, mew=0, zorder=4)

    n_ctrl = 0
    if with_controls:
        # Controls sit behind the localizations in grey with no CI band: they are
        # a reference level, and three more ribbons would bury the two curves the
        # figure is about.
        for arm, label in CTRL_ARMS:
            cells = [(m, t) for m, t in paired if acc.get((m, t, arm))]
            n_ctrl = max(n_ctrl, len(cells))
            xs, mu = [], []
            for k in KS:
                v = [acc[(m, t, arm)][k] for m, t in cells
                     if k in acc[(m, t, arm)]]
                if len(v) < 3:
                    continue
                xs.append(k)
                mu.append(st.mean(v))
            if not xs:
                continue
            dash = {"random": (0, (1, 1.6)),
                    "randomlayer_long": (0, (5, 1.8)),
                    "randomlayer_single": (0, (2.5, 1.6))}[arm]
            ax.plot(xs, mu, color=CTRL, lw=1.3, ls=dash, zorder=2,
                    solid_capstyle="round")
            curves[arm] = (xs, mu, None, None)
        # label the controls once, on the rightmost point of the topmost one
        top = max((a for a, _ in CTRL_ARMS if a in curves),
                  key=lambda a: max(curves[a][1]), default=None)
        if top is not None:
            xs, mu = curves[top][0], curves[top][1]
            ax.text(xs[-1] * 0.86, max(mu) + 0.02, "random controls",
                    color=CTRL, fontsize=fs(8.8), ha="right", va="bottom")

    xL, muL, _, _ = curves["long"]
    xS, muS, _, _ = curves["single"]
    i = xL.index(0.03) if 0.03 in xL else 1
    ax.text(xL[i] * 0.93, muL[i] + 0.035, "free-form", color=LF,
            fontsize=fs(9.5), ha="center", va="bottom")
    j = xS.index(0.03) if 0.03 in xS else 1
    ax.text(xS[j] * 1.02, muS[j] - 0.035, "single-token", color=ST,
            fontsize=fs(9.5), ha="center", va="top")

    pk = max(range(len(muL)), key=lambda z: muL[z])
    # shade the budgets where free-form actually leads, so the title's claim is
    # something the reader can see rather than something they must take on trust
    # Shade only as far as the last budget actually measured with free-form
    # ahead. The grid jumps 0.1 -> 0.5, so the crossover is unresolved; shading
    # to a midpoint would imply a location the data does not establish.
    # Shade a leading region only when one arm leads over a contiguous run from
    # the smallest budget; on single-token eval the arms trade places and a
    # shaded band would assert a structure the data does not have.
    lead = [z for z in range(len(xL)) if muL[z] > muS[z]]
    contiguous = lead[:1] == [0] and lead == list(range(len(lead)))
    if contiguous and len(lead) < len(xL):
        edge = xL[lead[-1]] * 1.14
        ax.axvspan(0.0085, edge, color=LF, alpha=0.05, lw=0, zorder=0)
        ax.text(edge * 0.94, 0.022,
                "free-form ahead at every\nbudget measured here",
                fontsize=fs(8.5), color=INK, alpha=0.62, ha="right",
                va="bottom", linespacing=1.35)

    ax.set_xticks([0.01, 0.03, 0.05, 0.1, 0.5, 1.0])
    ax.set_xticklabels(["1%", "3%", "5%", "10%", "50%", "100%"])
    ax.set_xlim(0.0085, 1.25)
    top_y = max(max(curves["long"][3]), max(curves["single"][3]))
    if with_controls:            # a control curve may sit above both CI ribbons
        top_y = max([top_y] + [max(curves[a][1]) for a, _ in CTRL_ARMS
                               if a in curves])
    ax.set_ylim(0, top_y + 0.07)
    ax.set_xlabel("Fraction of attention heads steered")
    ax.set_ylabel("Judged accuracy")
    gx, gmu, glo, ghi = [], [], [], []
    for k in KS:
        d = [acc[(m, t, "long")][k] - acc[(m, t, "single")][k] for m, t in paired
             if k in acc[(m, t, "long")] and k in acc[(m, t, "single")]]
        if len(d) < 3:
            continue
        gx.append(k)
        l, h = boot_ci(d)
        glo.append(l)
        ghi.append(h)
    aheadL, aheadS = verdict(gx, glo, ghi)
    sub = (f"{'free-form' if evalmode == 'long' else 'single-token'} evaluation "
           f"and steering   ·   bands: bootstrap 95% CI of the mean")
    if with_controls:
        # state both counts: the curves are averaged over different cell sets
        sub += (f"\nlocalizations averaged over {len(paired)} model–task pairs, "
                f"controls over the {n_ctrl} that have them")
    titled(ax, headline(aheadL, aheadS, evalmode), sub)
    # "cells" is internal shorthand; readers see the setup, not the tally
    fig.savefig(HERE / f"{out}.pdf")
    fig.savefig(HERE / f"{out}.png", dpi=400)
    plt.close(fig)
    return len(paired), curves

def supplement(acc, out, evalmode, note=None, empty="not\nlocalized"):
    """Per-cell grid. Detail is the point here, so the panel count is the design.

    `note` names the intervention site and response span under the title. It is
    not decoration: dose is not comparable across sites (CLAUDE.md 9.3), so a
    grid that does not say which tree it was read from invites exactly the
    cross-site comparison the data cannot support. `empty` is what an absent
    cell says -- on a variant tree "not localized" would be a false claim about
    cells that are localized and merely have not been steered and judged there.
    """
    n_ctrl = set()
    nrow, ncol = len(TASKS), len(GRID_MODELS)
    fig = plt.figure(figsize=(1.6 * ncol + 1.0, 1.42 * nrow + 1.1))
    # left/bottom margins leave room for the shared axis titles; the per-panel
    # ylabels carry task names, so the quantity label goes outside them
    # left margin holds exactly three things: the rotated quantity label, the
    # rotated task name and the "0/.5/1" ticks. Sized to the content (and scaled
    # with the font); if a longer task name than "Summarization" is ever added,
    # widen this rather than shrinking the font.
    gs = GridSpec(nrow, ncol, figure=fig, top=from_top(0.862),
                  bottom=margin(0.10), left=margin(0.083), right=0.985,
                  hspace=0.45 * FONT_SCALE, wspace=0.28 * FONT_SCALE)
    n = 0
    # rows are tasks, columns are models
    for i, task in enumerate(TASKS):
        # the row's label and y tick labels belong on the leftmost panel that
        # actually has data: set_axis_off() on an empty first column would
        # otherwise take the label and the y axis with it
        first = next((jj for jj, mm in enumerate(GRID_MODELS)
                      if acc.get((mm, task, "long")) and
                      acc.get((mm, task, "single"))), None)
        for j, model in enumerate(GRID_MODELS):
            ax = fig.add_subplot(gs[i, j])
            cL = acc.get((model, task, "long"))
            cS = acc.get((model, task, "single"))
            if not cL or not cS:
                ax.set_axis_off()
                ax.text(0.5, 0.5, empty, ha="center", va="center",
                        fontsize=fs(7), color=HAIR, transform=ax.transAxes)
            else:
                n += 1
                bare(ax)
                ax.set_xscale("log")
                # controls first and faint, so they read as background; each
                # depth-matched control takes its own arm's colour
                for cname, colour in (("random", CTRL),):
                    cc = acc.get((model, task, cname))
                    if not cc:
                        continue
                    ks = [k for k in KS if k in cc]
                    ax.plot(ks, [cc[k] for k in ks], ls=(0, (1, 1.7)),
                            color=colour, lw=1.0, alpha=0.85, clip_on=False,
                            zorder=1)
                    n_ctrl.add((model, task))
                for curve, colour, ls in ((cL, LF, "-"), (cS, ST, (0, (3, 2)))):
                    ks = [k for k in KS if k in curve]
                    ax.plot(ks, [curve[k] for k in ks], ls=ls, color=colour,
                            lw=1.1, clip_on=False, zorder=3)
                ax.set_xlim(0.0085, 1.25)
                ax.set_ylim(-0.03, 1.03)
                ax.set_xticks([0.01, 0.1, 1.0])
                ax.set_xticklabels(["1%", "10%", "100%"], fontsize=fs(7))
                ax.set_yticks([0, 0.5, 1.0])
                ax.set_yticklabels(["0", ".5", "1"] if j == first else [],
                                   fontsize=fs(7))
                if j == first:
                    ax.set_ylabel(task.capitalize(), fontsize=fs(8.5),
                                  labelpad=4)
            if i == 0:
                ax.set_title(SHORT[model], fontsize=fs(9), pad=5)
    where = "free-form" if evalmode == "long" else "single-token"
    # the note takes a line between title and legend, so the whole header slides
    # up rather than letting the three overlap; offsets grow with the font
    y_title, y_note, y_legend = (0.988, 0.9755, 0.9605) if note else \
                                (0.972, None, 0.952)
    y_title, y_legend = from_top(y_title), from_top(y_legend)
    y_note = from_top(y_note) if y_note is not None else None
    fig.text(0.5, y_title, f"Localization on {where} evaluation, "
             f"per task and model", ha="center", fontsize=fs(10.5))
    if note:
        fig.text(0.5, y_note, note, ha="center", va="top", fontsize=fs(8),
                 color=INK, alpha=0.62)
    handles = [Line2D([], [], color=LF, lw=1.4, ls="-",
                      label="Free-form localization"),
               Line2D([], [], color=ST, lw=1.4, ls=(0, (3, 2)),
                      label="Single-token localization"),
               Line2D([], [], color=CTRL, lw=1.0, ls=(0, (1, 1.7)),
                      label="Uniform random")]
    if not n_ctrl:
        handles = handles[:2]
    # two rows rather than one: five entries on a single line stretched the
    # legend the full figure width and pushed the panels down
    ncol = 3 if len(handles) > 2 else 2
    fig.legend(handles=handles, loc="upper center", ncol=ncol,
               bbox_to_anchor=(0.5, y_legend), fontsize=fs(8.5),
               handlelength=2.6, columnspacing=2.0, labelspacing=0.5,
               borderpad=0)
    fig.supxlabel("Fraction of attention heads steered", fontsize=fs(10),
                  y=margin(0.028), color=INK)
    # the two evaluation modes are scored by different metrics (EVALS), and this
    # label was hardcoded to the long-form one back when only the long-form grid
    # was ever built -- a single-token grid captioned "judged accuracy" names a
    # judge that never ran on it
    metric = next(m for e, _, m in EVALS if e == evalmode)
    fig.supylabel(metric[0].upper() + metric[1:], fontsize=fs(10), x=0.012,
                  color=INK)
    fig.savefig(HERE / f"{out}.pdf")
    fig.savefig(HERE / f"{out}.png", dpi=400)
    plt.close(fig)
    return n



# ============================================================================
# bytask -- levels, one panel per task, both evaluation modes
# ============================================================================
# The two ROWS use different metrics (judged vs token/letter match) and are
# never averaged into one number -- they share a y-range so curve SHAPES can be
# compared, not so a point in one equals a point in the other.

RND_LS = (0, (1, 1.6))       # dotted, distinct from both localization styles


def _bytask_panel(ax, acc, models, task, show_y, show_x, with_random=True):
    bare(ax)
    ax.set_xscale("log")
    n_seen = n_rand = 0
    # random first, so it sits behind the two localizations
    if with_random:
        xs, mu, lo, hi, n = curve(acc, models, task, "random")
        if xs:
            n_rand = n
            ax.fill_between(xs, lo, hi, color=CTRL, alpha=0.10, lw=0, zorder=1)
            ax.plot(xs, mu, color=CTRL, lw=1.6, ls=RND_LS, zorder=2,
                    dash_capstyle="round")
            ax.plot(xs, mu, "o", color=CTRL, ms=2.6, mew=0, zorder=2)
    for arm, colour, ls in (("long", LF, "-"), ("single", ST, (0, (4, 2.2)))):
        xs, mu, lo, hi, n = curve(acc, models, task, arm)
        if not xs:
            continue
        n_seen = max(n_seen, n)
        ax.fill_between(xs, lo, hi, color=colour, alpha=0.13, lw=0, zorder=1)
        ax.plot(xs, mu, color=colour, lw=2.0, ls=ls, zorder=3,
                solid_capstyle="round")
        ax.plot(xs, mu, "o", color=colour, ms=3.4, mew=0, zorder=4)
    ax.set_xlim(0.0085, 1.25)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xticks([0.01, 0.1, 1.0])
    ax.set_xticklabels(["1%", "10%", "100%"] if show_x else [], fontsize=fs(8))
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_yticklabels(["0", ".5", "1"] if show_y else [], fontsize=fs(8))
    return n_seen, n_rand


def bytask_figure(out="fig_localization_by_task", models=None, algo_dir="atp",
                  with_random=True):
    """Levels per task x evaluation mode, averaged across models."""
    models = models or GRID_MODELS
    # bias last: the one task where the single-token arm leads
    bt_tasks = [t for t in TASKS if t != "bias"] + (["bias"] if "bias" in TASKS else [])
    nrow, ncol = len(EVALS), len(bt_tasks)
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.05 * ncol + 0.9,
                                                  2.15 * nrow + 0.95),
                             squeeze=False)
    counts, rcounts, peaks = {}, {}, {}
    sparse = [k for k in KS if k <= 0.1]
    any_random = False
    for r, (mode, row_title, row_metric) in enumerate(EVALS):
        acc = harvest(mode, include_random=with_random, algo_dir=algo_dir)
        for c, task in enumerate(bt_tasks):
            ax = axes[r][c]
            counts[(mode, task)], rcounts[(mode, task)] = _bytask_panel(
                ax, acc, models, task, show_y=(c == 0), show_x=(r == nrow - 1),
                with_random=with_random)
            any_random |= rcounts[(mode, task)] > 0
            pk = {}
            for arm in ("long", "single", "random"):
                vals = [max(acc[(m, task, arm)][k] for k in sparse
                            if k in acc[(m, task, arm)])
                        for m in models if acc.get((m, task, arm))]
                pk[arm] = st.mean(vals) if vals else float("nan")
            peaks[(mode, task)] = pk
            if c == 0:
                # pushed further out as the font grows, so the two-line row
                # label does not run into the y tick labels
                ax.text(-0.42 * FONT_SCALE, 0.5, f"{row_title}\n{row_metric}",
                        transform=ax.transAxes, ha="center", va="center",
                        rotation=90, fontsize=fs(9), color=INK,
                        linespacing=1.5)
    # n varies by TASK, so it goes in the column header
    for c, task in enumerate(bt_tasks):
        ns = {counts[(mode, task)] for mode, _, _ in EVALS}
        n_txt = str(ns.pop()) if len(ns) == 1 else "/".join(
            str(counts[(mode, task)]) for mode, _, _ in EVALS)
        rs = {rcounts[(mode, task)] for mode, _, _ in EVALS}
        r_txt = ""
        if with_random and any(rs):
            rv = str(rs.pop()) if len(rs) == 1 else "/".join(
                str(rcounts[(mode, task)]) for mode, _, _ in EVALS)
            if rv != n_txt:          # only spell it out when it differs
                r_txt = f", random $n$={rv}"
        ax = axes[0][c]
        ax.set_title(task.capitalize(), fontsize=fs(9.5), pad=17 * FONT_SCALE)
        ax.text(0.5, 1.035, f"$n$={n_txt}{r_txt}",
                transform=ax.transAxes, ha="center", va="bottom",
                fontsize=fs(7.4), color=INK, alpha=0.62)

    fig.text(0.5, margin(0.085), "steering budget (fraction of blocks steered)",
             ha="center", fontsize=fs(9.5), color=INK)
    handles = [Line2D([], [], color=LF, lw=2.0, ls="-",
                      label="Free-form localization"),
               Line2D([], [], color=ST, lw=2.0, ls=(0, (4, 2.2)),
                      label="Single-token localization")]
    if any_random:
        handles.append(Line2D([], [], color=CTRL, lw=1.6, ls=RND_LS,
                              label="Random baseline"))
    fig.legend(handles, [h.get_label() for h in handles], loc="lower center",
               ncol=len(handles), fontsize=fs(9), bbox_to_anchor=(0.5, 0.005),
               handlelength=2.6, columnspacing=2.4)
    fig.suptitle("Free-form localization's advantage is task- and "
                 "evaluation-dependent", fontsize=fs(12), y=0.995)
    # Caption note: the top row is judged accuracy and the bottom is
    # token/letter match, so the rows are NOT comparable point-for-point.
    fig.subplots_adjust(left=margin(0.105), right=0.99, top=from_top(0.872),
                        bottom=margin(0.175), hspace=0.22 * FONT_SCALE,
                        wspace=0.16 * FONT_SCALE)
    for ext in ("png", "pdf"):
        fig.savefig(HERE / f"{out}.{ext}", dpi=200)
    plt.close(fig)
    return counts, peaks


def main():
    global RESULTS_ROOT, FONT_SCALE
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--eval", default="long", choices=["long", "single"],
                    help="evaluation mode for levels/grid (bytask does both)")
    ap.add_argument("--which", default="levels,bytask",
                    help="comma list: levels,bytask,grid")
    ap.add_argument("--out", default=None, help="basename for levels")
    ap.add_argument("--controls", action="store_true",
                    help="add the random baseline to the levels figure")
    ap.add_argument("--no_random", action="store_true",
                    help="leave the random baseline off the bytask figure")
    ap.add_argument("--models", default=None,
                    help="comma-separated model dirs (default: all five)")
    ap.add_argument("--results_root", default=None,
                    help=f"accuracy tree (default: {RESULTS_ROOT})")
    ap.add_argument("--patch_algo", default="atp",
                    help="localization directory under each from_* cell")
    ap.add_argument("--out_suffix", default="",
                    help="appended to every figure stem")
    ap.add_argument("--font_scale", type=float, default=FONT_SCALE,
                    help=f"multiplier on every font size "
                         f"(default: {FONT_SCALE}; 1.0 = original sizes)")
    a = ap.parse_args()
    FONT_SCALE = a.font_scale
    if a.results_root:
        RESULTS_ROOT = Path(a.results_root).resolve()
    want = [w.strip() for w in a.which.split(",")]
    algo_dir = a.patch_algo
    out = (a.out or f"fig_{a.eval}eval_levels") + a.out_suffix
    fam = style()
    print(f"font: {fam}  (scale {FONT_SCALE:g})")
    print(f"results: {RESULTS_ROOT}  (localization dir '{algo_dir}')")

    models = a.models.split(",") if a.models else GRID_MODELS

    if "levels" in want:
        acc = harvest(a.eval, algo_dir=algo_dir)
        nl, _ = levels_figure(acc, out, a.eval, models=models)
        print(f"wrote {out}.pdf / .png   ({nl} cells averaged)")
        if a.controls:
            acc_ctrl = harvest(a.eval, include_random=True, algo_dir=algo_dir)
            nc, _ = levels_figure(acc_ctrl, out + "_controls", a.eval,
                                  with_controls=True, models=models)
            print(f"wrote {out}_controls.pdf / .png   ({nc} cells averaged)")

    if "bytask" in want:
        bt = f"fig_localization_by_task{a.out_suffix}"
        counts, peaks = bytask_figure(out=bt, models=models, algo_dir=algo_dir,
                                      with_random=not a.no_random)
        print(f"wrote {bt}.pdf / .png")
        print(f"\n{'eval':8s}{'task':16s}{'n':>2s}  {'free-form':>10s}"
              f"{'single-tok':>12s}{'random':>9s}{'diff':>9s}   (peak k<=0.1)")
        for mode, _, _ in EVALS:
            for task in TASKS:
                pk = peaks[(mode, task)]
                print(f"{mode:8s}{task:16s}{counts[(mode, task)]:2d}  "
                      f"{pk['long']:10.3f}{pk['single']:12.3f}"
                      f"{pk['random']:9.3f}{pk['long'] - pk['single']:+9.3f}")

    if "grid" in want:
        acc_ctrl = harvest(a.eval, include_random=True, algo_dir=algo_dir)
        g = f"fig_{a.eval}eval_grid{a.out_suffix}"
        ns = supplement(acc_ctrl, g, a.eval, empty="unable to\ninduce behavior")
        print(f"wrote {g}.pdf / .png   ({ns} paired cells of "
              f"{len(TASKS) * len(GRID_MODELS)})")


if __name__ == "__main__":
    main()