"""One selection/evaluation path for every dataset's matched baseline suite."""
import time
from .base import select_scorer, evaluate_scorer
from .itemknn import ItemKNN
from .randomwalk import P3alpha, RP3beta
from .slim import SLIM
from .bundleknn import BundleKNN
from .ials import IALS
from .combiner import run_linear_combiner


def run_matched_seed(graph_op, freq, eval_mat, vset, tset, hybrid=None, seed=0):
    """Select configurations on validation; evaluate only the winners on test."""
    methods, matrices = {}, {}
    for baseline in (ItemKNN(), P3alpha(), RP3beta(), SLIM(), BundleKNN(), IALS()):
        started = time.monotonic()
        scorer, hp, val_nd = select_scorer(baseline, graph_op, freq, eval_mat, vset)
        if scorer is None:
            methods[baseline.name] = {"skipped": "catalogue exceeds implementation memory/runtime gate"}
            continue
        if baseline.name in ("ItemKNN", "P3alpha", "RP3beta"):
            matrices[baseline.name] = scorer.W
        metrics = evaluate_scorer(scorer, eval_mat, tset)
        methods[baseline.name] = {"hp": hp, "val_ndcg20": val_nd, "test": metrics,
                                  "seconds": time.monotonic() - started}
        print(f"BASELINE seed={seed} {baseline.name} hp={hp} R@20={metrics[20][0]:.6f}", flush=True)
    methods["LinearCombiner"] = run_linear_combiner(matrices, graph_op, eval_mat, vset, tset, details=True)
    if hybrid is not None:
        methods.update(hybrid.select(eval_mat, vset, tset))
    return {"seed": int(seed), "methods": methods}
