"""The matched baseline suite: select each baseline on validation, evaluate on test.

Used by the (seed, gt)-pair evaluators (``eval/bundlerec.py``, ``eval/dunnhumby.py``) via
``run_matched_seed``. Every baseline is evaluated on the SAME splits/candidates/metrics as
BundleFuse; selection is strict-greater on validation NDCG@20 (first maximiser wins).
"""
from .base import _seeds
from .itemknn import ItemKNN
from .randomwalk import P3alpha, RP3beta
from .slim import SLIM
from .bundleknn import BundleKNN
from .ials import IALS
from .combiner import run_linear_combiner

# evaluation order (matches the reported tables)
INDEPENDENT = [ItemKNN(), P3alpha(), RP3beta()]   # item--item; their W's feed the combiner
QUERY_DEP = [BundleKNN(), IALS()]


def _select(bl, graph_op, freq, eval_mat, vset, tset):
    """Grid-select ``bl`` on validation NDCG@20, return (best_scorer, best_hp, test_metrics)."""
    best_hp, best_nd, best_scorer = None, -1.0, None
    for hp in bl.grid:
        scorer = bl.make_scorer(graph_op, freq, hp)
        if scorer is None:
            return None, None, None            # gated out (e.g. SLIM at large n)
        nd = eval_mat(scorer(_seeds(vset)), vset, Ks=(10, 20))[20][1]
        if nd > best_nd:
            best_nd, best_hp, best_scorer = nd, hp, scorer
    return best_scorer, best_hp, eval_mat(best_scorer(_seeds(tset)), tset, Ks=(10, 20))


def run_matched_seed(graph_op, freq, eval_mat, vset, tset):
    """One seed of the matched suite. Returns ``{method: {K: (recall, ndcg)}}`` for this seed."""
    res, comp = {}, {}
    for bl in INDEPENDENT:
        scorer, _, m = _select(bl, graph_op, freq, eval_mat, vset, tset)
        comp[bl.name] = getattr(scorer, "W", None)
        res[bl.name] = m
    scorer, _, m = _select(SLIM(), graph_op, freq, eval_mat, vset, tset)
    if scorer is not None:
        res["SLIM"] = m
    for bl in QUERY_DEP:
        _, _, m = _select(bl, graph_op, freq, eval_mat, vset, tset)
        res[bl.name] = m
    res["LinearCombiner"] = run_linear_combiner(comp, graph_op, eval_mat, vset, tset)
    return res
