"""
LEVEL 6 -- genetic-programming alpha miner.

Levels 1-4 test factors WE defined. This level evolves NEW formulas
(WorldQuant "101 alphas" style) by combining raw factors with arithmetic
and rank operators, e.g.:

    rank(gap_pct) * sub(rvol_30m, iv_pctile)
    div(dist_vwap, add(atr_pct, 0.5))

Search:  tournament-selection GP over expression trees (max depth 4),
         fitness = |Spearman rank IC| on the TRAINING window only,
         with a parsimony penalty so simpler formulas win ties.
Honesty: the top formulas are then (a) re-scored on the untouched OOS
         window and (b) sent through the Level-5 validation gauntlet by
         the caller. Evolution NEVER sees OOS data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sps

from config import (GP_GENERATIONS, GP_MAX_DEPTH, GP_POPULATION, GP_SEED,
                    GP_TOP_KEEP, TRAIN_END)
from stats import train_test_split_by_date

# ---------------------------------------------------------------- operators
UNARY = {
    "neg": lambda a: -a,
    "abs": lambda a: a.abs(),
    "rank": lambda a: a.rank(pct=True),
    "sign": lambda a: np.sign(a),
    "log1p_abs": lambda a: np.log1p(a.abs()) * np.sign(a),
}
BINARY = {
    "add": lambda a, b: a + b,
    "sub": lambda a, b: a - b,
    "mul": lambda a, b: a * b,
    "div": lambda a, b: a / b.replace(0, np.nan),
    "max": lambda a, b: pd.concat([a, b], axis=1).max(axis=1),
    "min": lambda a, b: pd.concat([a, b], axis=1).min(axis=1),
}


class Node:
    __slots__ = ("op", "kids", "leaf")

    def __init__(self, op=None, kids=None, leaf=None):
        self.op, self.kids, self.leaf = op, kids or [], leaf

    def depth(self) -> int:
        return 1 + (max((k.depth() for k in self.kids), default=0))

    def size(self) -> int:
        return 1 + sum(k.size() for k in self.kids)

    def evaluate(self, df: pd.DataFrame) -> pd.Series:
        if self.leaf is not None:
            return df[self.leaf]
        if self.op in UNARY:
            return UNARY[self.op](self.kids[0].evaluate(df))
        return BINARY[self.op](self.kids[0].evaluate(df),
                               self.kids[1].evaluate(df))

    def __repr__(self):
        if self.leaf is not None:
            return self.leaf
        return f"{self.op}({', '.join(map(repr, self.kids))})"


def _random_tree(rng, cols, depth):
    if depth <= 1 or rng.random() < 0.3:
        return Node(leaf=cols[rng.integers(len(cols))])
    if rng.random() < 0.35:
        op = list(UNARY)[rng.integers(len(UNARY))]
        return Node(op=op, kids=[_random_tree(rng, cols, depth - 1)])
    op = list(BINARY)[rng.integers(len(BINARY))]
    return Node(op=op, kids=[_random_tree(rng, cols, depth - 1),
                             _random_tree(rng, cols, depth - 1)])


def _all_nodes(tree):
    yield tree
    for k in tree.kids:
        yield from _all_nodes(k)


def _clone(tree):
    if tree.leaf is not None:
        return Node(leaf=tree.leaf)
    return Node(op=tree.op, kids=[_clone(k) for k in tree.kids])


def _crossover(rng, a, b, max_depth):
    child = _clone(a)
    nodes = list(_all_nodes(child))
    target = nodes[rng.integers(len(nodes))]
    donor_nodes = list(_all_nodes(b))
    donor = _clone(donor_nodes[rng.integers(len(donor_nodes))])
    target.op, target.kids, target.leaf = donor.op, donor.kids, donor.leaf
    return child if child.depth() <= max_depth else _clone(a)


def _mutate(rng, tree, cols, max_depth):
    child = _clone(tree)
    nodes = list(_all_nodes(child))
    target = nodes[rng.integers(len(nodes))]
    repl = _random_tree(rng, cols, max(1, max_depth - 1))
    target.op, target.kids, target.leaf = repl.op, repl.kids, repl.leaf
    return child if child.depth() <= max_depth else _clone(tree)


def _fitness(tree, df, y) -> float:
    try:
        sig = tree.evaluate(df)
    except Exception:
        return -np.inf
    sig = sig.replace([np.inf, -np.inf], np.nan)
    ok = sig.notna() & y.notna()
    if ok.sum() < 200 or sig[ok].nunique() < 10:
        return -np.inf
    ic, _ = sps.spearmanr(sig[ok], y[ok])
    if np.isnan(ic):
        return -np.inf
    return abs(ic) - 0.002 * tree.size()  # parsimony pressure


def evolve_alphas(panel: pd.DataFrame, target: str, factor_cols: list[str],
                  population: int = GP_POPULATION,
                  generations: int = GP_GENERATIONS,
                  max_depth: int = GP_MAX_DEPTH,
                  seed: int = GP_SEED) -> pd.DataFrame:
    """Evolve formulas on the training window, report train + OOS rank IC.
    Returns a DataFrame sorted by |OOS IC| with the formula strings."""
    rng = np.random.default_rng(seed)
    train, test = train_test_split_by_date(panel.dropna(subset=[target]), TRAIN_END)
    if len(train) < 500 or len(test) < 200:
        return pd.DataFrame()
    # keep only well-covered numeric factors
    cols = [c for c in factor_cols
            if train[c].notna().mean() > 0.7 and train[c].nunique() > 10]
    if len(cols) < 3:
        return pd.DataFrame()
    ytr, yte = train[target], test[target]

    pop = [_random_tree(rng, cols, max_depth) for _ in range(population)]
    fits = np.array([_fitness(t, train, ytr) for t in pop])

    for _ in range(generations):
        new_pop = []
        # elitism: carry the best 10% forward untouched
        elite_idx = np.argsort(fits)[-max(2, population // 10):]
        new_pop.extend(_clone(pop[i]) for i in elite_idx)
        while len(new_pop) < population:
            # tournament selection (size 4)
            cand = rng.integers(len(pop), size=4)
            a = pop[cand[np.argmax(fits[cand])]]
            cand = rng.integers(len(pop), size=4)
            b = pop[cand[np.argmax(fits[cand])]]
            child = _crossover(rng, a, b, max_depth) if rng.random() < 0.7 \
                else _mutate(rng, a, cols, max_depth)
            new_pop.append(child)
        pop = new_pop
        fits = np.array([_fitness(t, train, ytr) for t in pop])

    # dedupe by formula string, then score OOS
    seen, rows = set(), []
    for i in np.argsort(fits)[::-1]:
        if fits[i] == -np.inf:
            break
        formula = repr(pop[i])
        if formula in seen:
            continue
        seen.add(formula)
        try:
            sig_te = pop[i].evaluate(test).replace([np.inf, -np.inf], np.nan)
        except Exception:
            continue
        ok = sig_te.notna() & yte.notna()
        if ok.sum() < 100:
            continue
        oos_ic, oos_p = sps.spearmanr(sig_te[ok], yte[ok])
        # NOTE: fitness used |IC|, so recover the SIGNED train IC for the
        # same-sign check (an alpha whose IC flips sign OOS is not confirmed)
        sig_tr = pop[i].evaluate(train).replace([np.inf, -np.inf], np.nan)
        ok_tr = sig_tr.notna() & ytr.notna()
        train_ic_signed, _ = sps.spearmanr(sig_tr[ok_tr], ytr[ok_tr]) \
            if ok_tr.sum() >= 100 else (np.nan, np.nan)
        same_sign = bool(np.sign(oos_ic) == np.sign(train_ic_signed)) \
            if not np.isnan(train_ic_signed) else False
        rows.append({
            "formula": formula,
            "train_ic": round(float(train_ic_signed), 4)
                if not np.isnan(train_ic_signed) else np.nan,
            "oos_ic": round(float(oos_ic), 4),
            "oos_p": float(oos_p),
            "oos_n": int(ok.sum()),
            "same_sign": same_sign,
            "confirmed": bool(same_sign and abs(oos_ic) >= 0.02
                              and oos_p < 0.05),
            "tree": pop[i],
        })
        if len(rows) >= GP_TOP_KEEP * 3:
            break
    if not rows:
        return pd.DataFrame()
    res = pd.DataFrame(rows)
    return res.reindex(res["oos_ic"].abs().sort_values(ascending=False).index) \
        .head(GP_TOP_KEEP).reset_index(drop=True)


def alpha_top_decile_mask(tree: Node, panel: pd.DataFrame,
                          oos_ic_sign: float) -> pd.Series:
    """Boolean mask of events in the alpha's favourable decile, for handing
    the evolved alpha to the Level-5 gauntlet as a tradeable rule."""
    sig = tree.evaluate(panel).replace([np.inf, -np.inf], np.nan)
    pct = sig.rank(pct=True)
    return pct >= 0.9 if oos_ic_sign >= 0 else pct <= 0.1
