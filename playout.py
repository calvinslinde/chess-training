"""
Counterfactual simulation: what an error actually costs *you*, in points.

Centipawns are the engine's currency, not yours. Losing 300cp matters only if
someone at your level then converts. So from the position, play the game out
with Maia at your rating against Maia at your opponent's, sampling from the
human policy, and adjudicate. The difference between playing your move and the
engine's move is the real price -- in expected points, not pawns.
"""

import random, chess
import accuracy as AC

MATE_CP = 10000


def _sample(policy, legal_ucis, rnd, temp=1.0):
    items = [(u, policy.get(u, 1e-6) ** (1.0 / temp)) for u in legal_ucis]
    tot = sum(w for _, w in items)
    if tot <= 0:
        return rnd.choice(legal_ucis)
    r = rnd.random() * tot
    acc = 0.0
    for u, w in items:
        acc += w
        if acc >= r:
            return u
    return items[-1][0]


def playout(board, me_color, my_maia, opp_maia, sf, plies=24, seed=0, adj_depth=12):
    """One simulated continuation. Returns expected points for `me_color` (0..1)."""
    rnd = random.Random(seed)
    b = board.copy(stack=False)
    for _ in range(plies):
        if b.is_game_over():
            break
        engine = my_maia if b.turn == me_color else opp_maia
        legal = [m.uci() for m in b.legal_moves]
        try:
            pol = engine.policy(b.fen())
        except Exception:
            pol = {}
        b.push(chess.Move.from_uci(_sample(pol, legal, rnd)))

    if b.is_game_over():
        o = b.outcome()
        if o.winner is None:
            return 0.5
        return 1.0 if o.winner == me_color else 0.0
    info = sf.analyse(b, chess.engine.Limit(depth=adj_depth))
    s = info["score"].pov(me_color)
    cp = MATE_CP if (s.is_mate() and s.mate() > 0) else \
         (-MATE_CP if s.is_mate() else s.score())
    return AC.win_prob(cp) / 100.0


def cost_in_points(board, played, best, me_color, my_maia, opp_maia, sf,
                   n=150, plies=24, adj_depth=12):
    """
    Expected score after your move vs after the engine's, simulated at your level.

    Always returns a standard error and confidence interval. A single playout has
    a standard deviation near 0.4, so n=10 gives a 95% interval of roughly +/-0.34
    -- wide enough to span zero. Reporting a bare point estimate from this is how
    you end up quoting noise as a finding.
    """
    import statistics
    import chess.engine  # noqa
    res = {}
    for key, mv in (("played", played), ("best", best)):
        if mv is None:
            continue
        b = board.copy(stack=False)
        if mv not in b.legal_moves:
            continue
        b.push(mv)
        # paired seeds across both branches: common random numbers cut the
        # variance of the *difference*, which is the quantity we actually report
        vals = [playout(b, me_color, my_maia, opp_maia, sf, plies,
                        seed=1000 * i + 7, adj_depth=adj_depth) for i in range(n)]
        res[key] = {"mean": sum(vals) / len(vals),
                    "sd": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
                    "n": len(vals), "_vals": vals}

    if "played" in res and "best" in res:
        a, b_ = res["best"], res["played"]
        diffs = [x - y for x, y in zip(a["_vals"], b_["_vals"])]
        cost = sum(diffs) / len(diffs)
        sd = statistics.pstdev(diffs) if len(diffs) > 1 else 0.0
        se = sd / (len(diffs) ** 0.5)
        res["cost"] = cost
        res["cost_se"] = se
        res["cost_ci95"] = 1.96 * se
        res["significant"] = abs(cost) > 1.96 * se
    for k in ("played", "best"):
        res.get(k, {}).pop("_vals", None)
    return res
