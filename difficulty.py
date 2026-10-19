#!/usr/bin/env python3
"""
How hard is this position FOR THIS PLAYER?

My first attempt combined delta, depth-turnover and entropy into a weighted
score, and it rated the starting position "genuinely hard" -- because a small
best-vs-second gap means the choice does not MATTER, not that it is difficult.
Wrong concept.

The right definition is direct and needs no weights: difficulty is the
centipawn loss a player at that rating is expected to incur here.

    expected_loss = sum over legal moves of  P_maia(move | rating) * loss(move)

Maia supplies the human move distribution at the player's level; the engine
supplies each move's cost. High expected loss = a position that punishes players
at that level. This is the quantity Anderson/Kleinberg/Mullainathan (KDD 2016)
found predicts human error better than skill or time.
"""
import math, chess, chess.engine

MATE = 10000


def _cp(pov):
    if pov.is_mate():
        return MATE if pov.mate() > 0 else -MATE
    return max(-MATE, min(MATE, pov.score()))


def expected_human_loss(board, sf, maia, limit, multipv=12, infos=None):
    """
    Returns dict with:
      expected_loss  -- mean centipawn loss under the human policy at that rating
      coverage       -- share of human policy mass the engine actually priced
      p_best         -- chance a player at that level finds the engine's move
      delta          -- best minus second-best (Guid & Bratko's decisiveness)
      n_good         -- engine moves within 50cp of best
      policy         -- the Maia distribution itself, {uci: p}
      loss           -- the engine's price of each priced move, {uci: cp lost}

    `infos` lets a caller that has already run the multi-PV analysis hand it
    in rather than pay for it twice (core/review.py needs the same analysis
    for the best line and the price of the move actually played). Same
    semantics as before when omitted.
    """
    n_legal = board.legal_moves.count()
    if n_legal == 0:
        return None
    if infos is None:
        infos = sf.analyse(board, limit, multipv=min(multipv, n_legal))
    if isinstance(infos, dict):
        infos = [infos]
    scored = []
    for inf in infos:
        pv = inf.get("pv")
        if not pv:
            continue
        scored.append((pv[0].uci(), _cp(inf["score"].pov(board.turn))))
    if not scored:
        return None
    best = scored[0][1]
    loss = {u: max(0, best - c) for u, c in scored}
    worst_priced = max(loss.values()) if loss else 0

    try:
        policy = maia.policy(board.fen())
    except Exception:
        policy = {}
    if not policy:
        return None

    covered = sum(p for u, p in policy.items() if u in loss)
    total_p = sum(policy.values()) or 1.0

    # Expected loss over the mass the engine actually priced, renormalised.
    # Unpriced policy mass is left out rather than charged at the worst priced
    # move's loss: in lopsided positions the multi-PV spread is huge and most
    # human mass sits outside it, and charging it would make accurate moves
    # score "harder" than blunders. Losses are also capped: once a move is
    # 300cp worse the position is decided, and extra centipawns say nothing
    # further about how hard the choice was.
    CAP = 300
    num = sum(p * min(loss[u], CAP) for u, p in policy.items() if u in loss)
    exp = (num / covered) if covered > 0 else None

    return {"expected_loss": exp,
            "coverage": covered / total_p,
            "p_best": policy.get(scored[0][0], 0.0),
            "delta": (scored[0][1] - scored[1][1]) if len(scored) > 1 else None,
            "n_good": sum(1 for _, c in scored if best - c <= 50),
            "n_legal": n_legal,
            "policy": policy, "loss": loss}


def band(expected_loss):
    """Plain-language bucket. Thresholds are a stated prior, not fitted."""
    if expected_loss is None:
        return ""
    if expected_loss >= 120:
        return "brutal for your level"
    if expected_loss >= 70:
        return "hard for your level"
    if expected_loss >= 35:
        return "moderate"
    if expected_loss >= 15:
        return "fairly easy"
    return "easy"


def depth_turnover(engine, board, max_depth=18, min_depth=6):
    """Supplementary: how often the engine changes its mind as depth grows."""
    changes, last, seen = 0, None, 0
    try:
        with engine.analysis(board, chess.engine.Limit(depth=max_depth)) as an:
            for info in an:
                d, pv = info.get("depth"), info.get("pv")
                if not d or not pv or d < min_depth:
                    continue
                seen += 1
                if last is not None and pv[0] != last:
                    changes += 1
                last = pv[0]
    except Exception:
        return None
    return {"changes": changes, "depths_seen": seen}
