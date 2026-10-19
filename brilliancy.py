#!/usr/bin/env python3
"""
Brilliant-move detection by two-engine differential.

chess.com's rule is essentially "best move that sacrifices material." Zaidi &
Guerzhoy (arXiv 2406.11895), training on 820 human-labelled `!!` moves, found a
better signal: a move reads as brilliant when a WEAKER engine rates it worse
than a strong one does. That is what "looks wrong but is right" means
mechanically, and it does not require a sacrifice at all.

So: score every candidate with a deep search and a deliberately shallow one. The
gap between the two rankings is the brilliance signal. Maia adds the human half
-- a move nobody at that level would consider.
"""
import chess, chess.engine
import tactics as T

MATE = 10000


def _cp(pov):
    if pov.is_mate():
        return MATE if pov.mate() > 0 else -MATE
    return max(-MATE, min(MATE, pov.score()))


def _eval_after(sf, board, move, limit):
    b = board.copy(stack=False)
    b.push(move)
    if b.is_game_over():
        o = b.outcome()
        if o.winner is None:
            return 0
        return MATE if o.winner != b.turn else -MATE
    info = sf.analyse(b, limit)
    return -_cp(info["score"].pov(b.turn))


def differential(sf, board, move, deep=18, shallow=4):
    """
    How much better the move looks to a deep search than to a shallow one,
    in centipawns, from the mover's point of view.
    """
    d = _eval_after(sf, board, move, chess.engine.Limit(depth=deep))
    s = _eval_after(sf, board, move, chess.engine.Limit(depth=shallow))
    return {"deep": d, "shallow": s, "differential": d - s}


def assess(sf, board, move, maia=None, deep=18, shallow=4,
           multipv=4, is_best_required=True):
    """
    Returns a brilliance verdict for one move.

    brilliant  -- deep search likes it, shallow search does not, humans at that
                  level do not find it, and the position is not already resolved
    great      -- the only move that holds, and hard to see
    """
    n_legal = board.legal_moves.count()
    if n_legal < 4 or board.is_check():
        return None
    infos = sf.analyse(board, chess.engine.Limit(depth=deep), multipv=min(multipv, n_legal))
    if isinstance(infos, dict):
        infos = [infos]
    best_mv = infos[0]["pv"][0] if infos[0].get("pv") else None
    best_cp = _cp(infos[0]["score"].pov(board.turn))
    if is_best_required and best_mv is not None and move != best_mv:
        played_deep = _eval_after(sf, board, move, chess.engine.Limit(depth=deep))
        if best_cp - played_deep > 30:
            return None

    diff = differential(sf, board, move, deep, shallow)
    sac = T.see(board, move) <= -150
    p_h = None
    if maia is not None:
        try:
            p_h = maia.policy(board.fen()).get(move.uci(), 0.0)
        except Exception:
            p_h = None

    # already completely winning -> nothing is brilliant
    if abs(best_cp) > 600:
        return None
    verdict = None
    reasons = []
    if diff["differential"] >= 150:
        reasons.append(f"looks {diff['differential']/100:.1f} pawns worse to a shallow search")
    if sac:
        reasons.append(f"gives up material by static exchange ({T.see(board, move)}cp)")
    if p_h is not None and p_h <= 0.10:
        reasons.append(f"only {p_h*100:.0f}% of players at your level would consider it")

    strong = diff["differential"] >= 150 or sac
    hidden = (p_h is None) or (p_h <= 0.15)
    if strong and hidden and diff["deep"] >= -30:
        verdict = "brilliant"
    elif diff["differential"] >= 80 and hidden:
        verdict = "hard to see, and correct"

    if verdict is None:
        return None
    return {"verdict": verdict, "differential": diff["differential"],
            "deep": diff["deep"], "shallow": diff["shallow"],
            "sacrifice": sac, "p_human": p_h, "reasons": reasons}
