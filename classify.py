"""
Move classification with credit for good play, plus position difficulty.

The old version only ever told you what you did wrong. It also had no notion of
whether a position was hard: a blunder in a position with one saving move and a
blunder in a position where anything wins are not the same mistake.
"""

import math, chess
import tactics as T

MATE_CP = 10000

# labels, worst to best
BLUNDER, MISTAKE, INACCURACY = "blunder", "mistake", "inaccuracy"
GOOD, EXCELLENT, BEST = "good", "excellent", "best"
GREAT, BRILLIANT, BOOK, MISS = "great", "brilliant", "book", "miss"

SYMBOL = {BRILLIANT: "!!", GREAT: "!", BEST: "", EXCELLENT: "", GOOD: "",
          BOOK: "", INACCURACY: "?!", MISTAKE: "?", BLUNDER: "??", MISS: "×"}

ORDER = [BRILLIANT, GREAT, BEST, EXCELLENT, GOOD, BOOK,
         INACCURACY, MISTAKE, MISS, BLUNDER]


def complexity(board, multipv_infos, maia_policy=None, tol=50):
    """
    How hard was this position?
      n_good      -- engine moves within `tol` of the best
      only_move   -- exactly one move holds
      entropy     -- spread of human policy; high means no move suggests itself
      trappiness  -- the natural human move is much worse than the right one
    """
    scores = []
    for ln in multipv_infos:
        s = ln["score"].pov(board.turn)
        scores.append(MATE_CP if s.is_mate() and s.mate() > 0 else
                      (-MATE_CP if s.is_mate() else s.score()))
    if not scores:
        return {}
    best = scores[0]
    n_good = sum(1 for s in scores if best - s <= tol)
    n_legal = board.legal_moves.count()
    quiet = (not board.is_check()) and n_legal >= 8
    out = {"n_good": n_good, "n_legal": n_legal, "quiet": quiet,
           "only_move": quiet and n_good == 1 and len(scores) > 1,
           "spread": best - scores[-1] if len(scores) > 1 else 0}

    if maia_policy:
        ps = [p for p in maia_policy.values() if p > 0]
        ent = -sum(p * math.log(p) for p in ps) if ps else 0.0
        out["entropy"] = ent
        best_uci = multipv_infos[0]["pv"][0].uci() if multipv_infos[0].get("pv") else None
        top_human = max(maia_policy, key=maia_policy.get)
        out["p_best"] = maia_policy.get(best_uci, 0.0)
        out["human_pick"] = top_human
        # a trap: humans are drawn to a move that is much worse than the only good one
        if quiet and best_uci and top_human != best_uci:
            hs = next((s for ln, s in zip(multipv_infos, scores)
                       if ln.get("pv") and ln["pv"][0].uci() == top_human), None)
            if hs is not None:
                out["trappiness"] = max(0, best - hs) * maia_policy.get(top_human, 0)
    return out


def is_sacrifice(board, move, min_value=150):
    """
    A real sacrifice: the move itself loses material by static exchange, or it
    plants the moved piece where the opponent can profitably take it.

    A loose friendly piece after the move is not enough on its own; that would
    make every ordinary recapture look brilliant.
    """
    piece = board.piece_at(move.from_square)
    if piece is None or piece.piece_type == chess.PAWN:
        return False
    if T.see(board, move) <= -min_value:
        return True
    b = board.copy(stack=False)
    b.push(move)
    best_take = 0
    for m in b.legal_moves:
        if m.to_square == move.to_square and b.is_capture(m):
            best_take = max(best_take, T.see(b, m))
    return best_take >= min_value


def is_recapture(board, move, prev_to):
    """Taking back on the square the opponent just moved to."""
    return (prev_to is not None and move.to_square == prev_to
            and board.is_capture(move))


def classify(board, played, drop, cp_before, cp_after, multipv_infos,
             in_book=False, comp=None, tb=None, prev_to=None):
    """One label per move. Order matters -- most specific wins."""
    comp = comp or {}
    best_move = multipv_infos[0]["pv"][0] if multipv_infos and multipv_infos[0].get("pv") else None
    is_best = best_move is not None and played == best_move
    n_legal = board.legal_moves.count()
    forced = board.is_check() or n_legal < 8 or is_recapture(board, played, prev_to)
    p_best = comp.get("p_best")

    if tb and tb.get("kind") == "threw_win":
        return BLUNDER
    if in_book:
        return BOOK

    if is_best or drop < 0.6:
        # Brilliant: a sound sacrifice, actually hard to see, not already crushing.
        if (not forced and is_sacrifice(board, played) and cp_after >= -50
                and cp_before < 500 and (p_best is None or p_best <= 0.40)):
            return BRILLIANT
        # Great: the genuinely only move, in a live position, that your level misses.
        if (not forced and comp.get("only_move") and comp.get("spread", 0) >= 200
                and abs(cp_before) < 400 and (p_best is None or p_best <= 0.35)):
            return GREAT
        return BEST if is_best else EXCELLENT

    # BLUNDER outranks MISS. MISS used to be tested first, so every error made
    # from a winning position was relabelled -- the three most expensive moves in
    # the sample were filed as "miss" and the reported blunder count was wrong.
    if drop >= 20:
        return BLUNDER
    if (cp_before >= 300) and cp_after < cp_before - 250 and drop >= 8:
        return MISS
    if drop >= 10:
        return MISTAKE
    if drop >= 5:
        return INACCURACY
    if drop >= 2:
        return GOOD
    return EXCELLENT


def difficulty_sentence(comp):
    if not comp:
        return ""
    if comp.get("only_move"):
        s = "Only one move held here"
        if comp.get("p_best") is not None:
            s += f", and {comp['p_best']*100:.0f}% of players at your level find it"
        return s + "."
    if comp.get("trappiness", 0) >= 60:
        return ("This is a trap: the natural-looking move is much worse than the right one, "
                "which is why the position punishes intuition.")
    if comp.get("n_good", 0) >= 4:
        return "Several moves were fine here, so the exact choice mattered less than the plan."
    return ""
