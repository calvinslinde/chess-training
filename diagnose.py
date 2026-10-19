#!/usr/bin/env python3
"""
Combined error diagnosis.

Two motif libraries disagreed on 10.Ba3, and the disagreement was a real
distinction rather than a bug in either. `motifs.py` asks "did this move CREATE
a motif" -- correct, and it returns nothing here, because the c3 knight was
already loose and the engine's move (Bd2) simply defended it. `tactics.py` asks
"is material hanging in the position you reached" -- which is what actually cost
the game.

A player needs both: what your move created, and what it failed to address.
"""
import chess
import tactics as T
import motifs as MO

SUPPRESS = {"x_ray", "attacks_loose_piece"}   # over-fire; keep as features, not output


def diagnose(board, move, pv=None, min_gain=150):
    """
    Returns {"created": [...], "unaddressed": [...], "summary": str}
    """
    created = [m for m in MO.detect_all(board, move, pv=pv)
               if m["motif"] not in SUPPRESS]

    after = board.copy(stack=False)
    after.push(move)
    us = board.turn
    unaddressed = []
    for sq, pt, gain in T.hanging_pieces(after, us):
        if gain >= min_gain:
            # Was it already hanging BEFORE the move?
            #
            # hanging_pieces validates each capture against board.legal_moves, so
            # on the pre-move board -- where it is still our turn -- the opponent's
            # captures are not legal and nothing is ever found. Flip the side to
            # move first, or every standing hang is misreported as one this move
            # created.
            probe = board.copy(stack=False)
            probe.turn = not us
            try:
                pre = {s for s, _, g in T.hanging_pieces(probe, us) if g >= min_gain}
            except Exception:
                pre = set()
            unaddressed.append({
                "motif": "standing_hang" if sq in pre else "created_hang",
                "square": chess.square_name(sq),
                "piece": chess.piece_name(pt), "gain": gain,
                "detail": (f"your {chess.piece_name(pt)} on {chess.square_name(sq)} "
                           f"can be won for {gain/100:.1f} pawns"
                           + (" and was already loose before this move"
                              if sq in pre else " -- this move put it there"))})

    bits = []
    for u in unaddressed[:2]:
        bits.append(u["detail"])
    for c in created[:2]:
        bits.append(c.get("detail") or c["motif"].replace("_", " "))
    return {"created": created, "unaddressed": unaddressed,
            "summary": "; ".join(bits) if bits else ""}
