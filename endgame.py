"""
Perfect endgame answers from the free lichess Syzygy tablebase API.

No auth, no disk. Any position with 7 or fewer pieces has an exact answer, so
the engine's heuristic eval is simply replaced by ground truth there.
"""

import chess, requests

UA = {"User-Agent": "chess-training-toolkit/1.0 (+https://github.com/calvinslinde/chess-training)"}
_cache = {}


def piece_count(board):
    return chess.popcount(board.occupied)


def probe(board):
    """{'category','dtz','dtm','best_san','moves'} or None if out of range."""
    if piece_count(board) > 7:
        return None
    fen = board.fen()
    if fen in _cache:
        return _cache[fen]
    try:
        r = requests.get("https://tablebase.lichess.ovh/standard",
                         params={"fen": fen}, headers=UA, timeout=15)
        if r.status_code != 200:
            return None
        d = r.json()
        mv = d.get("moves") or []
        out = {"category": d.get("category"), "dtz": d.get("dtz"), "dtm": d.get("dtm"),
               "best_san": mv[0]["san"] if mv else None,
               "moves": [{"san": m["san"], "category": m["category"], "dtz": m.get("dtz")}
                         for m in mv[:5]]}
        _cache[fen] = out
        return out
    except Exception:
        return None


# category is from the point of view of the side to move
WINNING = {"win", "cursed-win"}
LOSING = {"loss", "blessed-loss"}


def verdict(board):
    p = probe(board)
    if not p:
        return None
    c = p["category"]
    if c in WINNING:
        return ("win", p)
    if c in LOSING:
        return ("loss", p)
    if c == "draw":
        return ("draw", p)
    return None


def classify_endgame_move(board, played):
    """Did this move throw away a tablebase-won or tablebase-drawn position?"""
    before = verdict(board)
    if not before:
        return None
    b2 = board.copy(stack=False)
    b2.push(played)
    after = verdict(b2)
    if not after:
        return None
    # after the move it is the opponent's turn, so flip their result to ours
    flip = {"win": "loss", "loss": "win", "draw": "draw"}
    ours_after = flip[after[0]]
    if before[0] == "win" and ours_after != "win":
        return {"kind": "threw_win", "was": "win", "now": ours_after,
                "best": before[1]["best_san"], "dtz": before[1]["dtz"]}
    if before[0] == "draw" and ours_after == "loss":
        return {"kind": "threw_draw", "was": "draw", "now": "loss",
                "best": before[1]["best_san"], "dtz": before[1]["dtz"]}
    if before[0] == "loss" and ours_after == "draw":
        return {"kind": "saved_draw", "was": "loss", "now": "draw",
                "best": before[1]["best_san"], "dtz": before[1]["dtz"]}
    return None
