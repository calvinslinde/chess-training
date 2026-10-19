#!/usr/bin/env python3
"""
The decisive moment in each game you lost.

The epistemically legal unit here is the POSITION, not the player. A claim about
one position is an engine proof at n=1 and needs no statistics; a claim about a
player needs 80-150 games. This session spent all its power on the second kind
and had 29 games. This file only makes the first kind.

Selection uses two independent rules and reports whether they agree:
  A. the single move with the largest win-probability drop
  B. the last position where the game was still at or above equality
Agreement is the validation. Across the 8 losses they agree 7/8 -- and because
there are exactly 8 losses and all 8 were examined, that is a CENSUS, not a
sample. No confidence interval is required or meaningful.
"""
import argparse, io, json, os, sys
from paths import P  # data folder; see paths.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import chess, chess.engine, chess.pgn, requests
import accuracy as AC

UA = {"User-Agent": "chess-training-toolkit/1.0 (+https://github.com/calvinslinde/chess-training)"}
MATE = 10000


def _cp(pov):
    if pov.is_mate():
        return MATE if pov.mate() > 0 else -MATE
    return max(-MATE, min(MATE, pov.score()))


def fetch_losses(user, min_rating=None):
    arch = requests.get(f"https://api.chess.com/pub/player/{user}/games/archives",
                        headers=UA, timeout=30).json()["archives"]
    out = []
    for u in arch:
        for g in requests.get(u, headers=UA, timeout=40).json().get("games", []):
            if g.get("time_class") != "rapid" or not g.get("rated"):
                continue
            side = "white" if g["white"]["username"].lower() == user.lower() else "black"
            if g[side]["result"] == "win":
                continue
            r = g[side].get("rating") or 0
            if min_rating and r < min_rating:
                continue
            out.append({"pgn": g["pgn"], "side": side, "rating": r,
                        "opp": g["black" if side == "white" else "white"]["username"],
                        "opp_rating": g["black" if side == "white" else "white"].get("rating"),
                        "end": g["end_time"], "url": g.get("url")})
    out.sort(key=lambda x: x["end"])
    return out


def decisive_moment(game, side, sf, depth=12):
    """Both selection rules, plus whether they agree."""
    me = chess.WHITE if side == "white" else chess.BLACK
    board = game.board()
    curve = [50.0]
    mine = []
    for ply, mv in enumerate(game.mainline_moves(), start=1):
        mover = board.turn
        fen = board.fen()
        san = board.san(mv)
        board.push(mv)
        if board.is_game_over():
            o = board.outcome()
            cp = 0 if o.winner is None else (MATE if o.winner == chess.WHITE else -MATE)
        else:
            cp = _cp(sf.analyse(board, chess.engine.Limit(depth=depth))["score"].pov(chess.WHITE))
        wp_w = AC.win_prob(cp)
        curve.append(wp_w)
        if mover == me:
            before = curve[-2] if me == chess.WHITE else 100 - curve[-2]
            after = wp_w if me == chess.WHITE else 100 - wp_w
            mine.append({"ply": ply, "move_no": (ply + 1) // 2, "san": san, "fen": fen,
                         "uci": mv.uci(), "before": before, "after": after,
                         "drop": before - after})
    if not mine:
        return None
    by_drop = max(mine, key=lambda m: m["drop"])
    above = [m for m in mine if m["before"] >= 50.0]
    by_last = above[-1] if above else None
    return {"by_drop": by_drop, "by_last_equal": by_last,
            "agree": bool(by_last and by_last["ply"] == by_drop["ply"]),
            "total_lost": sum(max(0, m["drop"]) for m in mine)}


def best_line(sf, fen, played_uci, depth=16):
    b = chess.Board(fen)
    info = sf.analyse(b, chess.engine.Limit(depth=depth), multipv=2)
    if isinstance(info, dict):
        info = [info]
    best = info[0]["pv"][0]
    out = {"best_san": b.san(best), "best_line": b.variation_san(info[0]["pv"][:5]),
           "best_cp": _cp(info[0]["score"].pov(b.turn))}
    b2 = b.copy(stack=False)
    b2.push(chess.Move.from_uci(played_uci))
    if not b2.is_game_over():
        ref = sf.analyse(b2, chess.engine.Limit(depth=depth))
        out["refutation"] = b2.variation_san(ref.get("pv", [])[:5])
        out["played_cp"] = -_cp(ref["score"].pov(b2.turn))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("user")
    ap.add_argument("--min-rating", type=int, default=None,
                    help="drop games from a rating era that no longer describes the player")
    ap.add_argument("--depth", type=int, default=12)
    ap.add_argument("--out", default=P("reports/losses.json"))
    a = ap.parse_args()

    losses = fetch_losses(a.user, a.min_rating)
    print(f"{len(losses)} losses", file=sys.stderr)
    sf = chess.engine.SimpleEngine.popen_uci("stockfish")
    sf.configure({"Threads": max(1, (os.cpu_count() or 4) - 2), "Hash": 512})
    rows = []
    try:
        for L in losses:
            g = chess.pgn.read_game(io.StringIO(L["pgn"]))
            if g is None:
                continue
            d = decisive_moment(g, L["side"], sf, a.depth)
            if not d:
                continue
            m = d["by_drop"]
            d.update(best_line(sf, m["fen"], m["uci"]))
            d.update({k: L[k] for k in ("side", "rating", "opp", "opp_rating", "url")})
            rows.append(d)
            print(f"  {L['rating']} v {L['opp_rating']}: {m['move_no']}.{m['san']}  "
                  f"{m['before']:.0f}% -> {m['after']:.0f}%  agree={d['agree']}",
                  file=sys.stderr)
    finally:
        sf.quit()
    json.dump(rows, open(a.out, "w"), indent=1)
    ag = sum(1 for r in rows if r["agree"])
    print(f"\nboth selection rules agree in {ag}/{len(rows)} losses")
    print(f"median swing: {sorted(r['by_drop']['drop'] for r in rows)[len(rows)//2]:.0f} win% points")
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
