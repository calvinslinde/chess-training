#!/usr/bin/env python3
"""Full per-game breakdown: every notable moment, opening departure, clock curve."""

import argparse, json, os, statistics, sys
from paths import P  # data folder; see paths.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import chess, chess.engine, chess.pgn, requests
import analyze as A, tactics as T, maia as M

UA = {"User-Agent": "chess-training-toolkit/1.0 (+https://github.com/calvinslinde/chess-training)"}


def book_departure(game, side, max_ply=30, min_games=150):
    """Walk the lichess database until the played move stops being a known move."""
    board = game.board()
    last_known, out = None, []
    for ply, mv in enumerate(game.mainline_moves(), start=1):
        if ply > max_ply:
            break
        try:
            r = requests.get("https://explorer.lichess.ovh/lichess",
                             params={"variant": "standard", "fen": board.fen(),
                                     "speeds": "rapid,blitz", "ratings": "1000,1200,1400",
                                     "moves": "8"}, headers=UA, timeout=15)
            if r.status_code != 200:
                break
            d = r.json()
            total = d.get("white", 0) + d.get("draws", 0) + d.get("black", 0)
            if total < min_games:
                break
            book = {m["uci"]: m for m in d.get("moves", [])}
            san = board.san(mv)
            if mv.uci() in book:
                last_known = (ply, san, total)
            else:
                top = d.get("moves", [])[:3]
                out.append({
                    "ply": ply, "move_no": (ply + 1) // 2,
                    "side": "white" if board.turn == chess.WHITE else "black",
                    "played": san, "db_total": total,
                    "book": [{"san": board.san(chess.Move.from_uci(m["uci"])),
                              "n": m["white"] + m["draws"] + m["black"],
                              "score": round(100 * (m["white"] + 0.5 * m["draws"]) /
                                             max(1, m["white"] + m["draws"] + m["black"]))}
                             for m in top]})
                break
        except Exception:
            break
        board.push(mv)
    return {"left_at": out[0] if out else None, "last_known": last_known}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pgn")
    ap.add_argument("--me", required=True)
    ap.add_argument("--rating", type=int, default=1500)
    ap.add_argument("--time", type=float, default=0.8)
    ap.add_argument("--min-drop", type=float, default=4.0)
    ap.add_argument("--out", default=P("reports/deep.json"))
    ap.add_argument("--no-book", action="store_true")
    args = ap.parse_args()

    sf = chess.engine.SimpleEngine.popen_uci("stockfish")
    sf.configure({"Threads": max(1, (os.cpu_count() or 4) - 2), "Hash": 512})
    try:
        maia_e = M.MaiaPolicy(args.rating)
    except Exception:
        maia_e = None

    games_out = []
    limit = chess.engine.Limit(time=args.time)
    try:
        with open(args.pgn, encoding="utf-8", errors="replace") as fh:
            gi = 0
            while True:
                game = chess.pgn.read_game(fh)
                if game is None:
                    break
                gi += 1
                m = args.me.lower()
                if game.headers.get("White", "").lower() == m:
                    side = chess.WHITE
                elif game.headers.get("Black", "").lower() == m:
                    side = chess.BLACK
                else:
                    continue
                print(f"game {gi}: {game.headers.get('White')} vs {game.headers.get('Black')}",
                      file=sys.stderr)
                rep = A.analyse_game(game, sf, limit, multipv=2)
                if not rep.moves:
                    continue

                flagged = [x for x in rep.moves if x.color == side and x.drop >= args.min_drop]
                for x in flagged:
                    b = chess.Board(x.fen_before)
                    best_uci = ""
                    try:
                        best_uci = b.parse_san(x.best_san).uci() if x.best_san else ""
                    except Exception:
                        pass
                    refut = []
                    if x.refutation:
                        af = b.copy(stack=False)
                        af.push(chess.Move.from_uci(x.uci))
                        try:
                            for s in x.refutation.replace("...", " ").split():
                                if s[0].isdigit():
                                    continue
                                mv2 = af.parse_san(s)
                                refut.append(mv2)
                                af.push(mv2)
                        except Exception:
                            pass
                    x.tag, x.tag_note = T.classify_error(
                        x.fen_before, x.uci, best_uci, refut, x.cp_before, x.cp_after,
                        is_mate_after=(x.cp_after <= -A.MATE_CP + 1))
                    if maia_e and best_uci:
                        try:
                            x.maia = M.findability(maia_e.policy(x.fen_before), best_uci, x.uci)
                        except Exception:
                            x.maia = {}

                mine = rep.side_moves(side)
                times = [x.time_spent for x in mine if x.time_spent is not None]
                games_out.append({
                    "n": gi, "headers": {k: v for k, v in rep.headers.items()},
                    "side": "white" if side == chess.WHITE else "black",
                    "result": rep.result,
                    "acpl": rep.acpl(side), "accuracy": round(rep.accuracy(side), 1),
                    "counts": rep.counts(side),
                    "phase": {k: v for k, v in rep.phase_apl(side).items()}
                             if hasattr(rep, "phase_apl") else
                             {k: list(v) for k, v in rep.phase_acpl(side).items()},
                    "n_moves": len(mine),
                    "clock_end": mine[-1].clock if mine and mine[-1].clock is not None else None,
                    "time_median": round(statistics.median(times), 1) if times else None,
                    "time_max": round(max(times), 1) if times else None,
                    "book": None if args.no_book else book_departure(game, side),
                    "eval_curve": [round(x.wp_after if x.color == chess.WHITE
                                         else 100 - x.wp_after, 1) for x in rep.moves],
                    "errors": [{
                        "move_no": x.move_no, "ply": x.ply, "san": x.san, "fen": x.fen_before,
                        "uci": x.uci, "label": x.label, "tag": x.tag, "note": x.tag_note,
                        "cp_before": x.cp_before, "cp_after": x.cp_after,
                        "drop": round(x.drop, 1), "best": x.best_san, "best_pv": x.best_pv,
                        "refutation": x.refutation, "clock": x.clock,
                        "time_spent": x.time_spent,
                        "maia": {k: v for k, v in (x.maia or {}).items() if k != "top"},
                    } for x in sorted(flagged, key=lambda z: z.ply)],
                })
    finally:
        sf.quit()
        if maia_e:
            maia_e.close()

    with open(args.out, "w") as fh:
        json.dump(games_out, fh, indent=2)
    print(f"\n{len(games_out)} games -> {args.out}")


if __name__ == "__main__":
    main()
