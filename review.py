#!/usr/bin/env python3
"""
Game review, second generation.

Fixes over v1: accuracy uses Lichess's actual volatility-weighted + harmonic
method rather than an arithmetic mean; positional errors are decomposed into
named features instead of a "drift" bucket; good moves are recognised, not just
bad ones; openings are identified from a real database; endgames use tablebase
ground truth; and every error is priced in expected points at your own level via
simulation rather than in centipawns.

Two passes: a fast sweep for the eval curve, then deep multi-PV analysis only on
the positions that turned out to matter.
"""

import argparse, json, math, os, statistics, sys
from paths import P  # data folder; see paths.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import chess, chess.engine, chess.pgn
import accuracy as AC, tactics as T, features as FT, classify as CL
import book as BK, endgame as EG, maia as M, playout as PO

MATE_CP = AC.MATE_CP


def cp_of(pov):
    if pov.is_mate():
        return MATE_CP if pov.mate() > 0 else -MATE_CP
    return max(-MATE_CP, min(MATE_CP, pov.score()))


def san_line(board, moves, n=6):
    try:
        return board.variation_san(moves[:n])
    except Exception:
        return ""


def _acpl(evals, side):
    """
    Average centipawn loss from the single full-game eval curve.

    Mixing fast-pass and deep-pass evals in one average made this swing by 30
    between identical runs; one consistent source makes it reproducible.
    """
    losses = []
    for i in range(len(evals) - 1):
        mover_white = (i % 2 == 0)
        if mover_white != (side == chess.WHITE):
            continue
        before = evals[i] if side == chess.WHITE else -evals[i]
        after = evals[i + 1] if side == chess.WHITE else -evals[i + 1]
        losses.append(max(0, min(1000, before - after)))
    return round(sum(losses) / len(losses)) if losses else 0


def review_game(game, sf, me, opts):
    m = (me or "").lower()
    if game.headers.get("White", "").lower() == m:
        side = chess.WHITE
    elif game.headers.get("Black", "").lower() == m:
        side = chess.BLACK
    else:
        return None

    nodes = list(game.mainline())
    moves = [n.move for n in nodes]
    if not moves:
        return None

    opening = BK.identify(game)
    book_plies = opening["book_plies"]

    # ---- pass 1: fast eval curve over the whole game
    fast = chess.engine.Limit(depth=opts.fast_depth)
    board = game.board()
    evals = [0]                                  # white-perspective cp before ply 1
    for mv in moves:
        board.push(mv)
        if board.is_game_over():
            o = board.outcome()
            evals.append(0 if o.winner is None else
                         (MATE_CP if o.winner == chess.WHITE else -MATE_CP))
            break
        info = sf.analyse(board, fast)
        evals.append(cp_of(info["score"].pov(chess.WHITE)))
    wps = [AC.win_prob(c) for c in evals]

    # ---- which plies deserve a real look
    critical = set()
    for i, mv in enumerate(moves):
        if i + 1 >= len(wps):
            break
        mover_white = (i % 2 == 0)
        wb = wps[i] if mover_white else 100 - wps[i]
        wa = wps[i + 1] if mover_white else 100 - wps[i + 1]
        if wb - wa >= opts.flag_drop or i < 2 * opts.deep_opening:
            critical.add(i)
        if abs(evals[i]) < 120:                   # balanced positions are where games turn
            critical.add(i)

    # ---- pass 2: deep multi-PV on those
    deep = chess.engine.Limit(time=opts.deep_time)
    board = game.board()
    out_moves = []
    for i, mv in enumerate(moves):
        mover = board.turn
        san = board.san(mv)
        fen = board.fen()
        is_mine = (mover == side)
        clk = None
        cm = nodes[i].comment
        if cm:
            import re
            mt = re.search(r"\[%clk\s+(\d+):(\d+):([\d.]+)\]", cm)
            if mt:
                h, mi, s = mt.groups()
                clk = int(h) * 3600 + int(mi) * 60 + float(s)

        rec = {"ply": i + 1, "move_no": (i // 2) + 1, "san": san, "uci": mv.uci(),
               "white": mover == chess.WHITE, "mine": is_mine, "fen": fen, "clock": clk}

        wb = wps[i] if mover == chess.WHITE else 100 - wps[i]
        wa = wps[i + 1] if i + 1 < len(wps) else wb
        if mover != chess.WHITE:
            wa = 100 - wps[i + 1] if i + 1 < len(wps) else wb
        drop = max(0.0, wb - wa)
        rec.update({"wp_before": wb, "wp_after": wa, "drop": drop,
                    "cp_before": evals[i] if mover == chess.WHITE else -evals[i],
                    "cp_after": (evals[i + 1] if mover == chess.WHITE else -evals[i + 1])
                                if i + 1 < len(evals) else None})

        if is_mine and i in critical and not board.is_game_over():
            infos = sf.analyse(board, deep, multipv=opts.multipv)
            if isinstance(infos, dict):
                infos = [infos]
            best_mv = infos[0]["pv"][0] if infos[0].get("pv") else None
            cpb = cp_of(infos[0]["score"].pov(mover))
            b2 = board.copy(stack=False)
            b2.push(mv)
            if b2.is_game_over():
                o = b2.outcome()
                cpa = 0 if o.winner is None else (MATE_CP if o.winner == mover else -MATE_CP)
                ref_pv = []
            else:
                ai = sf.analyse(b2, deep)
                cpa = -cp_of(ai["score"].pov(b2.turn))
                ref_pv = ai.get("pv", [])
            d2 = max(0.0, AC.win_prob(cpb) - AC.win_prob(cpa))
            rec.update({"cp_before": cpb, "cp_after": cpa, "drop": d2,
                        "wp_before": AC.win_prob(cpb), "wp_after": AC.win_prob(cpa),
                        "best": board.san(best_mv) if best_mv else "",
                        "best_pv": san_line(board, infos[0].get("pv", [])),
                        "refutation": san_line(b2, ref_pv, 5), "deep": True})

            pol = None
            if opts.maia:
                try:
                    pol = opts.maia.policy(fen)
                except Exception:
                    pol = None
            comp = CL.complexity(board, infos, pol)
            rec["complexity"] = comp
            tb = EG.classify_endgame_move(board, mv) if EG.piece_count(board) <= 7 else None
            rec["tablebase"] = tb
            prev_to = moves[i - 1].to_square if i > 0 else None
            rec["label"] = CL.classify(board, mv, d2, cpb, cpa, infos,
                                       in_book=(i < book_plies), comp=comp, tb=tb,
                                       prev_to=prev_to)

            if d2 >= opts.flag_drop and best_mv is not None:
                best_uci = best_mv.uci()
                rec["tag"], rec["tag_note"] = T.classify_error(
                    fen, mv.uci(), best_uci, ref_pv, cpb, cpa,
                    is_mate_after=(cpa <= -MATE_CP + 1))
                # named positional decomposition, comparing quiescent positions
                played_line = [mv] + list(ref_pv[:5])
                rec["positional"] = FT.explain_lines(board, played_line,
                                                     list(infos[0].get("pv", []))[:6], mover)
                if pol:
                    rec["maia"] = M.findability(pol, best_uci, mv.uci())
                    rec["maia"]["band"] = opts.maia.band
        elif is_mine:
            rec["label"] = CL.BOOK if i < book_plies else (
                CL.BLUNDER if drop >= 20 else CL.MISTAKE if drop >= 10 else
                CL.INACCURACY if drop >= 5 else CL.GOOD if drop >= 2 else CL.EXCELLENT)

        out_moves.append(rec)
        board.push(mv)

    mine = [r for r in out_moves if r["mine"]]
    acc = AC.game_accuracy(wps, side == chess.WHITE)
    counts = {}
    for r in mine:
        counts[r.get("label", "good")] = counts.get(r.get("label", "good"), 0) + 1

    return {"headers": dict(game.headers), "side": "white" if side == chess.WHITE else "black",
            "result": game.headers.get("Result", "*"), "opening": opening,
            "accuracy": round(acc, 1),
            "acpl": _acpl(evals, side),
            "counts": counts, "n_moves": len(mine),
            "eval_curve": [round(w, 1) for w in wps], "moves": out_moves}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pgn")
    ap.add_argument("--me", required=True)
    ap.add_argument("--rating", type=int, default=1500)
    ap.add_argument("--opp-rating", type=int, default=1100)
    ap.add_argument("--fast-depth", type=int, default=14)
    ap.add_argument("--deep-time", type=float, default=1.2)
    ap.add_argument("--multipv", type=int, default=5)
    ap.add_argument("--flag-drop", type=float, default=4.0)
    ap.add_argument("--deep-opening", type=int, default=8)
    ap.add_argument("--playouts", type=int, default=0, help="simulate the top-N errors")
    ap.add_argument("--playout-n", type=int, default=150)
    ap.add_argument("--out", default=P("reports/review.json"))
    args = ap.parse_args()

    sf = chess.engine.SimpleEngine.popen_uci("stockfish")
    sf.configure({"Threads": max(1, (os.cpu_count() or 4) - 2), "Hash": 1024})
    try:
        args.maia = M.MaiaPolicy(args.rating)
    except Exception as e:
        print(f"(no Maia: {e})", file=sys.stderr)
        args.maia = None

    games = []
    try:
        with open(args.pgn, encoding="utf-8", errors="replace") as fh:
            while True:
                g = chess.pgn.read_game(fh)
                if g is None:
                    break
                r = review_game(g, sf, args.me, args)
                if r:
                    print(f"  {r['headers'].get('White')} vs {r['headers'].get('Black')}: "
                          f"acc {r['accuracy']}  acpl {r['acpl']}  {r['opening']['name']}",
                          file=sys.stderr, flush=True)
                    games.append(r)

        if args.playouts and args.maia:
            opp = M.MaiaPolicy(args.opp_rating)
            try:
                flagged = sorted(
                    [(g, mv) for g in games for mv in g["moves"]
                     if mv.get("mine") and mv.get("drop", 0) >= 10 and mv.get("best")],
                    key=lambda gm: -gm[1]["drop"])[:args.playouts]
                for g, mv in flagged:
                    b = chess.Board(mv["fen"])
                    played = chess.Move.from_uci(mv["uci"])
                    best = b.parse_san(mv["best"])
                    me_color = chess.WHITE if g["side"] == "white" else chess.BLACK
                    mv["playout"] = PO.cost_in_points(b, played, best, me_color,
                                                      args.maia, opp, sf, n=args.playout_n)
                    po = mv["playout"]
                    sig = "significant" if po.get("significant") else "NOT significant"
                    print(f"    simulated {mv['move_no']}.{mv['san']}: cost "
                          f"{po.get('cost', 0):+.3f} +/- {po.get('cost_ci95', 0):.3f}  ({sig})",
                          file=sys.stderr, flush=True)
            finally:
                opp.close()
    finally:
        sf.quit()
        if args.maia:
            args.maia.close()

    with open(args.out, "w") as fh:
        json.dump(games, fh, indent=2)
    print(f"\n{len(games)} games -> {args.out}")


if __name__ == "__main__":
    main()
