#!/usr/bin/env python3
"""
Estimate playing strength two independent ways.

1. Maia band likelihood. Maia networks predict what humans at a given rating
   play. Score every one of your moves under all five bands; the band that
   assigns your moves the highest likelihood is the maximum-likelihood estimate
   of your strength. This reads move *choice*, so it is independent of results
   and of whatever your rating currently says.

2. Elo performance rating, solved properly from results against rated opposition.

A peak matters more than an argmax. If the likelihood curve rises monotonically
to an edge band, the estimate is censored -- a bound, not a point -- and this
says so rather than reporting the edge as if it were a peak.
"""

import argparse, math, os, random, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import chess, chess.pgn
import maia as M


def collect_moves(pgn_path, me, min_ply=17):
    """Every position where the player had a real choice."""
    rows = []
    with open(pgn_path, encoding="utf-8", errors="replace") as fh:
        gi = 0
        while True:
            game = chess.pgn.read_game(fh)
            if game is None:
                break
            gi += 1
            m = (me or "").lower()
            if game.headers.get("White", "").lower() == m:
                side = chess.WHITE
            elif game.headers.get("Black", "").lower() == m:
                side = chess.BLACK
            else:
                continue
            board = game.board()
            for ply, mv in enumerate(game.mainline_moves(), start=1):
                if board.turn == side and board.legal_moves.count() > 1:
                    rows.append({"fen": board.fen(), "uci": mv.uci(),
                                 "ply": ply, "game": gi,
                                 "n_legal": board.legal_moves.count()})
                board.push(mv)
    return [r for r in rows if r["ply"] >= min_ply], rows


def sweep(rows, bands=M.BANDS):
    """Mean log-likelihood and top-1 agreement for each Maia band."""
    engines, out = {}, {}
    try:
        for b in bands:
            engines[b] = M.MaiaPolicy(b)
        per_move = {b: [] for b in bands}
        hits = {b: 0 for b in bands}
        for i, r in enumerate(rows):
            for b in bands:
                pol = engines[b].policy(r["fen"])
                p = max(pol.get(r["uci"], 0.0), 1e-6)
                per_move[b].append(math.log(p))
                if pol and max(pol, key=pol.get) == r["uci"]:
                    hits[b] += 1
            if sys.stderr.isatty():
                sys.stderr.write(f"\r  scoring move {i+1}/{len(rows)}")
                sys.stderr.flush()
        if sys.stderr.isatty():
            sys.stderr.write("\r" + " " * 40 + "\r")
        for b in bands:
            out[b] = {"logl": sum(per_move[b]) / len(per_move[b]),
                      "top1": hits[b] / len(rows),
                      "per_move": per_move[b]}
    finally:
        for e in engines.values():
            e.close()
    return out


def bootstrap_argmax(res, bands, n=2000, seed=7):
    """Resample moves to see how stable the winning band actually is."""
    rnd = random.Random(seed)
    n_moves = len(res[bands[0]]["per_move"])
    counts = {b: 0 for b in bands}
    for _ in range(n):
        idx = [rnd.randrange(n_moves) for _ in range(n_moves)]
        best, best_v = None, -1e18
        for b in bands:
            pm = res[b]["per_move"]
            v = sum(pm[i] for i in idx)
            if v > best_v:
                best, best_v = b, v
        counts[best] += 1
    return {b: counts[b] / n for b in bands}


def perf_rating(opp_ratings, score):
    """Solve sum(expected) == score for the performance rating."""
    if not opp_ratings:
        return None
    lo, hi = 0.0, 4000.0
    for _ in range(80):
        mid = (lo + hi) / 2
        exp = sum(1 / (1 + 10 ** ((r - mid) / 400)) for r in opp_ratings)
        if exp < score:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pgn")
    ap.add_argument("--me", required=True)
    ap.add_argument("--min-ply", type=int, default=17)
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    rows, all_rows = collect_moves(args.pgn, args.me, args.min_ply)
    if not rows:
        sys.exit("No moves found for that player in this PGN.")
    print(f"\n{args.label or args.me}: {len(rows)} decision points "
          f"(of {len(all_rows)} total moves; excluded ply<{args.min_ply} and forced)")

    res = sweep(rows)
    bands = M.BANDS
    best = max(bands, key=lambda b: res[b]["logl"])
    boot = bootstrap_argmax(res, bands)

    print(f"\n  {'Maia band':<12}{'mean logL':>11}{'top-1 match':>13}{'bootstrap':>11}")
    for b in bands:
        mark = "  <--" if b == best else ""
        print(f"  {b:<12}{res[b]['logl']:>11.4f}{res[b]['top1']*100:>12.1f}%"
              f"{boot[b]*100:>10.0f}%{mark}")

    interior = best not in (bands[0], bands[-1])
    lls = [res[b]["logl"] for b in bands]
    spread = max(lls) - min(lls)
    print(f"\n  argmax band: {best}   (spread across bands: {spread:.4f} nats)")
    if interior:
        print("  -> interior peak: the estimate is identified.")
    elif best == bands[0]:
        print(f"  -> peak sits on the LOW edge. Strength is at or below Maia's floor "
              f"({bands[0]} lichess). This is a bound, not a point estimate.")
    else:
        print(f"  -> peak sits on the HIGH edge. Strength is at or above Maia's ceiling "
              f"({bands[-1]} lichess). A bound, not a point estimate.")
    if max(boot.values()) < 0.7:
        print(f"  -> bootstrap is not decisive (top band only {max(boot.values())*100:.0f}%"
              f" of resamples). Treat the band as soft.")
    print()


if __name__ == "__main__":
    main()
