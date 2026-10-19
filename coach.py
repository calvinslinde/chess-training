#!/usr/bin/env python3
"""
Chess coach: fetch -> screen -> deep analysis -> cross-game diagnosis.

The point is not to find your blunders. Engines do that and it teaches nothing.
The point is to say which *kind* of error costs you the most, whether it is an
error your own rating band routinely avoids, and what to do about it.
"""

import argparse, collections, datetime, html, json, os, statistics, sys
from paths import P  # data folder; see paths.py

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import chess, chess.engine, chess.pgn, chess.svg
import analyze as A
import tactics as T
import maia as M
import fetch as F


# ------------------------------------------------------------------ screening

def screen(pgn_path, sf, me, depth=13, max_games=40):
    """Cheap pass over every game to decide which ones are worth deep analysis."""
    out = []
    limit = chess.engine.Limit(depth=depth)
    with open(pgn_path, encoding="utf-8", errors="replace") as fh:
        idx = 0
        while idx < max_games:
            game = chess.pgn.read_game(fh)
            if game is None:
                break
            idx += 1
            side = user_color(game, me)
            if side is None:
                continue
            rep = A.analyse_game(game, sf, limit, multipv=1, progress=False)
            if not rep.moves:
                continue
            mine = rep.side_moves(side)
            if not mine:
                continue
            drops = [m.drop for m in mine]
            peak = max((m.wp_before for m in mine), default=50)
            trough = min((m.wp_after for m in mine), default=50)
            result = game.headers.get("Result", "*")
            won = (result == "1-0" and side == chess.WHITE) or \
                  (result == "0-1" and side == chess.BLACK)
            drew = result == "1/2-1/2"

            score = 0.0
            score += 2.0 * sum(1 for d in drops if d >= 20)
            score += 1.0 * sum(1 for d in drops if 10 <= d < 20)
            if peak >= 75 and not won:
                score += 4.0                      # a win was there and did not land
            if peak >= 65 and trough <= 35:
                score += 2.0                      # the game actually changed hands
            if drew and peak >= 70:
                score += 1.5
            out.append({"idx": idx, "game": game, "side": side, "score": score,
                        "acpl": rep.acpl(side), "peak": peak, "trough": trough,
                        "result": result, "won": won,
                        "blunders": sum(1 for d in drops if d >= 20),
                        "screen_rep": rep})
            sys.stderr.write(f"\r  screened {idx} games")
            sys.stderr.flush()
    sys.stderr.write("\r" + " " * 40 + "\r")
    return out


def user_color(game, me):
    if not me:
        return chess.WHITE
    m = me.strip().lower()
    if game.headers.get("White", "").lower() == m:
        return chess.WHITE
    if game.headers.get("Black", "").lower() == m:
        return chess.BLACK
    return None


# ------------------------------------------------------------- deep annotation

def annotate(rep, side, maia_engine, min_drop=6.0):
    """Attach an error class and a findability read to every move that mattered."""
    flagged = [m for m in rep.moves if m.color == side and m.drop >= min_drop]
    for m in flagged:
        board = chess.Board(m.fen_before)
        best_uci = ""
        if m.best_san:
            try:
                best_uci = board.parse_san(m.best_san).uci()
            except Exception:
                pass

        refut_pv = []
        if m.refutation:
            after = board.copy(stack=False)
            after.push(chess.Move.from_uci(m.uci))
            try:
                for san in m.refutation.replace("...", " ").split():
                    if san[0].isdigit():
                        continue
                    mv = after.parse_san(san)
                    refut_pv.append(mv)
                    after.push(mv)
            except Exception:
                pass

        m.tag, m.tag_note = T.classify_error(
            m.fen_before, m.uci, best_uci, refut_pv,
            m.cp_before, m.cp_after, is_mate_after=(m.cp_after <= -A.MATE_CP + 1))

        if maia_engine and best_uci:
            try:
                pol = maia_engine.policy(m.fen_before)
                m.maia = M.findability(pol, best_uci, m.uci)
                m.maia["band"] = maia_engine.band
            except Exception:
                m.maia = {}
    return flagged


def detect_drift(rep, side, window=6, thresh=15.0, single_cap=8.0):
    """Stretches where nothing looks like a blunder but the position rots anyway."""
    mine = [m for m in rep.moves if m.color == side]
    runs = []
    i = 0
    while i + window <= len(mine):
        chunk = mine[i:i + window]
        total = sum(c.drop for c in chunk)
        if total >= thresh and max(c.drop for c in chunk) < single_cap:
            runs.append({"from": chunk[0].move_no, "to": chunk[-1].move_no,
                         "loss": total, "start_fen": chunk[0].fen_before,
                         "plan": chunk[0].best_pv})
            i += window
        else:
            i += 1
    return runs


# -------------------------------------------------------------- cross-game roll-up

def aggregate(games, me):
    """The part that only works with a stack of games: what recurs."""
    agg = {
        "n_games": len(games), "by_tag": collections.Counter(),
        "loss_by_tag": collections.Counter(), "verdicts": collections.Counter(),
        "loss_by_verdict": collections.Counter(),
        "phase": {p: [] for p in ("opening", "middlegame", "endgame")},
        "by_color": {"white": [], "black": []},
        "openings": collections.defaultdict(lambda: {"n": 0, "acpl": [], "pts": 0.0}),
        "clock": {"scramble": [], "normal": []},
        "acpl": [], "accuracy": [], "results": collections.Counter(),
        "drift_runs": 0, "critical": [],
    }
    for g in games:
        rep, side = g["rep"], g["side"]
        agg["acpl"].append(rep.acpl(side))
        agg["accuracy"].append(rep.accuracy(side))
        agg["results"][g["result"]] += 1
        agg["by_color"]["white" if side == chess.WHITE else "black"].append(rep.acpl(side))
        for p, (v, n) in rep.phase_acpl(side).items():
            if v is not None:
                agg["phase"][p].append(v)

        eco = (rep.headers.get("ECO", "") + " " + rep.headers.get("Opening", "")).strip() \
              or rep.headers.get("ECOUrl", "").rsplit("/", 1)[-1].replace("-", " ") or "unknown"
        eco = eco[:60]
        o = agg["openings"][eco]
        o["n"] += 1
        o["acpl"].append(rep.acpl(side))
        r = g["result"]
        o["pts"] += 1.0 if g["won"] else (0.5 if r == "1/2-1/2" else 0.0)

        agg["drift_runs"] += len(g.get("drift", []))

        for m in g["flagged"]:
            if m.tag:
                agg["by_tag"][m.tag] += 1
                agg["loss_by_tag"][m.tag] += m.drop
            v = (m.maia or {}).get("verdict")
            if v:
                agg["verdicts"][v] += 1
                agg["loss_by_verdict"][v] += m.drop
            if m.time_spent is not None and m.clock is not None:
                bucket = "scramble" if m.clock <= 30 else "normal"
                agg["clock"][bucket].append(m.drop)
            agg["critical"].append((g, m))

        for m in rep.side_moves(side):
            if m.clock is not None and m.drop < 6:
                agg["clock"]["scramble" if m.clock <= 30 else "normal"].append(m.drop)

    agg["critical"].sort(key=lambda gm: -gm[1].drop)
    return agg


# ------------------------------------------------------------------- terminal

def bar(frac, width=24, ch="#"):
    n = int(round(frac * width))
    return ch * n + "." * (width - n)


def print_diagnosis(agg, me):
    B, R, D = "\033[1m", "\033[0m", "\033[2m"
    if not sys.stdout.isatty():
        B = R = D = ""
    print(f"\n{B}{'=' * 64}\n DIAGNOSIS — {me} — {agg['n_games']} games\n{'=' * 64}{R}")
    if agg["acpl"]:
        print(f"\n  Average centipawn loss   {statistics.mean(agg['acpl']):.0f}")
        print(f"  Average accuracy         {statistics.mean(agg['accuracy']):.1f}%")
    rec = agg["results"]
    print(f"  Record                   {dict(rec)}")

    tot = sum(agg["loss_by_tag"].values()) or 1
    print(f"\n{B}  Where the points actually go{R}")
    for tag, loss in agg["loss_by_tag"].most_common(8):
        frac = loss / tot
        print(f"    {T.TAXONOMY_LABEL.get(tag, tag):<22}{bar(frac)} {frac*100:4.0f}%"
              f"  {D}({agg['by_tag'][tag]} moves){R}")

    if agg["verdicts"]:
        tv = sum(agg["loss_by_verdict"].values()) or 1
        print(f"\n{B}  Was it findable at your level?{R}   {D}(Maia policy){R}")
        labels = {"should_have_seen": "Inside your range", "findable": "Borderline",
                  "genuinely_hard": "Genuinely hard", "played_best": "Found it"}
        for v, loss in agg["loss_by_verdict"].most_common():
            frac = loss / tv
            print(f"    {labels.get(v, v):<22}{bar(frac)} {frac*100:4.0f}%"
                  f"  {D}({agg['verdicts'][v]} moves){R}")

    print(f"\n{B}  By phase{R}   {D}(mean ACPL){R}")
    for p in ("opening", "middlegame", "endgame"):
        vals = agg["phase"][p]
        if vals:
            print(f"    {p:<22}{statistics.mean(vals):.0f}")

    w, b = agg["by_color"]["white"], agg["by_color"]["black"]
    if w and b:
        print(f"\n{B}  By colour{R}   white {statistics.mean(w):.0f}"
              f"   black {statistics.mean(b):.0f}")

    sc, nm = agg["clock"]["scramble"], agg["clock"]["normal"]
    if len(sc) >= 5 and len(nm) >= 5:
        print(f"\n{B}  Clock{R}   under 30s: {statistics.mean(sc):.1f}%"
              f" avg loss per move   vs   {statistics.mean(nm):.1f}% otherwise")

    ops = [(k, v) for k, v in agg["openings"].items() if v["n"] >= 2]
    if ops:
        print(f"\n{B}  Openings{R}   {D}(n>=2, sorted by score){R}")
        for k, v in sorted(ops, key=lambda kv: kv[1]["pts"] / kv[1]["n"])[:6]:
            print(f"    {k[:44]:<46}{v['pts']/v['n']*100:3.0f}%  "
                  f"acpl {statistics.mean(v['acpl']):.0f}  {D}n={v['n']}{R}")

    if agg["drift_runs"]:
        print(f"\n{B}  Slow drift{R}   {agg['drift_runs']} stretches where the position "
              f"rotted with no single blunder")


# ------------------------------------------------------------------ HTML report

EXTRA_CSS = """
.hl{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:1rem;margin:1.5rem 0}
.hl div{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:1rem}
.hl .n{font-size:1.8rem;font-weight:600;letter-spacing:-.02em;line-height:1.1}
.hl .l{font-size:.75rem;text-transform:uppercase;letter-spacing:.06em;color:var(--muted);margin-top:.3rem}
.barrow{display:grid;grid-template-columns:170px 1fr 90px;gap:.75rem;align-items:center;margin:.45rem 0}
.barout{background:var(--border);border-radius:999px;height:9px;overflow:hidden}
.barin{background:var(--accent);height:100%;border-radius:999px}
.barrow .v{font-size:.85rem;color:var(--muted);text-align:right;font-variant-numeric:tabular-nums}
.lead{font-size:1.05rem;line-height:1.7;background:var(--card);border-left:3px solid var(--accent);
border-radius:0 8px 8px 0;padding:1.1rem 1.4rem;margin:1.5rem 0}
"""


def barrow(label, frac, right):
    return (f"<div class='barrow'><div>{html.escape(label)}</div>"
            f"<div class='barout'><div class='barin' style='width:{frac*100:.1f}%'></div></div>"
            f"<div class='v'>{html.escape(right)}</div></div>")


def render_report(agg, games, me, meta, drill_n=12):
    o = [f"<title>Chess coaching report</title><style>{A.CSS}{EXTRA_CSS}</style><div class='wrap'>"]
    o.append(f"<h1>{html.escape(me)} — coaching report</h1>"
             f"<div class='sub'>{html.escape(meta)}</div>")

    acpl = statistics.mean(agg["acpl"]) if agg["acpl"] else 0
    acc = statistics.mean(agg["accuracy"]) if agg["accuracy"] else 0
    tv = sum(agg["loss_by_verdict"].values()) or 1
    inrange = agg["loss_by_verdict"].get("should_have_seen", 0) / tv
    o.append("<div class='hl'>"
             f"<div><div class='n'>{acpl:.0f}</div><div class='l'>Avg centipawn loss</div></div>"
             f"<div><div class='n'>{acc:.1f}%</div><div class='l'>Avg accuracy</div></div>"
             f"<div><div class='n'>{agg['n_games']}</div><div class='l'>Games analyzed</div></div>"
             f"<div><div class='n'>{inrange*100:.0f}%</div><div class='l'>Loss to moves your level finds</div></div>"
             "</div>")

    top_tag = agg["loss_by_tag"].most_common(1)
    if top_tag:
        tag, _ = top_tag[0]
        tot = sum(agg["loss_by_tag"].values()) or 1
        share = agg["loss_by_tag"][tag] / tot
        o.append(f"<div class='lead'><b>{T.TAXONOMY_LABEL.get(tag, tag)}</b> is your most "
                 f"expensive habit — {share*100:.0f}% of everything you lose. "
                 f"And {inrange*100:.0f}% of your losses were to moves that players at your "
                 f"own rating find routinely, which means the gap is in checking discipline "
                 f"rather than in knowledge.</div>")

    o.append("<h2>Where the points go</h2><div class='card'>")
    tot = sum(agg["loss_by_tag"].values()) or 1
    for tag, loss in agg["loss_by_tag"].most_common(10):
        o.append(barrow(T.TAXONOMY_LABEL.get(tag, tag), loss / tot,
                        f"{loss/tot*100:.0f}% · {agg['by_tag'][tag]} moves"))
    o.append("</div>")

    if agg["verdicts"]:
        labels = {"should_have_seen": "Inside your range", "findable": "Borderline",
                  "genuinely_hard": "Genuinely hard", "played_best": "Found it"}
        o.append("<h2>Findability at your rating</h2><div class='card'>")
        for v, loss in agg["loss_by_verdict"].most_common():
            o.append(barrow(labels.get(v, v), loss / tv,
                            f"{loss/tv*100:.0f}% · {agg['verdicts'][v]} moves"))
        o.append("<p style='color:var(--muted);font-size:.85rem;margin:.9rem 0 0'>"
                 "Measured with Maia, a network trained to predict what human players at a "
                 "given rating actually play. \"Inside your range\" means your own band finds "
                 "that move often — those are the cheapest points to win back.</p></div>")

    o.append("<h2>Phase and colour</h2><div class='scroll'><table><thead><tr>"
             "<th>Split</th><th>Opening</th><th>Middlegame</th><th>Endgame</th>"
             "</tr></thead><tbody><tr><td>Mean ACPL</td>")
    for p in ("opening", "middlegame", "endgame"):
        vals = agg["phase"][p]
        o.append(f"<td>{statistics.mean(vals):.0f}</td>" if vals else "<td>—</td>")
    o.append("</tr></tbody></table></div>")

    ops = [(k, v) for k, v in agg["openings"].items() if v["n"] >= 2]
    if ops:
        o.append("<h2>Openings</h2><div class='scroll'><table><thead><tr><th>Opening</th>"
                 "<th>Games</th><th>Score</th><th>ACPL</th></tr></thead><tbody>")
        for k, v in sorted(ops, key=lambda kv: kv[1]["pts"] / kv[1]["n"])[:10]:
            o.append(f"<tr><td>{html.escape(k)}</td><td>{v['n']}</td>"
                     f"<td>{v['pts']/v['n']*100:.0f}%</td>"
                     f"<td>{statistics.mean(v['acpl']):.0f}</td></tr>")
        o.append("</tbody></table></div>")

    o.append("<h2>Your drill set</h2><p style='color:var(--muted)'>"
             "The positions that cost you the most, each one you to move.</p>")
    for gme, m in agg["critical"][:drill_n]:
        rep = gme["rep"]
        board = chess.Board(m.fen_before)
        svg = chess.svg.board(board, size=300, orientation=m.color, coordinates=False)
        num = f"{m.move_no}.{'' if m.color == chess.WHITE else '..'}"
        opp = rep.headers.get("Black" if m.color == chess.WHITE else "White", "opponent")
        maia = m.maia or {}
        mline = ""
        if maia.get("p_best") is not None:
            mline = (f"<div class='kv'><b>At your level:</b> "
                     f"{maia['p_best']*100:.0f}% find the right move; "
                     f"{maia['p_played']*100:.0f}% play yours</div>")
        clk = (f"<div class='kv'><b>Clock:</b> {m.clock/60:.1f} min left"
               + (f", {m.time_spent:.0f}s spent" if m.time_spent else "") + "</div>") \
              if m.clock is not None else ""
        o.append(
            f"<div class='mom'><div>{svg}</div><div>"
            f"<h3>{num} {html.escape(m.san)}<span class='tag {m.label}'>"
            f"{T.TAXONOMY_LABEL.get(m.tag, m.label)}</span></h3>"
            f"<div class='kv'><b>vs</b> {html.escape(opp)} · "
            f"eval {m.cp_before/100:+.2f} → {m.cp_after/100:+.2f} "
            f"({m.drop:.0f}% win probability lost)</div>"
            f"<div class='kv'><b>Better:</b> <span class='pv'>{html.escape(m.best_pv)}</span></div>"
            + (f"<div class='kv'><b>Punished by:</b> <span class='pv'>"
               f"{html.escape(m.refutation)}</span></div>" if m.refutation else "")
            + mline + clk
            + f"<div class='kv' style='color:var(--muted)'>{html.escape(m.tag_note)}</div>"
            + (f"<div class='kv' style='color:var(--muted)'>{html.escape(maia.get('note',''))}</div>"
               if maia.get("note") else "")
            + "</div></div>")

    o.append("<footer>Stockfish for evaluation; Maia for what a human at your rating "
             "would play. Accuracy and win probability use Lichess's centipawn curves."
             "</footer></div>")
    return "".join(o)


# ------------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser(description="Analyze a stack of your games and diagnose the patterns.")
    ap.add_argument("user", help="chess.com or lichess username")
    ap.add_argument("--site", choices=["chess.com", "lichess"], default="chess.com")
    ap.add_argument("-n", "--fetch", type=int, default=20, help="games to pull")
    ap.add_argument("-k", "--deep", type=int, default=5, help="games to analyze deeply")
    ap.add_argument("--time-class", help="bullet | blitz | rapid | daily")
    ap.add_argument("--color", choices=["white", "black"])
    ap.add_argument("--rating", type=int, help="your rating (picks the Maia band)")
    ap.add_argument("--pgn", help="skip fetching; use this PGN")
    ap.add_argument("--time", type=float, default=0.35, help="seconds/position, deep pass")
    ap.add_argument("--screen-depth", type=int, default=13)
    ap.add_argument("--no-maia", action="store_true")
    ap.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--out")
    args = ap.parse_args()

    if args.pgn:
        pgn_path, meta_games = args.pgn, []
    else:
        print(f"Fetching up to {args.fetch} games for {args.user} from {args.site}...")
        fn = F.chesscom if args.site == "chess.com" else F.lichess
        kw = dict(count=args.fetch, time_class=args.time_class,
                  rated_only=True, color=args.color)
        meta_games = fn(args.user, **kw)
        if not meta_games:
            sys.exit("No games matched.")
        stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
        pgn_path = os.path.expanduser(
            f"$CHESS_HOME/games/{args.user}-{args.time_class or 'all'}-{stamp}.pgn")
        F.write_pgn(meta_games, pgn_path)
        print(f"  {len(meta_games)} games -> {pgn_path}")

    rating = args.rating
    if not rating and meta_games:
        rs = [g["user_rating"] for g in meta_games if g.get("user_rating")]
        if rs:
            rating = int(statistics.mean(rs))
            print(f"  detected rating ~{rating}")

    sf = chess.engine.SimpleEngine.popen_uci("stockfish")
    sf.configure({"Threads": args.threads, "Hash": 512})

    maia_engine = None
    if not args.no_maia:
        try:
            maia_engine = M.MaiaPolicy(rating or 1500)
            print(f"  Maia band: {maia_engine.band}")
        except Exception as e:
            print(f"  (Maia unavailable: {e})", file=sys.stderr)

    try:
        print("\nScreening for the instructive games...")
        screened = screen(pgn_path, sf, args.user, depth=args.screen_depth,
                          max_games=args.fetch)
        if not screened:
            sys.exit(f"No games found where '{args.user}' is a player. "
                     "Check the username spelling.")
        screened.sort(key=lambda g: -g["score"])
        picked = screened[:args.deep]
        print(f"  {len(screened)} games screened; going deep on {len(picked)}:")
        for g in picked:
            h = g["game"].headers
            print(f"    #{g['idx']:<3} {h.get('White','?')[:16]:<17} vs "
                  f"{h.get('Black','?')[:16]:<17} {g['result']:<8} "
                  f"acpl {g['acpl']:<4} score {g['score']:.1f}")

        limit = chess.engine.Limit(time=args.time)
        for g in picked:
            h = g["game"].headers
            print(f"\nDeep pass: {h.get('White','?')} vs {h.get('Black','?')}", file=sys.stderr)
            rep = A.analyse_game(g["game"], sf, limit, multipv=2)
            g["rep"] = rep
            g["flagged"] = annotate(rep, g["side"], maia_engine)
            g["drift"] = detect_drift(rep, g["side"])
    finally:
        sf.quit()
        if maia_engine:
            maia_engine.close()

    games = [g for g in picked if g.get("rep")]
    agg = aggregate(games, args.user)
    print_diagnosis(agg, args.user)

    os.makedirs(P("reports"), exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    out = args.out or P(f"reports/{args.user}-coach-{stamp}.html")
    meta = (f"{len(games)} games analyzed of {len(screened)} screened · "
            f"{args.time_class or 'all time controls'} · "
            f"Stockfish {args.time}s/move"
            + (f" · Maia {maia_engine.band}" if maia_engine else ""))
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(render_report(agg, games, args.user, meta))
    print(f"\nReport: {out}")

    drill = [{
        "fen": m.fen_before, "played": m.san, "best": m.best_san,
        "best_line": m.best_pv, "refutation": m.refutation,
        "tag": m.tag, "note": m.tag_note, "drop": round(m.drop, 1),
        "cp_before": m.cp_before, "cp_after": m.cp_after,
        "move_no": m.move_no, "side": "white" if m.color == chess.WHITE else "black",
        "maia": {k: v for k, v in (m.maia or {}).items() if k != "top"},
        "game": f"{g['rep'].headers.get('White','?')} vs {g['rep'].headers.get('Black','?')}",
        "url": g["rep"].headers.get("Link", ""),
    } for g, m in agg["critical"][:20]]
    dpath = P(f"reports/{args.user}-drills-{stamp}.json")
    with open(dpath, "w") as fh:
        json.dump(drill, fh, indent=2)
    print(f"Drills: {dpath}")


if __name__ == "__main__":
    main()
