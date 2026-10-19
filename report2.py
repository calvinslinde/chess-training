#!/usr/bin/env python3
"""HTML report for the v2 review."""
import json, html, os, sys, collections, statistics
from paths import P  # data folder; see paths.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import chess
import analyze as A, classify as CL, tactics as T
from board_svg import board_svg

CSS = """
.hl{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:.9rem;margin:1.4rem 0}
.hl div{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:.9rem}
.hl .n{font-size:1.6rem;font-weight:600;letter-spacing:-.02em;line-height:1.1}
.hl .l{font-size:.7rem;text-transform:uppercase;letter-spacing:.06em;color:var(--muted);margin-top:.3rem}
.lead{font-size:1.02rem;line-height:1.7;background:var(--card);border-left:3px solid var(--accent);
border-radius:0 8px 8px 0;padding:1.05rem 1.3rem;margin:1.4rem 0}
.chips{display:flex;flex-wrap:wrap;gap:.5rem;margin:1rem 0}
.chip{border:1px solid var(--border);border-radius:999px;padding:.3rem .8rem;font-size:.82rem;
background:var(--card)}
.chip b{font-weight:600}
.game{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:1.2rem;margin:1.1rem 0}
.pos{display:grid;grid-template-columns:280px 1fr;gap:1.3rem;align-items:start;margin-top:1rem;
padding-top:1rem;border-top:1px solid var(--border)}
@media(max-width:680px){.pos{grid-template-columns:1fr}}
.pos svg{width:100%;height:auto}
.kv{margin:.3rem 0;font-size:.9rem}.kv b{color:var(--muted);font-weight:500}
.pv{font-family:ui-monospace,Menlo,monospace;font-size:.84rem}
.lost{color:var(--blunder)}.gain{color:var(--best)}
ul.tight{margin:.4rem 0 .2rem;padding-left:1.1rem}ul.tight li{margin:.15rem 0;font-size:.9rem}
"""
COL = {"brilliant":"var(--best)","great":"var(--best)","best":"var(--best)",
       "excellent":"var(--muted)","good":"var(--muted)","book":"var(--muted)",
       "inaccuracy":"var(--inaccuracy)","mistake":"var(--mistake)",
       "miss":"var(--mistake)","blunder":"var(--blunder)"}


def build(games, me, out):
    o=[f"<title>{html.escape(me)} — game review</title><style>{A.CSS}{CSS}</style><div class='wrap'>"]
    agg=collections.Counter()
    for g in games:
        for k,v in g["counts"].items(): agg[k]+=v
    acc=statistics.mean([g["accuracy"] for g in games])
    acpl=statistics.mean([g["acpl"] for g in games])
    n=sum(agg.values())
    o.append(f"<h1>{html.escape(me)} — game review</h1><div class='sub'>{len(games)} games · "
             f"Stockfish 18 two-pass · Maia policy · Lichess accuracy method · tablebase endgames</div>")
    o.append("<div class='hl'>"
      f"<div><div class='n'>{acc:.1f}%</div><div class='l'>Accuracy</div></div>"
      f"<div><div class='n'>{acpl:.0f}</div><div class='l'>Avg centipawn loss</div></div>"
      f"<div><div class='n'>{agg['great']+agg['brilliant']+agg['best']}</div><div class='l'>Best or better</div></div>"
      f"<div><div class='n'>{agg['blunder']+agg['miss']}</div><div class='l'>Blunders + misses</div></div>"
      f"<div><div class='n'>{n}</div><div class='l'>Your moves</div></div></div>")

    o.append("<h2>Every move, classified</h2><div class='chips'>")
    for k in CL.ORDER:
        if agg[k]:
            o.append(f"<span class='chip' style='color:{COL[k]}'><b>{agg[k]}</b> {k}</span>")
    o.append("</div>")

    pos=[(g,mv) for g in games for mv in g["moves"] if mv.get("label") in ("brilliant","great")]
    if pos:
        o.append("<h2>What you did well</h2><div class='card'><ul class='tight'>")
        for g,mv in pos:
            c=mv.get("complexity") or {}
            o.append(f"<li><b>{mv['move_no']}{'.' if mv['white'] else '...'}{html.escape(mv['san'])}</b> "
                     f"({mv['label']}) — the only move that held, and every alternative was "
                     f"{c.get('spread',0)/100:.1f} pawns worse. "
                     f"{(c.get('p_best') or 0)*100:.0f}% of players at your level find it.</li>")
        o.append("</ul></div>")

    cost=[(g,mv) for g in games for mv in g["moves"] if mv.get("playout",{}).get("cost") is not None]
    if cost:
        o.append("<h2>What each error actually cost</h2>"
                 "<p style='color:var(--muted)'>Win-probability loss is the engine's currency. This is yours: "
                 "the position simulated forward 150 times with Maia at your rating against Maia at your "
                 "opponents', so the cost is in expected points against the players you actually face.</p>"
                 "<div class='scroll'><table><thead><tr><th>Move</th><th>Win% lost</th>"
                 "<th>Expected points lost</th><th>95% CI</th><th>Real?</th></tr></thead><tbody>")
        for g,mv in sorted(cost,key=lambda x:-x[1]["playout"]["cost"]):
            p=mv["playout"]
            o.append(f"<tr><td>{mv['move_no']}{'.' if mv['white'] else '...'}{html.escape(mv['san'])}</td>"
                     f"<td>{mv['drop']:.1f}</td><td>{p['cost']:+.3f}</td>"
                     f"<td>±{p['cost_ci95']:.3f}</td>"
                     f"<td>{'yes' if p.get('significant') else '<span style=color:var(--muted)>no</span>'}</td></tr>")
        o.append("</tbody></table></div>")

    o.append("<h2>Game by game</h2>")
    for g in games:
        h=g["headers"]; opp=h["Black"] if g["side"]=="white" else h["White"]
        # order matters: "won" is True for a draw as Black under the naive
        # equality test, so the draw case has to be checked first
        if g["result"] == "1/2-1/2":
            res = "DRAW"
        else:
            res = "WIN" if (g["result"] == "1-0") == (g["side"] == "white") else "LOSS"
        o.append(f"<div class='game'><h3>as {g['side']} vs {html.escape(opp)} — "
                 f"<span style='color:{'var(--best)' if res=='WIN' else 'var(--blunder)'}'>{res}</span></h3>"
                 f"<div class='kv'><b>{html.escape(g['opening']['name'] or '')}</b> "
                 f"({g['opening']['eco']}) · in book through move {(g['opening']['book_plies']+1)//2} · "
                 f"accuracy {g['accuracy']}% · ACPL {g['acpl']}</div>")
        errs=[m for m in g["moves"] if m.get("mine") and m.get("drop",0)>=10 and m.get("best")]
        for mv in sorted(errs,key=lambda m:-m["drop"])[:2]:
            b=chess.Board(mv["fen"]); m0=chess.Move.from_uci(mv["uci"])
            svg=board_svg(mv["fen"], flip=(g["side"]=="black"), size=250, x0=12, y0=42,
                          highlight={m0.from_square,m0.to_square})
            svg=svg.replace('viewBox="0 0 680','viewBox="0 0 275').replace('x="340" y="30"','x="137" y="30"')
            po=mv.get("playout") or {}
            comp=mv.get("complexity") or {}
            posx=mv.get("positional") or {}
            o.append(f"<div class='pos'><div>{svg}</div><div>"
              f"<div class='kv'><b>Played</b> <span class='lost'>{mv['move_no']}"
              f"{'.' if mv['white'] else '...'}{html.escape(mv['san'])}</span> "
              f"({mv.get('label','')}) — {mv['drop']:.0f}% win probability lost, "
              f"{mv['cp_before']/100:+.2f} → {mv['cp_after']/100:+.2f}</div>"
              f"<div class='kv'><b>Engine</b> <span class='pv'>{html.escape(mv.get('best_pv',''))}</span></div>"
              + (f"<div class='kv'><b>Punished by</b> <span class='pv'>{html.escape(mv['refutation'])}</span></div>"
                 if mv.get("refutation") else ""))
            if posx.get("lost"):
                o.append("<div class='kv'><b>What it cost, specifically</b><ul class='tight'>"
                         + "".join(f"<li class='lost'>your move {html.escape(x['text'])} "
                                   f"(~{x['weighted']:.0f}cp)</li>" for x in posx["lost"]) + "</ul></div>")
            if posx.get("gained"):
                o.append("<div class='kv'><ul class='tight'>"
                         + "".join(f"<li class='gain'>your move did {html.escape(x['text'])}</li>"
                                   for x in posx["gained"]) + "</ul></div>")
            ds=CL.difficulty_sentence(comp)
            if ds: o.append(f"<div class='kv' style='color:var(--muted)'>{html.escape(ds)}</div>")
            if mv.get("maia",{}).get("note"):
                o.append(f"<div class='kv' style='color:var(--muted)'>{html.escape(mv['maia']['note'])}</div>")
            if po.get("cost") is not None:
                sig = "" if po.get("significant") else " — not distinguishable from zero"
                o.append(f"<div class='kv'><b>Real cost</b> {po['cost']:+.3f} ± {po['cost_ci95']:.3f} "
                         f"expected points{sig}</div>")
            if mv.get("clock") is not None:
                o.append(f"<div class='kv'><b>Clock</b> {mv['clock']/60:.1f} min left</div>")
            o.append("</div></div>")
        o.append("</div>")

    o.append("<h2>Method, and what it cannot do</h2><div class='card'><ul class='tight'>"
      "<li><b>Accuracy</b> uses Lichess's published volatility-weighted plus harmonic method. It will "
      "not match chess.com, whose CAPS2 formula is private; the two disagree by roughly 10 points on "
      "these games, in both directions.</li>"
      "<li><b>ACPL</b> comes from one consistent eval pass and reproduces to about ±3; the residual is "
      "Stockfish's threaded nondeterminism.</li>"
      "<li><b>Expected-points cost</b> is a Monte Carlo estimate over 150 paired simulations. A single "
      "playout has a standard deviation near 0.4, so anything under about ±0.08 is noise. Rows marked "
      "not significant should not be read as findings.</li>"
      "<li><b>Maia bands</b> are Lichess ratings by construction and top out at 1900.</li>"
      "<li><b>Positional decomposition</b> compares the quiescent position at the end of each line, so "
      "exchanges resolve before features are counted.</li>"
      "<li><b>Endgames</b> with seven pieces or fewer use Syzygy tablebase ground truth, not engine eval.</li>"
      "</ul></div>")
    o.append("</div>")
    open(out,"w").write("".join(o))
    return out


if __name__ == "__main__":
    games=json.load(open(sys.argv[1]))
    print(build(games, sys.argv[2] if len(sys.argv)>2 else "player",
                sys.argv[3] if len(sys.argv)>3 else P("reports/review.html")))
