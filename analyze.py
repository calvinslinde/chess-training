#!/usr/bin/env python3
"""
Chess game analyzer.

Stockfish supplies tactical ground truth (centipawn loss, blunder detection,
accuracy). Leela Chess Zero -- the open-source AlphaZero -- optionally supplies
a second, positional opinion on the moments that mattered.

Outputs a terminal summary and a self-contained HTML report.
"""

import argparse, html, math, os, re, shutil, sys, datetime
from paths import P  # data folder; see paths.py
from dataclasses import dataclass, field

import chess, chess.engine, chess.pgn, chess.svg

# ---------------------------------------------------------------- engine paths

def find_engine(name, override=None):
    if override:
        return override if os.path.exists(override) else None
    return shutil.which(name)


CLK_RE = re.compile(r"\[%clk\s+(\d+):(\d+):([\d.]+)\]")

def parse_clock(comment):
    """Seconds remaining, from a PGN [%clk h:mm:ss] tag."""
    if not comment:
        return None
    m = CLK_RE.search(comment)
    if not m:
        return None
    h, mi, s = m.groups()
    return int(h) * 3600 + int(mi) * 60 + float(s)


def parse_increment(tc):
    """Increment in seconds from a TimeControl header like '180+2'."""
    if not tc or "+" not in tc:
        return 0.0
    try:
        return float(tc.split("+")[1])
    except ValueError:
        return 0.0


# ------------------------------------------------------------ score conversion

MATE_CP = 10000

def score_cp(pov_score):
    """Centipawns from the point of view the PovScore was taken in."""
    s = pov_score
    if s.is_mate():
        m = s.mate()
        return MATE_CP if m > 0 else -MATE_CP
    return max(-MATE_CP, min(MATE_CP, s.score()))


def win_prob(cp):
    """Lichess's centipawn -> win percentage curve. Returns 0..100."""
    if cp >= MATE_CP:
        return 100.0
    if cp <= -MATE_CP:
        return 0.0
    return 50 + 50 * (2 / (1 + math.exp(-0.00368208 * cp)) - 1)


def accuracy_from_drop(drop):
    """Lichess's accuracy curve: win-percent drop -> accuracy 0..100."""
    a = 103.1668 * math.exp(-0.04354 * drop) - 3.1669
    return max(0.0, min(100.0, a))


# Classification thresholds, in win-percent lost by the move.
def classify(drop, was_best):
    if was_best:
        return "best"
    if drop >= 20:
        return "blunder"
    if drop >= 10:
        return "mistake"
    if drop >= 5:
        return "inaccuracy"
    return "ok"


CLASS_ORDER = ["blunder", "mistake", "inaccuracy", "ok", "best"]
CLASS_SYMBOL = {"blunder": "??", "mistake": "?", "inaccuracy": "?!", "ok": "", "best": "!"}


# ------------------------------------------------------------------ data model

@dataclass
class MoveReport:
    ply: int
    move_no: int
    color: bool               # chess.WHITE / chess.BLACK
    san: str
    fen_before: str
    uci: str
    cp_before: int            # mover's perspective, before the move
    cp_after: int             # mover's perspective, after the move
    wp_before: float
    wp_after: float
    drop: float               # win-percent lost
    cpl: int                  # centipawn loss, capped
    label: str
    best_san: str = ""
    best_pv: str = ""
    leela: str = ""           # optional Leela second opinion
    clock: float = None       # seconds left on the mover's clock
    time_spent: float = None  # seconds spent on this move
    tag: str = ""             # error taxonomy tag
    tag_note: str = ""        # plain-language diagnosis
    maia: dict = field(default_factory=dict)
    refutation: str = ""      # opponent's punishing line


@dataclass
class GameReport:
    headers: dict
    moves: list = field(default_factory=list)
    result: str = "*"

    def side_moves(self, color):
        return [m for m in self.moves if m.color == color]

    def acpl(self, color):
        ms = self.side_moves(color)
        return round(sum(m.cpl for m in ms) / len(ms)) if ms else 0

    def accuracy(self, color):
        ms = self.side_moves(color)
        if not ms:
            return 0.0
        return sum(accuracy_from_drop(m.drop) for m in ms) / len(ms)

    def counts(self, color):
        c = {k: 0 for k in CLASS_ORDER}
        for m in self.side_moves(color):
            c[m.label] += 1
        return c

    def phase_acpl(self, color):
        """opening (ply<=20) / middlegame / endgame (<=12 pieces on board)."""
        out = {}
        for phase in ("opening", "middlegame", "endgame"):
            ms = [m for m in self.side_moves(color) if move_phase(m) == phase]
            out[phase] = (round(sum(x.cpl for x in ms) / len(ms)), len(ms)) if ms else (None, 0)
        return out


def move_phase(m):
    board = chess.Board(m.fen_before)
    pieces = chess.popcount(board.occupied)
    if pieces <= 12:
        return "endgame"
    if m.ply <= 20:
        return "opening"
    return "middlegame"


# -------------------------------------------------------------------- analysis

def analyse_game(game, sf, limit, multipv=2, progress=True):
    progress = progress and sys.stderr.isatty()
    board = game.board()
    rep = GameReport(headers=dict(game.headers), result=game.headers.get("Result", "*"))
    nodes = list(game.mainline())
    node_moves = [n.move for n in nodes]
    total = len(node_moves)
    inc = parse_increment(game.headers.get('TimeControl', ''))
    if total == 0:
        return rep

    # One engine call per position: eval(P) is from side-to-move's view, so the
    # mover's eval after their move is -eval(P'). N+1 calls instead of 2N.
    def evaluate(b):
        info = sf.analyse(b, limit, multipv=multipv)
        if isinstance(info, dict):
            info = [info]
        return info

    prev_info = evaluate(board)
    for i, move in enumerate(node_moves):
        cp_before = score_cp(prev_info[0]["score"].pov(board.turn))
        pv = prev_info[0].get("pv", [])
        best_move = pv[0] if pv else None
        best_san = board.san(best_move) if best_move else ""
        best_pv = board.variation_san(pv[:6]) if pv else ""

        san = board.san(move)
        fen_before = board.fen()
        mover = board.turn
        board.push(move)

        if board.is_game_over():
            info_after = None
            outcome = board.outcome()
            if outcome and outcome.winner is not None:
                cp_after = MATE_CP if outcome.winner == mover else -MATE_CP
            else:
                cp_after = 0
        else:
            info_after = evaluate(board)
            # eval after is from the opponent's view; negate for the mover.
            cp_after = -score_cp(info_after[0]["score"].pov(board.turn))

        wpb, wpa = win_prob(cp_before), win_prob(cp_after)
        drop = max(0.0, wpb - wpa)
        was_best = (best_move is not None and move == best_move)
        if was_best:
            drop = min(drop, 0.0)

        clk = parse_clock(nodes[i].comment)
        prev_same_side = None
        for pm in reversed(rep.moves):
            if pm.color == mover:
                prev_same_side = pm
                break
        spent = None
        if clk is not None and prev_same_side and prev_same_side.clock is not None:
            spent = max(0.0, prev_same_side.clock - clk + inc)

        refut = ""
        if info_after and info_after[0].get("pv"):
            refut = board.variation_san(info_after[0]["pv"][:5])

        rep.moves.append(MoveReport(
            clock=clk, time_spent=spent, refutation=refut,
            ply=i + 1,
            move_no=(i // 2) + 1,
            color=mover,
            san=san,
            fen_before=fen_before,
            uci=move.uci(),
            cp_before=cp_before,
            cp_after=cp_after,
            wp_before=wpb,
            wp_after=wpa,
            drop=drop,
            cpl=max(0, min(1000, cp_before - cp_after)),
            label=classify(drop, was_best),
            best_san=best_san,
            best_pv=best_pv,
        ))

        prev_info = info_after
        if progress:
            done = i + 1
            bar = "#" * int(28 * done / total)
            sys.stderr.write(f"\r  [{bar:<28}] {done}/{total} plies")
            sys.stderr.flush()
        if info_after is None:
            break

    if progress:
        sys.stderr.write("\r" + " " * 52 + "\r")
    return rep


def leela_second_opinion(lc0, moves, limit, top=3):
    """Ask Leela what it thinks about the positions that mattered."""
    for m in moves:
        board = chess.Board(m.fen_before)
        try:
            info = lc0.analyse(board, limit, multipv=top)
            if isinstance(info, dict):
                info = [info]
            parts = []
            for line in info[:top]:
                pv = line.get("pv", [])
                if not pv:
                    continue
                cp = score_cp(line["score"].pov(board.turn))
                parts.append(f"{board.san(pv[0])} ({cp/100:+.2f})")
            m.leela = ", ".join(parts)
        except Exception as e:
            m.leela = f"[lc0 error: {e}]"


# ------------------------------------------------------------------- terminal

C = {"blunder": "\033[91m", "mistake": "\033[93m", "inaccuracy": "\033[96m",
     "best": "\033[92m", "ok": "", "dim": "\033[2m", "b": "\033[1m", "r": "\033[0m"}

def print_summary(rep, critical, use_color=True):
    def c(k):
        return C[k] if use_color else ""
    h = rep.headers
    w = h.get("White", "White"); b = h.get("Black", "Black")
    print(f"\n{c('b')}{w} vs {b}{c('r')}   {h.get('Result','*')}"
          f"   {c('dim')}{h.get('Event','')} {h.get('Date','')}{c('r')}")
    eco = " ".join(x for x in (h.get("ECO",""), h.get("Opening","")) if x)
    if eco:
        print(f"  {c('dim')}{eco}{c('r')}")

    print(f"\n  {'':<10}{'Acc':>7}{'ACPL':>7}{'??':>5}{'?':>5}{'?!':>5}{'best':>6}")
    for color, name in ((chess.WHITE, w), (chess.BLACK, b)):
        cnt = rep.counts(color)
        print(f"  {name[:10]:<10}{rep.accuracy(color):>6.1f}%{rep.acpl(color):>7}"
              f"{c('blunder')}{cnt['blunder']:>5}{c('r')}"
              f"{c('mistake')}{cnt['mistake']:>5}{c('r')}"
              f"{c('inaccuracy')}{cnt['inaccuracy']:>5}{c('r')}"
              f"{c('best')}{cnt['best']:>6}{c('r')}")

    print(f"\n  {c('dim')}ACPL by phase (opening / middlegame / endgame){c('r')}")
    for color, name in ((chess.WHITE, w), (chess.BLACK, b)):
        ph = rep.phase_acpl(color)
        cells = " ".join(
            f"{p[:3]} {v[0] if v[0] is not None else '--':>4}{c('dim')}({v[1]}){c('r')}"
            for p, v in ph.items())
        print(f"  {name[:10]:<10}{cells}")

    if critical:
        print(f"\n  {c('b')}Critical moments{c('r')}")
        for m in critical:
            side = "W" if m.color == chess.WHITE else "B"
            num = f"{m.move_no}.{'' if m.color == chess.WHITE else '..'}"
            print(f"    {c(m.label)}{num:>7} {m.san:<8}{CLASS_SYMBOL[m.label]:<3}{c('r')}"
                  f"{c('dim')}({side}){c('r')}  -{m.drop:.0f}% win  "
                  f"{c('dim')}eval {m.cp_before/100:+.2f} -> {m.cp_after/100:+.2f}{c('r')}")
            if m.best_san:
                print(f"             {c('best')}best: {m.best_pv}{c('r')}")
            if m.leela:
                print(f"             {c('dim')}leela: {m.leela}{c('r')}")


# ---------------------------------------------------------------- HTML report

def eval_graph_svg(rep, w=880, h=180):
    """Area chart of White's win probability across the game."""
    pts = [(0, 50.0)]
    for m in rep.moves:
        wp = m.wp_after if m.color == chess.WHITE else 100 - m.wp_after
        pts.append((m.ply, wp))
    if len(pts) < 2:
        return ""
    maxply = pts[-1][0]
    X = lambda p: p / maxply * w
    Y = lambda v: h - (v / 100 * h)
    line = " ".join(f"{X(p):.1f},{Y(v):.1f}" for p, v in pts)
    marks = []
    for m in rep.moves:
        if m.label in ("blunder", "mistake"):
            wp = m.wp_after if m.color == chess.WHITE else 100 - m.wp_after
            col = "var(--blunder)" if m.label == "blunder" else "var(--mistake)"
            marks.append(f'<circle cx="{X(m.ply):.1f}" cy="{Y(wp):.1f}" r="4" fill="{col}"/>')
    return f'''<svg viewBox="0 0 {w} {h}" class="evalgraph" preserveAspectRatio="none"
      role="img" aria-label="White win probability across the game">
  <rect width="{w}" height="{h}" fill="var(--evalblack)"/>
  <polygon points="0,{h} {line} {w},{h}" fill="var(--evalwhite)"/>
  <line x1="0" y1="{h/2}" x2="{w}" y2="{h/2}" stroke="var(--grid)" stroke-dasharray="4 4"/>
  <polyline points="{line}" fill="none" stroke="var(--evalline)" stroke-width="1.5"/>
  {''.join(marks)}
</svg>'''


CSS = """
:root{--bg:#fbfaf8;--fg:#1c1b19;--muted:#6b6862;--card:#fff;--border:#e3e0da;
--blunder:#c33c3c;--mistake:#d98324;--inaccuracy:#3f8fbf;--best:#3f8f5a;
--evalwhite:#e8e5df;--evalblack:#33312e;--evalline:#8a867e;--grid:#9a968e;--accent:#7a4f2a}
:root:not([data-theme=light]){@media (prefers-color-scheme:dark){
:root{--bg:#161513;--fg:#eceae5;--muted:#98948c;--card:#1e1d1a;--border:#33312d;
--evalwhite:#d8d4cc;--evalblack:#2a2825;--evalline:#7d7970;--grid:#5a564f;--accent:#c99a6a}}}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){
--bg:#161513;--fg:#eceae5;--muted:#98948c;--card:#1e1d1a;--border:#33312d;
--evalwhite:#d8d4cc;--evalblack:#2a2825;--evalline:#7d7970;--grid:#5a564f;--accent:#c99a6a}}
:root[data-theme=dark]{--bg:#161513;--fg:#eceae5;--muted:#98948c;--card:#1e1d1a;--border:#33312d;
--evalwhite:#d8d4cc;--evalblack:#2a2825;--evalline:#7d7970;--grid:#5a564f;--accent:#c99a6a}
*{box-sizing:border-box}
body{background:var(--bg);color:var(--fg);margin:0;padding:2.5rem 1.5rem 5rem;
font:16px/1.6 ui-sans-serif,-apple-system,"Segoe UI",system-ui,sans-serif}
.wrap{max-width:960px;margin:0 auto}
h1{font-size:1.5rem;margin:0 0 .2rem;letter-spacing:-.01em}
h2{font-size:1rem;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);
margin:3rem 0 1rem;font-weight:600}
.sub{color:var(--muted);font-size:.9rem;margin-bottom:2rem}
.card{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:1.25rem}
table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}
th{text-align:right;font-size:.75rem;text-transform:uppercase;letter-spacing:.05em;
color:var(--muted);font-weight:600;padding:.4rem .6rem;border-bottom:1px solid var(--border)}
th:first-child,td:first-child{text-align:left}
td{text-align:right;padding:.5rem .6rem;border-bottom:1px solid var(--border)}
tbody tr:last-child td{border-bottom:none}
.scroll{overflow-x:auto;-webkit-overflow-scrolling:touch}
.evalgraph{width:100%;height:180px;display:block;border-radius:8px;border:1px solid var(--border)}
.blunder{color:var(--blunder)}.mistake{color:var(--mistake)}
.inaccuracy{color:var(--inaccuracy)}.best{color:var(--best)}
.mom{display:grid;grid-template-columns:300px 1fr;gap:1.5rem;align-items:start;
background:var(--card);border:1px solid var(--border);border-radius:10px;
padding:1.25rem;margin-bottom:1.25rem}
@media(max-width:680px){.mom{grid-template-columns:1fr}}
.mom svg{width:100%;height:auto;border-radius:6px}
.mom h3{margin:0 0 .5rem;font-size:1.05rem}
.kv{margin:.35rem 0;font-size:.9rem}
.kv b{font-weight:600;color:var(--muted);font-weight:500}
.pv{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.85rem}
.tag{display:inline-block;font-size:.7rem;text-transform:uppercase;letter-spacing:.06em;
padding:.15rem .5rem;border-radius:999px;border:1px solid currentColor;margin-left:.4rem}
.movelist{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.85rem;
line-height:2;word-spacing:.1em}
.movelist span{padding:.1rem .25rem;border-radius:3px}
footer{color:var(--muted);font-size:.8rem;margin-top:4rem;border-top:1px solid var(--border);
padding-top:1rem}
"""


def html_report(reports, meta):
    out = [f"<title>Chess analysis</title><style>{CSS}</style><div class='wrap'>"]
    out.append(f"<h1>Game analysis</h1><div class='sub'>{html.escape(meta)}</div>")

    for gi, rep in enumerate(reports):
        h = rep.headers
        w = html.escape(h.get("White", "White")); b = html.escape(h.get("Black", "Black"))
        eco = html.escape(" ".join(x for x in (h.get("ECO",""), h.get("Opening","")) if x))
        out.append(f"<h2>Game {gi+1} — {w} vs {b} · {html.escape(h.get('Result','*'))}</h2>")
        if eco:
            out.append(f"<div class='sub'>{eco}</div>")
        out.append(eval_graph_svg(rep))

        out.append("<div class='scroll'><table><thead><tr><th>Player</th><th>Accuracy</th>"
                   "<th>ACPL</th><th>Blunders</th><th>Mistakes</th><th>Inaccuracies</th>"
                   "<th>Best moves</th><th>Opening</th><th>Middle</th><th>End</th>"
                   "</tr></thead><tbody>")
        for color, name in ((chess.WHITE, w), (chess.BLACK, b)):
            c = rep.counts(color); ph = rep.phase_acpl(color)
            cells = "".join(f"<td>{ph[p][0] if ph[p][0] is not None else '—'}</td>"
                            for p in ("opening", "middlegame", "endgame"))
            out.append(f"<tr><td>{name}</td><td>{rep.accuracy(color):.1f}%</td>"
                       f"<td>{rep.acpl(color)}</td>"
                       f"<td class='blunder'>{c['blunder']}</td>"
                       f"<td class='mistake'>{c['mistake']}</td>"
                       f"<td class='inaccuracy'>{c['inaccuracy']}</td>"
                       f"<td class='best'>{c['best']}</td>{cells}</tr>")
        out.append("</tbody></table></div>")

        crit = getattr(rep, "_critical", [])
        if crit:
            out.append("<h2>Critical moments</h2>")
            for m in crit:
                board = chess.Board(m.fen_before)
                mv = chess.Move.from_uci(m.uci)
                arrows = [chess.svg.Arrow(mv.from_square, mv.to_square, color="#c33c3c")]
                if m.best_san:
                    try:
                        bm = board.parse_san(m.best_san)
                        arrows.append(chess.svg.Arrow(bm.from_square, bm.to_square, color="#3f8f5a"))
                    except Exception:
                        pass
                svg = chess.svg.board(board, arrows=arrows, size=300,
                                      orientation=m.color, coordinates=False)
                num = f"{m.move_no}.{'' if m.color == chess.WHITE else '..'}"
                who = w if m.color == chess.WHITE else b
                out.append(
                    f"<div class='mom'><div>{svg}</div><div>"
                    f"<h3>{num} {html.escape(m.san)} {CLASS_SYMBOL[m.label]}"
                    f"<span class='tag {m.label}'>{m.label}</span></h3>"
                    f"<div class='kv'><b>Player:</b> {who}</div>"
                    f"<div class='kv'><b>Eval:</b> {m.cp_before/100:+.2f} → {m.cp_after/100:+.2f} "
                    f"({m.drop:.0f}% win probability lost)</div>"
                    f"<div class='kv'><b>Stockfish:</b> <span class='pv'>{html.escape(m.best_pv)}</span></div>"
                    + (f"<div class='kv'><b>Leela:</b> <span class='pv'>{html.escape(m.leela)}</span></div>"
                       if m.leela else "")
                    + "<div class='kv' style='color:var(--muted);font-size:.8rem'>"
                      "Red = played. Green = engine's choice.</div></div></div>")

        out.append("<h2>Move list</h2><div class='card movelist'>")
        for m in rep.moves:
            if m.color == chess.WHITE:
                out.append(f"{m.move_no}.")
            cls = m.label if m.label in ("blunder", "mistake", "inaccuracy") else ""
            out.append(f"<span class='{cls}'>{html.escape(m.san)}{CLASS_SYMBOL[m.label]}</span> ")
        out.append("</div>")

    out.append("<footer>Stockfish for tactical evaluation; Leela Chess Zero for the "
               "positional second opinion. Accuracy and win probability use Lichess's "
               "centipawn curves; accuracy is a plain mean over moves.</footer></div>")
    return "".join(out)


# ------------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser(description="Analyze chess games with Stockfish and Leela.")
    ap.add_argument("pgn", nargs="?", help="PGN file (defaults to newest in $CHESS_HOME/games)")
    ap.add_argument("--game", type=int, help="analyze only game N (1-indexed)")
    ap.add_argument("--max-games", type=int, default=3, help="max games to analyze (default 3)")
    ap.add_argument("--time", type=float, default=0.4, help="seconds per position (default 0.4)")
    ap.add_argument("--depth", type=int, help="fixed depth instead of time")
    ap.add_argument("--leela", action="store_true", help="add Leela's opinion on critical moments")
    ap.add_argument("--leela-time", type=float, default=3.0, help="seconds per Leela position")
    ap.add_argument("--critical", type=int, default=5, help="how many critical moments (default 5)")
    ap.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--hash", type=int, default=512, help="Stockfish hash in MB")
    ap.add_argument("--out", help="HTML report path")
    ap.add_argument("--no-html", action="store_true")
    ap.add_argument("--stockfish", help="path to stockfish binary")
    ap.add_argument("--lc0", help="path to lc0 binary")
    args = ap.parse_args()

    pgn_path = args.pgn
    if not pgn_path:
        games_dir = P("games")
        cands = [os.path.join(games_dir, f) for f in os.listdir(games_dir)
                 if f.endswith(".pgn")] if os.path.isdir(games_dir) else []
        if not cands:
            sys.exit("No PGN given and none found in $CHESS_HOME/games. "
                     "Try: chess-fetch <lichess-username>")
        pgn_path = max(cands, key=os.path.getmtime)
        print(f"Using {pgn_path}")

    sf_path = find_engine("stockfish", args.stockfish)
    if not sf_path:
        sys.exit("Stockfish not found. Install it with: brew install stockfish")

    limit = (chess.engine.Limit(depth=args.depth) if args.depth
             else chess.engine.Limit(time=args.time))

    sf = chess.engine.SimpleEngine.popen_uci(sf_path)
    sf.configure({"Threads": args.threads, "Hash": args.hash})

    lc0 = None
    if args.leela:
        lc0_path = find_engine("lc0", args.lc0)
        if not lc0_path:
            print("lc0 not found; skipping Leela. Install with: brew install lc0", file=sys.stderr)
        else:
            try:
                lc0 = chess.engine.SimpleEngine.popen_uci(lc0_path)
            except Exception as e:
                print(f"Could not start lc0 ({e}); skipping Leela.", file=sys.stderr)

    reports = []
    try:
        with open(pgn_path, encoding="utf-8", errors="replace") as fh:
            idx = 0
            while len(reports) < args.max_games:
                game = chess.pgn.read_game(fh)
                if game is None:
                    break
                idx += 1
                if args.game and idx != args.game:
                    continue
                hdr = game.headers
                print(f"\nAnalyzing game {idx}: {hdr.get('White','?')} vs {hdr.get('Black','?')}"
                      f" ({hdr.get('Result','*')})", file=sys.stderr)
                rep = analyse_game(game, sf, limit)
                if not rep.moves:
                    print("  (no moves)", file=sys.stderr)
                    continue
                crit = sorted(rep.moves, key=lambda m: -m.drop)[:args.critical]
                crit = [m for m in crit if m.drop >= 3]
                crit.sort(key=lambda m: m.ply)
                if lc0:
                    print("  Asking Leela about the critical moments...", file=sys.stderr)
                    leela_second_opinion(lc0, crit, chess.engine.Limit(time=args.leela_time))
                rep._critical = crit
                reports.append(rep)
                print_summary(rep, crit, use_color=sys.stdout.isatty())
                if args.game:
                    break
    finally:
        sf.quit()
        if lc0:
            lc0.quit()

    if not reports:
        sys.exit("\nNo games analyzed.")

    if not args.no_html:
        out = args.out
        if not out:
            os.makedirs(P("reports"), exist_ok=True)
            stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            base = os.path.splitext(os.path.basename(pgn_path))[0]
            out = P(f"reports/{base}-{stamp}.html")
        meta = (f"{os.path.basename(pgn_path)} · {len(reports)} game(s) · "
                f"Stockfish {'depth ' + str(args.depth) if args.depth else str(args.time) + 's/move'}"
                + (" · Leela second opinion" if lc0 else ""))
        with open(out, "w", encoding="utf-8") as fh:
            fh.write(html_report(reports, meta))
        print(f"\nReport: {out}")


if __name__ == "__main__":
    main()
