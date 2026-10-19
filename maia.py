"""
Maia: neural networks trained to predict what a *human* at a given rating plays,
rather than what is objectively best.

The gap between "best move" and "move a player at your level would find" is the
whole diagnostic. Four cases, and they call for different advice:

  best move likely at your level, you missed it -> below your own standard here
  best move unlikely at your level, you missed it -> a new pattern to learn
  your move is what your whole band plays, and it's bad -> systemic, high transfer
  your move is one almost nobody plays, and it's bad -> a one-off; check the clock
"""

import os, re, subprocess
from paths import P  # data folder; see paths.py

import chess

WEIGHT_DIR = os.environ.get("MAIA_WEIGHTS") or P("engines/maia")  # folder holding maia-1100.pb.gz ... maia-1900.pb.gz
BANDS = [1100, 1300, 1500, 1700, 1900]
_P_RE = re.compile(r"^info string (\S+)\s+\(\s*\d+\s*\)\s+N:.*?\(P:\s*([\d.]+)%\)")


def nearest_band(rating):
    return min(BANDS, key=lambda b: abs(b - rating))


class MaiaPolicy:
    """A persistent lc0 process serving one Maia network's policy head."""

    def __init__(self, rating=1500, backend="eigen"):
        self.band = nearest_band(rating)
        self.weights = os.path.join(WEIGHT_DIR, f"maia-{self.band}.pb.gz")
        if not os.path.exists(self.weights):
            raise FileNotFoundError(f"Missing Maia weights: {self.weights}")
        self.proc = subprocess.Popen(
            ["lc0", f"--weights={self.weights}", f"--backend={backend}",
             "--verbose-move-stats"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, bufsize=1)
        self._send("uci")
        self._drain_until("uciok")

    def _send(self, s):
        self.proc.stdin.write(s + "\n")
        self.proc.stdin.flush()

    def _drain_until(self, token):
        lines = []
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError("lc0 exited unexpectedly")
            line = line.strip()
            lines.append(line)
            if line.startswith(token):
                return lines

    def policy(self, fen):
        """
        {uci: probability 0..1} over all legal moves, one forward pass.

        lc0 reports castling in king-takes-rook notation (e1h1); python-chess
        uses e1g1. Every `policy.get(move.uci())` lookup in this project
        therefore returned nothing for castling, silently dropping up to 38% of
        the human policy mass. It produced the worst single error in the
        project: 9.O-O was reported as "0% of players at your level play this,
        more likely a slip" while the raw policy held 0.327 for that move --
        the second most likely human move in the position. The correct reading
        was the opposite: a shared blind spot at his level.
        """
        self._send(f"position fen {fen}")
        self._send("go nodes 1")
        lines = self._drain_until("bestmove")
        board = chess.Board(fen)
        out = {}
        for line in lines:
            m = _P_RE.match(line)
            if not m or m.group(1) == "node":
                continue
            uci, p = m.group(1), float(m.group(2)) / 100.0
            try:
                mv = board.parse_uci(uci)          # accepts both castling forms
                uci = board.uci(mv, chess960=False)
            except Exception:
                pass
            out[uci] = out.get(uci, 0.0) + p
        return out

    def close(self):
        try:
            self._send("quit")
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


def findability(policy, best_uci, played_uci):
    """
    Turn a policy distribution into the four-way diagnosis above.
    Returns dict with p_best, p_played, rank_best, and a verdict string.
    """
    if not policy:
        return {}
    ordered = sorted(policy.items(), key=lambda kv: -kv[1])
    ranks = {u: i + 1 for i, (u, _) in enumerate(ordered)}
    p_best = policy.get(best_uci, 0.0)
    p_played = policy.get(played_uci, 0.0)

    if best_uci == played_uci:
        verdict, note = "played_best", "You found the engine's move."
    elif p_best >= 0.20:
        verdict = "should_have_seen"
        note = ("Players at your level find this move about "
                f"{p_best*100:.0f}% of the time. This one is inside your range -- "
                "the pattern is one you can already handle, so the failure is "
                "in the checking routine, not in your knowledge.")
    elif p_best >= 0.05:
        verdict = "findable"
        note = (f"Only about {p_best*100:.0f}% of players at your level find this. "
                "Borderline: worth studying, but not a lapse.")
    else:
        verdict = "genuinely_hard"
        note = (f"Almost nobody at your level plays this ({p_best*100:.1f}%). "
                "Treat it as a new pattern to learn rather than a mistake to scold "
                "yourself for.")

    if p_played >= 0.25 and verdict != "played_best":
        note += (f" Your move is the popular one at your level ({p_played*100:.0f}% "
                 "play it) -- a shared blind spot, so fixing it pays off repeatedly.")
    elif p_played < 0.03 and verdict != "played_best":
        note += (f" Your move is one your rating band almost never plays "
                 f"({p_played*100:.1f}%) -- more likely a slip than a habit; "
                 "check what the clock was doing.")

    return {"band": None, "p_best": p_best, "p_played": p_played,
            "rank_best": ranks.get(best_uci), "n_legal": len(ordered),
            "verdict": verdict, "note": note,
            "top": [(u, p) for u, p in ordered[:3]]}
