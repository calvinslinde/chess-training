#!/usr/bin/env python3
"""Self-test: no engine, no network, under a second.

Checks that every module imports and that the pure-Python parts give known
answers on known positions. The engine-backed checks are in tests_known.py.
"""
import importlib, io, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import chess, chess.pgn

fails = []
def check(name, cond, detail=""):
    print(("PASS  " if cond else "FAIL  ") + name + (f"   <- {detail}" if detail and not cond else ""))
    if not cond:
        fails.append(name)

for m in ["paths", "accuracy", "analyze", "board_svg", "book", "brilliancy", "classify", "coach",
          "deep_game", "diagnose", "difficulty", "drill", "endgame", "features", "fetch", "loss",
          "maia", "motifs", "pawn_structures", "phases", "playout", "rating_estimate", "report2",
          "review", "tactics"]:
    try:
        importlib.import_module(m); ok, err = True, ""
    except Exception as e:
        ok, err = False, repr(e)
    check(f"import {m}", ok, err)

import accuracy, board_svg, book, endgame, features, maia, motifs, paths, pawn_structures, phases, tactics

here = os.path.dirname(os.path.abspath(__file__))
game = chess.pgn.read_game(open(os.path.join(here, "examples", "opera-game.pgn")))
plies = len(list(game.mainline_moves()))
check("example PGN parses (33 plies)", game is not None and plies == 33, str(plies))

try:
    op = book.identify(game)
    check("book: the Opera Game is a Philidor Defence", "Philidor" in str(op), str(op)[:120])
except Exception as e:
    check("book: identify runs", False, repr(e))

try:
    div = phases.divide(game)
    check("phases: divide names the middlegame and endgame plies", isinstance(div, dict) and {"midgame", "endgame"} <= set(div), str(div))
except Exception as e:
    check("phases: divide runs", False, repr(e))

check("accuracy: win_prob(0) is 50%", abs(accuracy.win_prob(0) - 50) < 1e-6, str(accuracy.win_prob(0)))
check("accuracy: +300 cp is well above 50%", accuracy.win_prob(300) > 75, str(accuracy.win_prob(300)))
check("accuracy: a move that keeps the eval scores higher than one that drops it",
      accuracy.move_accuracy(50, 50) > accuracy.move_accuracy(50, 20))
try:
    ga = accuracy.game_accuracy([50, 52, 50, 48, 50, 55, 50, 49, 50], True)
    check("accuracy: game_accuracy is a percentage", 0 <= float(ga) <= 100, str(ga))
except Exception as e:
    check("accuracy: game_accuracy runs", False, repr(e))

# A black knight on d4 attacked by the white queen on d1 and defended by nothing.
b = chess.Board("4k3/8/8/8/3n4/8/8/3QK3 w - - 0 1")
check("motifs: the undefended knight is hanging", motifs.is_hanging(b, chess.D4) is True)
check("motifs: it is not defended", motifs.is_defended(b, chess.D4) is False)
try:
    hp = tactics.hanging_pieces(b, chess.BLACK)
    flat = []
    for x in (hp or []):
        flat.extend(list(x) if isinstance(x, (tuple, list)) else [x])
    check("tactics: hanging_pieces finds d4", chess.D4 in flat or "d4" in str(hp), str(hp)[:100])
except Exception as e:
    check("tactics: hanging_pieces runs", False, repr(e))

# White knight d5 to e7 forks the black king on g8 and queen on c8.
b2 = chess.Board("2q3k1/8/8/3N4/8/8/8/4K3 w - - 0 1")
try:
    check("tactics: Ne7 is a fork", bool(tactics.is_fork(b2, chess.Move.from_uci("d5e7"))))
except Exception as e:
    check("tactics: is_fork runs", False, repr(e))

start = chess.Board()
check("motifs: material is level at the start", motifs.material_diff(start, chess.WHITE) == 0)
check("endgame: 32 pieces at the start", endgame.piece_count(start) == 32, str(endgame.piece_count(start)))
check("maia: 1460 maps to the 1500 band", maia.nearest_band(1460) == 1500, str(maia.nearest_band(1460)))
check("maia: 900 maps to the 1100 band", maia.nearest_band(900) == 1100, str(maia.nearest_band(900)))
try:
    fx = features.features(start, chess.WHITE)
    check("features: returns a dict for the start position", isinstance(fx, dict) and len(fx) > 0)
except Exception as e:
    check("features: runs", False, repr(e))
try:
    d = pawn_structures.describe(chess.Board("rnbqkbnr/pp3ppp/4p3/2pp4/3P4/2P1P3/PP3PPP/RNBQKBNR w KQkq - 0 4"))
    check("pawn_structures: describe returns text", isinstance(d, str), repr(d)[:80])
except Exception as e:
    check("pawn_structures: describe runs", False, repr(e))
svg = board_svg.board_svg(start.fen())
check("board_svg: produces an SVG", isinstance(svg, str) and svg.lstrip().startswith("<svg"))
p = paths.P("reports/selftest-probe.json")
check("paths: files live under CHESS_HOME", p.startswith(paths.HOME) and os.path.isdir(os.path.dirname(p)))

print()
if fails:
    print(f"{len(fails)} check(s) failed:", ", ".join(fails)); sys.exit(1)
print("all checks passed")
