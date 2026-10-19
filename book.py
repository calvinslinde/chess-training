"""Opening identification and book departure, from the lichess ECO database (3,810 lines)."""

import os, csv, chess, chess.pgn, io
from paths import P  # data folder; see paths.py

DATA = P("data/eco_all.tsv")
_BY_EPD = None


def _load():
    global _BY_EPD
    if _BY_EPD is not None:
        return _BY_EPD
    _BY_EPD = {}
    with open(DATA, encoding="utf-8") as fh:
        for row in csv.reader(fh, delimiter="\t"):
            if len(row) < 3:
                continue
            eco, name, moves = row[0], row[1], row[2]
            b = chess.Board()
            try:
                for tok in moves.split():
                    if tok.endswith("."):
                        continue
                    tok = tok.split(".")[-1]
                    if tok:
                        b.push_san(tok)
            except Exception:
                continue
            _BY_EPD[b.epd()] = (eco, name, b.ply())
    return _BY_EPD


def identify(game):
    """Deepest ECO position the game passes through, and the ply it left book."""
    byepd = _load()
    board = game.board()
    best, last_ply = None, 0
    for ply, mv in enumerate(game.mainline_moves(), start=1):
        board.push(mv)
        hit = byepd.get(board.epd())
        if not hit:
            break          # book ends at the FIRST miss
        best, last_ply = hit, ply
        if ply > 40:
            break
    # The first version recorded the LAST matching ply anywhere in the first 40,
    # with no contiguity requirement -- so a game matching at plies [1,2,4,10]
    # reported 10 book plies and silently exempted the 6 non-matching moves from
    # any criticism. classify() returns BOOK before it ever checks for an error.
    return {"eco": best[0] if best else None,
            "name": best[1] if best else None,
            "book_plies": last_ply}


def in_book(board_epd):
    return _load().get(board_epd)
