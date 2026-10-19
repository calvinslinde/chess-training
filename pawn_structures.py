"""Named pawn-structure recognition for python-chess.

A faithful port of the pawn-structure classifier from Siderite/lichessTools
(``tools/showPawnStructure/tool.js``, MIT licence) onto ``python-chess``.

The idea: reduce a position to a canonical string key describing only the pawn
skeleton, then compare that key against a database of ~47 keys naming ~40
classical structures (Carlsbad, Maroczy Bind, Hedgehog, Dragon, d5 Chain, ...).
Comparison is fuzzy, so near-misses still resolve to the right family.

Key layout
----------
``structure_key`` emits 39 characters, 34 of them significant (5 are spaces
that act as group separators and are stripped before comparison).  Groups, in
order of decreasing importance:

    idx  0- 7   my d, my e, my c, their d, their e, their c, Qside, Kside
    idx  8-13   my f, my b, my g, their f, their b, their g
    idx 14-17   my a, my h, their a, their h
    idx 18-23   doubled flags for d, e, c (mine then theirs)
    idx 24-29   doubled flags for f, b, g (mine then theirs)
    idx 30-33   doubled flags for a, h (mine then theirs)

(indices above are into the space-stripped key; the raw key has spaces after
positions 7, 13, 17, 23 and 29.)

Per-file character:
    ``X``       no pawn on that file
    ``0``-``3`` squares the pawn has advanced from its home rank

Majority characters (``Q``/``K`` slots):
    ``M`` I have more pawns on that wing, ``T`` fewer, ``X`` equal.

Because database keys are short (8, 14 or 18 chars), similarity only compares
the leading, most-significant part of the key -- a-file and h-file detail and
the doubled flags are ignored for the 8-character entries.

"Reversed" matches
------------------
Each database entry is stored from one side's point of view.  ``opposing_key``
produces the colour-flipped key so a single entry matches both colours; such a
match is reported with ``reversed=True`` (lichessTools shows it as "name (R)").

Licence note: the algorithm and the bundled ``data/pawn_structures.json`` come
from lichessTools by Siderite Zackwehdex, MIT licensed.
"""

from __future__ import annotations

import json
import math
import os
from typing import Any, Dict, List, Optional, Sequence

import chess

__all__ = [
    "DATA_PATH",
    "load_database",
    "board_grid",
    "structure_key",
    "opposing_key",
    "similarity",
    "similarity_pct",
    "classify",
    "best_match",
    "describe",
]

DATA_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data", "pawn_structures.json"
)

# Exactly the primitive list from tool.js getStructure().
_TEMPLATE: Sequence[str] = (
    "PM3", "PM4", "PM2", "PT3", "PT4", "PT2", "Q", "K", " ",
    "PM5", "PM1", "PM6", "PT5", "PT1", "PT6", " ",
    "PM0", "PM7", "PT0", "PT7", " ",
    "LM3", "LM4", "LM2", "LT3", "LT4", "LT2", " ",
    "LM5", "LM1", "LM6", "LT5", "LT1", "LT6", " ",
    "LM0", "LM7", "LT0", "LT7",
)

_FLIP_MT = str.maketrans({"M": "T", "T": "M"})

_db_cache: Optional[Dict[str, Dict[str, Any]]] = None


# --------------------------------------------------------------------------
# database
# --------------------------------------------------------------------------

def load_database(path: Optional[str] = None, reload: bool = False) -> Dict[str, Dict[str, Any]]:
    """Load (and memoise) the structure database keyed by structure string."""
    global _db_cache
    if path is None:
        if _db_cache is not None and not reload:
            return _db_cache
        path = DATA_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(
            "pawn structure database not found at %s -- fetch it with:\n"
            "  curl -sLo %s https://raw.githubusercontent.com/Siderite/"
            "lichessTools/master/data/pawnStructures.json" % (path, path)
        )
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if path == DATA_PATH:
        _db_cache = data
    return data


# --------------------------------------------------------------------------
# board -> key
# --------------------------------------------------------------------------

def board_grid(board: chess.Board) -> List[List[Optional[str]]]:
    """8x8 array of piece letters, ``grid[y][x]`` with y=0 the 8th rank."""
    grid: List[List[Optional[str]]] = []
    for y in range(8):
        rank = 7 - y
        row: List[Optional[str]] = []
        for x in range(8):
            piece = board.piece_at(chess.square(x, rank))
            row.append(piece.symbol() if piece is not None else None)
        grid.append(row)
    return grid


def _islands(files: Sequence[int]) -> List[List[int]]:
    """Contiguous runs of occupied files as ``[start, end]`` pairs."""
    out: List[List[int]] = []
    current: Optional[List[int]] = None
    for x in range(8):
        if files[x] == -1:
            current = None
            continue
        if current is not None and current[1] == x - 1:
            current[1] = x
        else:
            current = [x, x]
            out.append(current)
    return out


def _extract(grid: Sequence[Sequence[Optional[str]]]):
    """Port of getStructure()'s scanning phase.

    Returns ``(pawns, led, qside, kside)``.  ``pawns[c][file]`` is -1 for an
    empty file, else squares advanced from the home rank.  ``led[c][file]`` is
    True when the file is doubled (lichessTools calls it "leading").
    """
    pawns = {"w": [-1] * 8, "b": [-1] * 8}
    led = {"w": [False] * 8, "b": [False] * 8}

    for y in range(8):
        row = grid[y]
        for x in range(8):
            sym = row[x]
            if sym == "p":
                # black pawns on ranks 3-1 (y>=5) are out of range and ignored
                if y <= 4:
                    if pawns["b"][x] > -1:
                        led["b"][x] = True
                    pawns["b"][x] = y - 1
            elif sym == "P":
                # white pawns on ranks 6-8 (y<=2) are out of range and ignored
                if y >= 3:
                    if pawns["w"][x] > -1:
                        led["w"][x] = True
                    pawns["w"][x] = 6 - y

    qside = {"w": 0, "b": 0}
    kside = {"w": 0, "b": 0}
    for color in ("w", "b"):
        islands = _islands(pawns[color])
        if not islands:
            continue
        first = islands[0]
        if first[0] < 2:
            qside[color] = min(3, first[1]) - first[0] + 1
        last = islands[-1]
        if last[1] > 5:
            kside[color] = last[1] - max(4, last[0]) + 1
    return pawns, led, qside, kside


def _majority(mine: int, theirs: int) -> str:
    if mine > theirs:
        return "M"
    if mine < theirs:
        return "T"
    return "X"


def structure_key(board: chess.Board, black_orientation: bool = False) -> str:
    """Canonical pawn-skeleton key for ``board``.

    ``black_orientation=True`` swaps the "me"/"they" roles so the key is built
    from Black's point of view.  Note that ``classify`` does not need this: it
    already tries the colour-flipped key via :func:`opposing_key`.
    """
    pawns, led, qside, kside = _extract(board_grid(board))
    me, they = ("b", "w") if black_orientation else ("w", "b")

    out: List[str] = []
    for prim in _TEMPLATE:
        if prim == " ":
            out.append(" ")
        elif prim == "Q":
            out.append(_majority(qside[me], qside[they]))
        elif prim == "K":
            out.append(_majority(kside[me], kside[they]))
        elif prim[0] == "P":
            color = me if prim[1] == "M" else they
            value = pawns[color][int(prim[2])]
            out.append("X" if value < 0 else str(value))
        elif prim[0] == "L":
            color = me if prim[1] == "M" else they
            out.append("L" if led[color][int(prim[2])] else "X")
        else:  # pragma: no cover - template is a constant
            raise ValueError("Unknown primitive: %r" % prim)
    return "".join(out)


def opposing_key(key: str) -> str:
    """Colour-flipped key: the same skeleton seen from the other side."""
    if len(key) != 39:
        raise ValueError("expected a 39-character structure key, got %d" % len(key))

    def s(start: int, length: int) -> str:
        return key[start:start + length]

    return (
        s(3, 3) + s(0, 3) + s(6, 2).translate(_FLIP_MT) + " "
        + s(12, 3) + s(9, 3) + " "
        + s(18, 2) + s(16, 2) + " "
        + s(24, 3) + s(21, 3) + " "
        + s(31, 3) + s(28, 3) + " "
        + s(37, 2) + s(35, 2)
    )


# --------------------------------------------------------------------------
# similarity
# --------------------------------------------------------------------------

def _js_round(value: float) -> int:
    """JavaScript ``Math.round`` -- halves go up, not to even."""
    return int(math.floor(value + 0.5))


def similarity_pct(key_search: str, key_named: str) -> int:
    """Integer 0..100 similarity, bit-for-bit the JS ``keySimilarityDirect``."""
    a = key_search.replace(" ", "")
    b = key_named.replace(" ", "")
    length = min(len(a), len(b))
    if length == 0:
        return 0
    sim = 0.0
    prefix_over = False
    for i in range(length):
        ca, cb = a[i], b[i]
        if ca == cb:
            sim += 0.8 if prefix_over else 1.0
        else:
            prefix_over = True
            # partial credit only when both are pawn ranks and the searched
            # pawn is LESS advanced than the named one (structure not yet
            # fully realised).  'X', 'M', 'T', 'L' are NaN in the original.
            if ca.isdigit() and cb.isdigit() and int(ca) < int(cb):
                sim += 0.2
    return _js_round(sim * 100.0 / length)


def similarity(key_search: str, key_named: str) -> float:
    """Similarity in 0..1.  Quantised to 1/100 to match the JS thresholds."""
    return similarity_pct(key_search, key_named) / 100.0


# --------------------------------------------------------------------------
# classification
# --------------------------------------------------------------------------

def classify(
    board: chess.Board,
    threshold: float = 0.90,
    black_orientation: bool = False,
    database: Optional[Dict[str, Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Name the pawn structure on ``board``.

    Returns a list of ``{"name", "eco_or_url", "similarity", "reversed",
    "key"}`` dicts sorted best-first, keeping only matches at or above
    ``threshold`` (1.0 means an exact match).  ``reversed=True`` means the
    named structure was matched with the colours swapped.  Duplicate names are
    collapsed to their best-scoring entry.  Empty list means "no named
    structure".
    """
    db = load_database() if database is None else database
    key = structure_key(board, black_orientation)
    opp = opposing_key(key)

    rows: List[Dict[str, Any]] = []
    # primary first so that ties keep the non-reversed reading (stable sort)
    for search, is_rev in ((key, False), (opp, True)):
        for db_key, value in db.items():
            rows.append({
                "name": value.get("name"),
                "eco_or_url": value.get("url"),
                "similarity": similarity(search, db_key),
                "reversed": is_rev,
                "key": db_key,
            })
    rows.sort(key=lambda r: -r["similarity"])

    seen = set()
    out: List[Dict[str, Any]] = []
    for row in rows:
        if row["similarity"] < threshold:
            continue
        ident = (row["name"], row["reversed"])
        if ident in seen:
            continue
        seen.add(ident)
        out.append(row)
    return out


def best_match(
    board: chess.Board,
    fuzzy: bool = True,
    only_named: bool = True,
    black_orientation: bool = False,
    database: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Optional[Dict[str, Any]]:
    """Line-for-line port of lichessTools' ``getStructureName``.

    Differs from :func:`classify` in returning at most one result and, when
    ``only_named=False``, falling back to a raw-key "approximation" record
    carrying the nearest named structure under ``best``.
    """
    db = load_database() if database is None else database
    threshold = 90 if fuzzy else 100
    key = structure_key(board, black_orientation)

    def get_arr(search: str, is_rev: bool) -> List[Dict[str, Any]]:
        arr = [{
            "name": v.get("name"),
            "eco_or_url": v.get("url"),
            "similarity": similarity_pct(search, k) / 100.0,
            "reversed": is_rev,
            "key": k,
        } for k, v in db.items()]
        arr.sort(key=lambda o: -o["similarity"])
        return arr

    arr = get_arr(key, False)
    if arr and arr[0]["similarity"] == 1.0:
        return arr[0]
    opp_arr = get_arr(opposing_key(key), True)
    if opp_arr and opp_arr[0]["similarity"] == 1.0:
        return opp_arr[0]

    # NOTE: the JS picks the opposing entry on a tie (`arr[0] > oppArr[0] ?
    # arr[0] : oppArr[0]`).  Preserved here for fidelity.
    item = arr[0] if arr[0]["similarity"] > opp_arr[0]["similarity"] else opp_arr[0]
    if item["similarity"] >= threshold / 100.0:
        return item
    if only_named:
        return None
    return {"name": key.split(" ")[0], "eco_or_url": None,
            "similarity": item["similarity"], "reversed": item["reversed"],
            "key": None, "best": item}


def describe(board: chess.Board, threshold: float = 0.90) -> str:
    """One-line human summary, e.g. ``Carlsbad (100%)`` or ``Dragon (R, 93%)``."""
    matches = classify(board, threshold)
    if not matches:
        return "unnamed (%s)" % structure_key(board)
    top = matches[0]
    tag = "R, " if top["reversed"] else ""
    return "%s (%s%d%%)" % (top["name"], tag, round(top["similarity"] * 100))


# ==========================================================================
# self-test
# ==========================================================================

if __name__ == "__main__":
    import sys
    import unicodedata

    # ---------------------------------------------------------------- cases
    # Every case is built by PLAYING a real opening line, never by pasting a
    # FEN.  ``wp``/``bp`` are the pawn squares the line is claimed to reach and
    # are asserted with python-chess BEFORE any classification is checked, so a
    # wrong move order fails loudly as a PREMISE error instead of silently
    # testing the wrong position.
    CASES = [
        dict(
            label="Carlsbad (QGD Exchange)",
            moves="d4 d5 c4 e6 Nc3 Nf6 cxd5 exd5 Bg5 Be7 e3",
            wp="a2 b2 d4 e3 f2 g2 h2",
            bp="a7 b7 c7 d5 f7 g7 h7",
            expect="Carlsbad", rev=False, kind="exact",
        ),
        dict(
            label="Orthodox Exchange (…c6)",
            moves="d4 d5 c4 e6 Nc3 Nf6 cxd5 exd5 Bg5 Be7 e3 O-O Bd3 c6",
            wp="a2 b2 d4 e3 f2 g2 h2",
            bp="a7 b7 c6 d5 f7 g7 h7",
            expect="Orthodox Exchange", rev=False, kind="exact",
        ),
        dict(
            label="Queen's Isolani (Nimzo IQP)",
            moves="d4 Nf6 c4 e6 Nc3 Bb4 e3 c5 Bd3 d5 Nf3 O-O O-O dxc4 Bxc4 cxd4 exd4",
            wp="a2 b2 d4 f2 g2 h2",
            bp="a7 b7 e6 f7 g7 h7",
            expect="Queen's Isolani", rev=False, kind="exact",
        ),
        dict(
            label="Queen's Isolani REVERSED (Tarrasch)",
            moves="d4 d5 c4 e6 Nc3 c5 cxd5 exd5 Nf3 Nc6 e3 Nf6 Be2 Be7 O-O O-O dxc5 Bxc5",
            wp="a2 b2 e3 f2 g2 h2",
            bp="a7 b7 d5 f7 g7 h7",
            expect="Queen's Isolani", rev=True, kind="exact",
        ),
        dict(
            label="Maróczy Bind (Accel. Dragon)",
            moves="e4 c5 Nf3 Nc6 d4 cxd4 Nxd4 g6 c4 Bg7 Be3 Nf6 Nc3 O-O Be2 d6",
            wp="a2 b2 c4 e4 f2 g2 h2",
            bp="a7 b7 d6 e7 f7 g6 h7",
            expect="Maróczi Bind", rev=False, kind="exact",
        ),
        dict(
            label="Dragon (Sicilian)",
            moves="e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 g6",
            wp="a2 b2 c2 e4 f2 g2 h2",
            bp="a7 b7 d6 e7 f7 g6 h7",
            expect="Dragon", rev=False, kind="exact",
        ),
        dict(
            label="Scheveningen (Sicilian)",
            moves="e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 e6",
            wp="a2 b2 c2 e4 f2 g2 h2",
            bp="a7 b7 d6 e6 f7 g7 h7",
            expect="Scheveningen", rev=False, kind="exact",
        ),
        dict(
            label="Boleslavsky Hole (Najdorf …e5)",
            moves="e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 Be2 e5",
            wp="a2 b2 c2 e4 f2 g2 h2",
            bp="a6 b7 d6 e5 f7 g7 h7",
            expect="Boleslavsky Hole", rev=False, kind="exact",
        ),
        dict(
            label="Hedgehog (English/Sicilian)",
            moves="c4 c5 Nf3 Nf6 d4 cxd4 Nxd4 e6 Nc3 a6 e4 Qc7 Be2 b6 O-O d6",
            wp="a2 b2 c4 e4 f2 g2 h2",
            bp="a6 b6 d6 e6 f7 g7 h7",
            expect="Hedgehog", rev=False, kind="exact",
        ),
        dict(
            label="d5 Chain (KID Mar del Plata)",
            moves="d4 Nf6 c4 g6 Nc3 Bg7 e4 d6 Nf3 O-O Be2 e5 d5",
            wp="a2 b2 c4 d5 e4 f2 g2 h2",
            bp="a7 b7 c7 d6 e5 f7 g6 h7",
            expect="d5 Chain", rev=False, kind="exact",
        ),
        dict(
            label="e5 Chain (French Advance)",
            moves="e4 e6 d4 d5 e5",
            wp="a2 b2 c2 d4 e5 f2 g2 h2",
            bp="a7 b7 c7 d5 e6 f7 g7 h7",
            expect="e5 Chain", rev=False, kind="exact",
        ),
        dict(
            label="Modern Benoni",
            moves="d4 Nf6 c4 c5 d5 e6 Nc3 exd5 cxd5 d6 e4 g6",
            wp="a2 b2 d5 e4 f2 g2 h2",
            bp="a7 b7 c5 d6 f7 g6 h7",
            expect="Modern Benoni", rev=False, kind="exact",
        ),
        dict(
            label="Lopez (Closed Ruy, Chigorin)",
            moves="e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O Be7 Re1 b5 Bb3 d6 c3 O-O h3 Na5 Bc2 c5 d4",
            wp="a2 b2 c3 d4 e4 f2 g2 h3",
            bp="a6 b5 c5 d6 e5 f7 g7 h7",
            expect="Lopez", rev=False, kind="exact",
        ),
        dict(
            label="Closed Sicilian",
            moves="e4 c5 Nc3 Nc6 g3 g6 Bg2 Bg7 d3 d6",
            wp="a2 b2 c2 d3 e4 f2 g3 h2",
            bp="a7 b7 c5 d6 e7 f7 g6 h7",
            expect="Closed Sicilian", rev=False, kind="exact",
        ),
        dict(
            label="Slav (main line, 7.Bxc4)",
            moves="d4 d5 c4 c6 Nf3 Nf6 Nc3 dxc4 a4 Bf5 e3 e6 Bxc4",
            wp="a4 b2 d4 e3 f2 g2 h2",
            bp="a7 b7 c6 e6 f7 g7 h7",
            expect="Slav", rev=False, kind="exact",
        ),
        dict(
            label="Stonewall (Dutch)",
            moves="d4 f5 Nf3 Nf6 Bf4 e6 Nbd2 d5",
            wp="a2 b2 c2 d4 e2 f2 g2 h2",
            bp="a7 b7 c7 d5 e6 f5 g7 h7",
            expect="Stonewall", rev=False, kind="exact",
        ),
        dict(
            label="Panov (Caro-Kann, c4-c5)",
            moves="e4 c6 d4 d5 exd5 cxd5 c4 Nf6 Nc3 Nc6 Bg5 e6 c5",
            wp="a2 b2 c5 d4 f2 g2 h2",
            bp="a7 b7 d5 e6 f7 g7 h7",
            expect="Panov", rev=False, kind="exact",
        ),
        dict(
            label="Grünfeld Center (Exchange)",
            moves="d4 Nf6 c4 g6 Nc3 d5 cxd5 Nxd5 e4 Nxc3 bxc3 Bg7 Nf3 c5",
            wp="a2 c3 d4 e4 f2 g2 h2",
            bp="a7 b7 c5 e7 f7 g6 h7",
            expect="Grünfeld Center", rev=False, kind="fuzzy",
        ),
        dict(
            label="Panov proper (no c4-c5)",
            moves="e4 c6 d4 d5 exd5 cxd5 c4 Nf6 Nc3 e6",
            wp="a2 b2 c4 d4 f2 g2 h2",
            bp="a7 b7 d5 e6 f7 g7 h7",
            expect=None, rev=None, kind="none",
        ),
    ]

    def norm(s):
        """Compare names without tripping over accent composition."""
        return unicodedata.normalize("NFC", s)

    def build(moves):
        b = chess.Board()
        for san in moves.split():
            b.push_san(san)
        return b

    def pawn_squares(b, color):
        return " ".join(sorted(chess.square_name(s) for s in b.pieces(chess.PAWN, color)))

    results = []
    boards = {}

    print("=" * 96)
    print("PAWN STRUCTURE CLASSIFIER - SELF TEST")
    print("=" * 96)

    for case in CASES:
        label = case["label"]
        # --- premise 1: the move sequence is legal ------------------------
        try:
            b = build(case["moves"])
        except Exception as exc:
            results.append((label, "FAIL", "PREMISE: illegal move sequence: %s" % exc, "", ""))
            continue
        # --- premise 2: it really reaches the claimed pawn skeleton -------
        got_w = pawn_squares(b, chess.WHITE)
        got_b = pawn_squares(b, chess.BLACK)
        want_w = " ".join(sorted(case["wp"].split()))
        want_b = " ".join(sorted(case["bp"].split()))
        if got_w != want_w or got_b != want_b:
            results.append((label, "FAIL",
                            "PREMISE: pawns are W[%s] B[%s], expected W[%s] B[%s]"
                            % (got_w, got_b, want_w, want_b), "", ""))
            continue

        boards[label] = b
        key = structure_key(b)
        matches = classify(b)
        top = matches[0] if matches else None
        top_s = ("%s%s %d%%" % (top["name"], " (R)" if top["reversed"] else "",
                                round(top["similarity"] * 100))) if top else "-none-"

        # --- the actual assertion ----------------------------------------
        if case["kind"] == "none":
            ok = top is None
            why = "correctly unnamed" if ok else "expected no match, got %s" % top_s
        elif top is None:
            ok, why = False, "expected %s, got no match" % case["expect"]
        else:
            name_ok = norm(top["name"]) == norm(case["expect"])
            rev_ok = top["reversed"] == case["rev"]
            sim = round(top["similarity"] * 100)
            kind_ok = (sim == 100) if case["kind"] == "exact" else (90 <= sim < 100)
            ok = name_ok and rev_ok and kind_ok
            if ok:
                why = "%s%s at %d%%" % (case["expect"], " (R)" if case["rev"] else "", sim)
            else:
                bad = []
                if not name_ok:
                    bad.append("name %r != %r" % (top["name"], case["expect"]))
                if not rev_ok:
                    bad.append("reversed=%s expected %s" % (top["reversed"], case["rev"]))
                if not kind_ok:
                    bad.append("similarity %d%% not %s" % (sim, case["kind"]))
                why = "; ".join(bad)
        results.append((label, "PASS" if ok else "FAIL", why, key.split(" ")[0], top_s))

    # ------------------------------------------------------- invariant test
    print()
    print("-" * 96)
    print("INVARIANT: structure_key(board.mirror()) == opposing_key(structure_key(board))")
    print("-" * 96)
    inv_bad = []
    for label, b in boards.items():
        k = structure_key(b)
        if structure_key(b.mirror()) != opposing_key(k):
            inv_bad.append(label)
    if inv_bad:
        print("  FAIL on: %s" % ", ".join(inv_bad))
    else:
        print("  PASS on all %d built positions (colour-flip port is exact)" % len(boards))

    # ------------------------------------------------------ similarity unit
    print()
    print("-" * 96)
    print("UNIT: keySimilarity arithmetic (pinned against the JS original)")
    print("-" * 96)
    unit = [
        # identical keys
        ("2XXX1XXT", "2XXX1XXT", 1.00, "identical -> 100%"),
        # one mismatch where the SEARCH pawn is less advanced -> +0.2 credit
        ("20X2X0TM", "21X2X0TM", 0.75, "search e2 vs named e3: partial credit"),
        # same mismatch the other way round -> no credit, and JS half-up round
        ("21X2X0TM", "20X2X0TM", 0.73, "search e3 vs named e2: no credit (72.5 -> 73)"),
        # non-numeric mismatch never earns the 0.2
        ("2XXX1XXM", "2XXX1XXT", 0.88, "majority-flag mismatch: no partial credit"),
        # only the shorter key's length is compared
        ("X2010XMXZZZZZZ", "X2010XMX", 1.00, "long search vs short named key"),
    ]
    unit_bad = 0
    for a, bb, want, note in unit:
        got = similarity(a, bb)
        ok = abs(got - want) < 1e-9
        unit_bad += (not ok)
        print("  [%s] %-52s got %.2f want %.2f" % ("PASS" if ok else "FAIL", note, got, want))

    # ---------------------------------------------- doubled-pawn port quirk
    print()
    print("-" * 96)
    print("QUIRK: doubled pawns (ported as-is from the JS, see report)")
    print("-" * 96)
    quirk_bad = 0
    # Ruy Lopez Exchange: Black gets doubled c-pawns on c7/c6.
    bq = build("e4 e5 Nf3 Nc6 Bb5 a6 Bxc6 dxc6")
    assert pawn_squares(bq, chess.BLACK) == "a6 b7 c6 c7 e5 f7 g7 h7", pawn_squares(bq, chess.BLACK)
    kq = structure_key(bq).replace(" ", "")
    ok = kq[5] == "1" and kq[23] == "L"   # their-c file value, their-c doubled flag
    quirk_bad += (not ok)
    print("  [%s] black doubled c7/c6 -> records the FRONT pawn (c6, value 1) + doubled flag"
          % ("PASS" if ok else "FAIL"))
    # Scandinavian-ish line: White gets doubled f-pawns on f2/f3.
    bq2 = build("e4 d5 exd5 Qxd5 Nc3 Qa5 d4 Nf6 Nf3 Bg4 h3 Bxf3 gxf3")
    assert pawn_squares(bq2, chess.WHITE) == "a2 b2 c2 d4 f2 f3 h3", pawn_squares(bq2, chess.WHITE)
    kq2 = structure_key(bq2).replace(" ", "")
    ok2 = kq2[8] == "0" and kq2[24] == "L"  # my-f file value, my-f doubled flag
    quirk_bad += (not ok2)
    print("  [%s] white doubled f2/f3 -> records the REAR pawn (f2, value 0) + doubled flag"
          % ("PASS" if ok2 else "FAIL"))
    print("       (asymmetric on purpose: it is what tool.js does)")

    # ------------------------------------------------------------ the table
    print()
    print("=" * 96)
    print("%-38s %-5s %-16s %s" % ("STRUCTURE TEST", "RES", "KEY (head)", "TOP MATCH"))
    print("=" * 96)
    for label, res, why, key, top_s in results:
        print("%-38s %-5s %-16s %s" % (label[:38], res, key, top_s))
        if res == "FAIL":
            print("%-38s       -> %s" % ("", why))
    npass = sum(1 for r in results if r[1] == "PASS")
    print("-" * 96)
    print("%d/%d structure tests passed | invariant %s | %d/%d unit | %d/2 quirk"
          % (npass, len(results), "OK" if not inv_bad else "FAILED",
             len(unit) - unit_bad, len(unit), 2 - quirk_bad))

    db = load_database()
    names = []
    for v in db.values():
        if v["name"] not in names:
            names.append(v["name"])
    print()
    print("Database: %d keys, %d distinct names" % (len(db), len(names)))
    print("  " + ", ".join(sorted(names, key=norm)))

    failed = (len(results) - npass) + len(inv_bad) + unit_bad + quirk_bad
    sys.exit(1 if failed else 0)
