"""
motifs.py -- tactical motif detection over python-chess.

Every detector has the same shape::

    detect_<motif>(board: chess.Board, move: chess.Move) -> dict | None

`board` is the position BEFORE `move`.  A detector returns ``None`` when the
motif is absent, otherwise::

    {"motif": str, "targets": [<square names>], "detail": str}

Motifs that genuinely need a continuation line take an extra optional
``pv`` argument and return ``None`` when it is missing rather than guessing:
deflection, interference, zwischenzug, overloaded_defender.

`pv` may either start with `move` itself (engine style, ``pv[0] == move``) or
be the continuation after `move`.  Both are accepted; the line is truncated at
the first illegal move rather than raising.

Written against the algorithms in kmike/chess-tactics (MIT),
CormacmacmCormac/chess_stats (MIT) and ornicar/lichess-puzzler (AGPL, read for
approach only).  No code is copied from any of them.

Values are in PAWN UNITS (P=1, N=B=3, R=5, Q=9), not centipawns.
"""

from __future__ import annotations

import chess

__all__ = [
    "VALUE", "piece_value", "material", "material_diff",
    "legal_attackers", "least_valuable_attacker", "exchange_eval",
    "capture_see", "is_defended", "is_hanging", "can_be_captured",
    "is_in_bad_spot", "is_trapped",
    "detect_fork", "detect_absolute_pin", "detect_relative_pin",
    "detect_skewer", "detect_discovered_attack", "detect_discovered_check",
    "detect_double_check", "detect_x_ray", "detect_hanging_piece",
    "detect_leaves_hanging", "detect_trapped_piece",
    "detect_back_rank_weakness", "detect_overloaded_defender",
    "detect_deflection", "detect_capturing_defender", "detect_interference",
    "detect_zwischenzug", "detect_promotion_threat",
    "detect_smothered_mate_pattern", "detect_attacks_loose_piece", "detect_all",
]

# --------------------------------------------------------------------------- #
#  values
# --------------------------------------------------------------------------- #

VALUE = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 1000,
    None: 0,
}

SLIDERS = (chess.BISHOP, chess.ROOK, chess.QUEEN)


def piece_value(piece_type) -> int:
    return VALUE[piece_type]


def _val(board: chess.Board, square: chess.Square) -> int:
    return VALUE[board.piece_type_at(square)]


def material(board: chess.Board, color: chess.Color) -> int:
    """Material of `color` in pawn units, kings excluded."""
    return sum(
        len(board.pieces(pt, color)) * VALUE[pt]
        for pt in (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN)
    )


def material_diff(board: chess.Board, color: chess.Color) -> int:
    return material(board, color) - material(board, not color)


def _name(square: chess.Square) -> str:
    return chess.square_name(square)


def _names(squares) -> list:
    return [chess.square_name(s) for s in squares]


def _desc(board: chess.Board, square: chess.Square) -> str:
    """'Nf6'-ish description of what stands on `square`."""
    p = board.piece_at(square)
    if p is None:
        return chess.square_name(square)
    sym = "" if p.piece_type == chess.PAWN else p.symbol().upper()
    return f"{sym}{chess.square_name(square)}"


# --------------------------------------------------------------------------- #
#  attacks that are actually available
# --------------------------------------------------------------------------- #

def legal_attackers(
    board: chess.Board,
    color: chess.Color,
    square: chess.Square,
    *,
    ignore_check: bool = False,
) -> chess.SquareSet:
    """Attackers of `square` that could really capture there.

    Unlike ``board.attackers`` this excludes

    * absolutely pinned pieces, unless the target lies on the pinning ray;
    * the king, when the square is defended;
    * pieces frozen because their own king is in check and the capture would
      neither take the checker nor block the check.

    This is the piece most naive SEE implementations get wrong.
    """
    king = board.king(color)
    in_check = False
    resolving = chess.SquareSet()
    n_checkers = 0
    if not ignore_check and king is not None:
        checkers = board.attackers(not color, king)
        n_checkers = len(checkers)
        in_check = n_checkers > 0
        if n_checkers == 1:
            checker = next(iter(checkers))
            # capturing the checker resolves it; so does interposing on the
            # line between a sliding checker and the king
            resolving = chess.SquareSet.between(checker, king)
            resolving.add(checker)

    out = chess.SquareSet()
    for attacker in board.attackers(color, square):
        if board.piece_type_at(attacker) == chess.KING:
            # a king may not capture onto a defended square
            if board.attackers(not color, square):
                continue
        else:
            # absolutely pinned: may only move along the pinning ray
            if square not in board.pin(color, attacker):
                continue
            if in_check:
                # only a king move can answer a double check
                if n_checkers > 1:
                    continue
                if square not in resolving:
                    continue
        out.add(attacker)
    return out


def least_valuable_attacker(board, color, square, *, ignore_check=False):
    attackers = legal_attackers(board, color, square, ignore_check=ignore_check)
    if not attackers:
        return None
    return min(attackers, key=lambda a: _val(board, a))


def _is_in_check(board: chess.Board, color: chess.Color) -> bool:
    king = board.king(color)
    if king is None:
        return False
    return bool(board.attackers(not color, king))


def _capture_move(board: chess.Board, frm: chess.Square, to: chess.Square) -> chess.Move:
    """Build the capture move, auto-queening a pawn that reaches the last rank."""
    if board.piece_type_at(frm) == chess.PAWN and chess.square_rank(to) in (0, 7):
        return chess.Move(frm, to, promotion=chess.QUEEN)
    return chess.Move(frm, to)


# --------------------------------------------------------------------------- #
#  static exchange evaluation
# --------------------------------------------------------------------------- #

def exchange_eval(
    board: chess.Board,
    color: chess.Color,
    square: chess.Square,
    *,
    occupant_value: int | None = None,
    _ignore_check: bool | None = None,
    _depth: int = 0,
) -> int:
    """Material `color` wins by starting an exchange on `square`.

    The first capture is NOT forced, so the result is never negative: either
    side may stand pat.  Batteries and x-rays fall out for free because each
    capture is actually played on a copy of the board.
    """
    if _ignore_check is None:
        # If the *other* side is in check it cannot be their move, so the
        # "frozen by check" filter would give nonsense.  Disable it, as
        # chess-tactics does.
        _ignore_check = _is_in_check(board, not color)

    if _depth > 32:
        return 0

    attacker = least_valuable_attacker(board, color, square, ignore_check=_ignore_check)
    if attacker is None:
        return 0

    move = _capture_move(board, attacker, square)
    if board.is_en_passant(move):
        captured = 1
    else:
        captured = _val(board, square)

    promoting = move.promotion is not None
    attacker_after = VALUE[chess.QUEEN] if promoting else _val(board, attacker)
    promo_bonus = (VALUE[chess.QUEEN] - VALUE[chess.PAWN]) if promoting else 0

    nxt = board.copy(stack=False)
    nxt.push(move)
    recapture = exchange_eval(
        nxt, not color, square,
        occupant_value=attacker_after,
        _ignore_check=_ignore_check,
        _depth=_depth + 1,
    )

    gross = captured + promo_bonus
    if gross < recapture:
        return 0  # not worth starting
    base = occupant_value if occupant_value is not None else captured
    return base + promo_bonus - recapture


def capture_see(board: chess.Board, move: chess.Move) -> int:
    """Material swing for the mover after `move`, which IS forced.

    Can be negative -- that is the whole point.
    """
    color = board.color_at(move.from_square)
    if color is None:
        return 0
    if board.is_en_passant(move):
        captured = 1
    else:
        captured = _val(board, move.to_square)

    promoting = move.promotion is not None
    occupant = VALUE[move.promotion] if promoting else _val(board, move.from_square)
    promo_bonus = (VALUE[move.promotion] - VALUE[chess.PAWN]) if promoting else 0

    nxt = board.copy(stack=False)
    nxt.push(move)
    loss = exchange_eval(nxt, not color, move.to_square, occupant_value=occupant)
    return captured + promo_bonus - loss


# --------------------------------------------------------------------------- #
#  defence / hanging / trapped
# --------------------------------------------------------------------------- #

def is_defended(board: chess.Board, square: chess.Square) -> bool:
    """Is the piece on `square` defended -- including defence *through* an
    attacker (a battery behind the piece that is capturing)?"""
    piece = board.piece_at(square)
    if piece is None:
        return False
    if board.attackers(piece.color, square):
        return True
    # x-ray defence: a friendly slider that only starts defending once the
    # capturing piece has vacated its square
    for attacker in board.attackers(not piece.color, square):
        if board.piece_type_at(attacker) in SLIDERS:
            probe = board.copy(stack=False)
            probe.remove_piece_at(attacker)
            if probe.attackers(piece.color, square):
                return True
    return False


def is_hanging(board: chess.Board, square: chess.Square) -> bool:
    """True when the opponent wins material by starting an exchange here."""
    color = board.color_at(square)
    if color is None:
        return False
    return exchange_eval(board, not color, square) > 0


def can_be_captured(board: chess.Board, square: chess.Square) -> bool:
    """Hanging, or tradeable off by an equal-or-cheaper attacker."""
    color = board.color_at(square)
    if color is None:
        return False
    if is_hanging(board, square):
        return True
    attacker = least_valuable_attacker(board, not color, square)
    if attacker is not None and _val(board, attacker) <= _val(board, square):
        return True
    return False


def can_be_taken_by_lower_piece(board: chess.Board, square: chess.Square) -> bool:
    piece = board.piece_at(square)
    if piece is None:
        return False
    for a in board.attackers(not piece.color, square):
        pt = board.piece_type_at(a)
        if pt != chess.KING and VALUE[pt] < VALUE[piece.piece_type]:
            return True
    return False


def is_in_bad_spot(board: chess.Board, square: chess.Square) -> bool:
    """Attacked, and either undefended or attackable by something cheaper."""
    piece = board.piece_at(square)
    if piece is None:
        return False
    if not board.attackers(not piece.color, square):
        return False
    return (not is_defended(board, square)) or can_be_taken_by_lower_piece(board, square)


def _with_side_to_move(board: chess.Board, color: chess.Color):
    """A copy of `board` with `color` to move, or None if that is incoherent."""
    if board.turn == color:
        return board.copy(stack=False)
    probe = board.copy(stack=False)
    probe.push(chess.Move.null())
    if probe.is_check():
        return None  # `color` was in check on the opponent's turn: illegal
    return probe


def is_trapped(board: chess.Board, square: chess.Square) -> bool:
    """The piece on `square` is attacked and has nowhere safe to go.

    Never mutates `board`.
    """
    piece = board.piece_at(square)
    if piece is None or piece.piece_type in (chess.PAWN, chess.KING):
        return False
    probe = _with_side_to_move(board, piece.color)
    if probe is None or probe.is_check():
        return False
    if probe.is_pinned(piece.color, square):
        return False
    if not is_in_bad_spot(probe, square):
        return False

    for escape in list(probe.legal_moves):
        if escape.from_square != square:
            continue
        grabbed = probe.piece_at(escape.to_square)
        if grabbed and VALUE[grabbed.piece_type] >= VALUE[piece.piece_type]:
            return False  # it can trade itself off profitably enough
        landed = probe.copy(stack=False)
        landed.push(escape)
        if not is_in_bad_spot(landed, escape.to_square):
            return False
    return True


# --------------------------------------------------------------------------- #
#  ray geometry
# --------------------------------------------------------------------------- #

def _behind(board: chess.Board, slider: chess.Square, front: chess.Square):
    """First occupied square beyond `front`, on the slider->front ray."""
    if slider == front:
        return None
    df = chess.square_file(front) - chess.square_file(slider)
    dr = chess.square_rank(front) - chess.square_rank(slider)
    sf = (df > 0) - (df < 0)
    sr = (dr > 0) - (dr < 0)
    f, r = chess.square_file(front), chess.square_rank(front)
    while True:
        f += sf
        r += sr
        if not (0 <= f <= 7 and 0 <= r <= 7):
            return None
        s = chess.square(f, r)
        if board.piece_at(s) is not None:
            return s


def _has_off_ray_move(pos: chess.Board, square: chess.Square, ray) -> bool:
    """Does the piece on `square` have anywhere to go that leaves `ray`?

    A piece that can only travel along the line it is 'pinned' on is not
    pinned at all -- a pawn on d2 in front of a queen on d1 is the classic
    false positive.
    """
    piece = pos.piece_at(square)
    if piece is None:
        return False
    if piece.piece_type == chess.PAWN:
        step = 8 if piece.color == chess.WHITE else -8
        cands = []
        one = square + step
        if 0 <= one <= 63 and pos.piece_at(one) is None:
            cands.append(one)
            home = 1 if piece.color == chess.WHITE else 6
            two = square + 2 * step
            if chess.square_rank(square) == home and 0 <= two <= 63 \
                    and pos.piece_at(two) is None:
                cands.append(two)
        for t in pos.attacks(square):
            tp = pos.piece_at(t)
            if tp is not None and tp.color != piece.color:
                cands.append(t)
    else:
        cands = [
            t for t in pos.attacks(square)
            if pos.piece_at(t) is None or pos.color_at(t) != piece.color
        ]
    return any(t not in ray for t in cands)


def _front_can_take_slider(pos: chess.Board, front: chess.Square,
                           slider: chess.Square) -> bool:
    """Can the 'pinned' piece just capture whatever is pinning it?"""
    if slider not in pos.attacks(front):
        return False
    return (VALUE[pos.piece_type_at(slider)] >= VALUE[pos.piece_type_at(front)]
            or not is_defended(pos, slider))


def _bites(pos: chess.Board, us: chess.Color, front: chess.Square,
           back: chess.Square) -> bool:
    """Would the piece behind actually fall if the front one stepped aside?"""
    if pos.piece_type_at(back) == chess.KING:
        return True
    probe = pos.copy(stack=False)
    probe.remove_piece_at(front)
    return exchange_eval(probe, us, back) > 0


def _alignments(board: chess.Board, color: chess.Color):
    """(slider_sq, front_sq, back_sq) with an enemy piece in front of another
    occupied square, on a line the slider can actually use."""
    out = []
    sliders = chess.SquareSet()
    for pt in SLIDERS:
        sliders |= board.pieces(pt, color)
    for s in sliders:
        for front in board.attacks(s):
            fp = board.piece_at(front)
            if fp is None or fp.color == color:
                continue
            back = _behind(board, s, front)
            if back is None:
                continue
            out.append((s, front, back))
    return out


def _after(board: chess.Board, move: chess.Move):
    """Board after `move`, or None if the move is not legal here."""
    if move not in board.legal_moves:
        return None
    nxt = board.copy(stack=False)
    nxt.push(move)
    return nxt


def _vacated(before: chess.Board, move: chess.Move) -> set:
    """Squares emptied by `move` -- including the rook's on a castle and the
    captured pawn's on an en passant."""
    out = {move.from_square}
    rank = chess.square_rank(move.from_square)
    if before.is_castling(move):
        out.add(chess.square(7 if move.to_square > move.from_square else 0, rank))
    if before.is_en_passant(move):
        out.add(move.to_square + (-8 if before.turn == chess.WHITE else 8))
    return out


def _arrived(before: chess.Board, move: chess.Move) -> set:
    """Squares a piece landed on, including the rook's on a castle."""
    out = {move.to_square}
    if before.is_castling(move):
        rank = chess.square_rank(move.from_square)
        out.add(chess.square(5 if move.to_square > move.from_square else 3, rank))
    return out


def _result(motif, targets, detail):
    return {"motif": motif, "targets": list(targets), "detail": detail}


def _continuation(board: chess.Board, move: chess.Move, pv):
    """Legal continuation AFTER `move`, truncated at the first illegal move."""
    if not pv:
        return None
    rest = list(pv)
    if rest and rest[0] == move:
        rest = rest[1:]
    probe = _after(board, move)
    if probe is None:
        return None
    out = []
    for m in rest:
        if m not in probe.legal_moves:
            break
        probe.push(m)
        out.append(m)
    return out


def _play(board: chess.Board, moves):
    probe = board.copy(stack=False)
    for m in moves:
        if m not in probe.legal_moves:
            break
        probe.push(m)
    return probe


# ==========================================================================  #
#  DETECTORS
# ==========================================================================  #

def detect_fork(board: chess.Board, move: chess.Move):
    """One piece, two or more worthwhile targets, and it cannot simply be taken."""
    after = _after(board, move)
    if after is None:
        return None
    us = board.turn
    them = not us
    sq = move.to_square
    forker = after.piece_at(sq)
    if forker is None or forker.color != us:
        return None

    if can_be_captured(after, sq):
        return None

    def qualifying(pos, from_sq, forker_type):
        found = []
        for t in pos.attacks(from_sq):
            tp = pos.piece_at(t)
            if tp is None or tp.color != them or tp.piece_type == chess.PAWN:
                continue
            if VALUE[tp.piece_type] > VALUE[forker_type]:
                found.append(t)
            elif not is_defended(pos, t) and from_sq not in pos.attackers(them, t):
                found.append(t)
        return found

    targets = qualifying(after, sq, forker.piece_type)
    if len(targets) < 2:
        return None

    mover = board.piece_at(move.from_square)
    before_targets = (
        qualifying(board, move.from_square, mover.piece_type) if mover else []
    )
    if not set(targets) - set(before_targets):
        return None  # the fork was already there; this move did not create it

    checks = after.king(them) in targets
    return _result(
        "fork",
        _names(sorted(targets)),
        f"{_desc(after, sq)} forks " + ", ".join(_desc(after, t) for t in sorted(targets))
        + (" (with check)" if checks else ""),
    )


def detect_absolute_pin(board: chess.Board, move: chess.Move):
    """An enemy piece is newly pinned against its own king."""
    after = _after(board, move)
    if after is None:
        return None
    them = not board.turn

    def pinned(pos):
        return {
            s for s in chess.SquareSet(pos.occupied_co[them])
            if pos.piece_type_at(s) != chess.KING and pos.is_pinned(them, s)
        }

    new = pinned(after) - pinned(board)
    if not new:
        return None

    details = []
    for s in sorted(new):
        pinners = [
            a for a in after.attackers(board.turn, s)
            if after.piece_type_at(a) in SLIDERS
        ]
        who = _desc(after, pinners[0]) if pinners else "a slider"
        details.append(f"{_desc(after, s)} pinned to the king by {who}")
    return _result("absolute_pin", _names(sorted(new)), "; ".join(details))


def detect_relative_pin(board: chess.Board, move: chess.Move):
    """An enemy piece is newly stuck in front of a MORE valuable enemy piece
    (not the king -- that is an absolute pin)."""
    after = _after(board, move)
    if after is None:
        return None
    us = board.turn

    def pins(pos):
        found = {}
        for slider, front, back in _alignments(pos, us):
            fp, bp = pos.piece_at(front), pos.piece_at(back)
            if fp.piece_type == chess.KING:
                continue
            if bp is None or bp.color == us or bp.piece_type == chess.KING:
                continue
            if VALUE[bp.piece_type] <= VALUE[fp.piece_type]:
                continue
            ray = chess.SquareSet(chess.ray(slider, front))
            if not _has_off_ray_move(pos, front, ray):
                continue  # it can only move along the line: not really pinned
            if _front_can_take_slider(pos, front, slider):
                continue  # it can simply capture the pinner
            if not _bites(pos, us, front, back):
                continue  # stepping aside costs nothing: not really a pin
            found[front] = (slider, back)
        return found

    now, was = pins(after), pins(board)
    new = {k: v for k, v in now.items() if k not in was}
    if not new:
        return None
    front = sorted(new)[0]
    slider, back = new[front]
    return _result(
        "relative_pin",
        _names([front, back]),
        f"{_desc(after, front)} is pinned by {_desc(after, slider)} against "
        f"{_desc(after, back)} behind it",
    )


def detect_skewer(board: chess.Board, move: chess.Move):
    """A valuable enemy piece is newly attacked with something worth taking
    directly behind it."""
    after = _after(board, move)
    if after is None:
        return None
    us = board.turn

    def skewers(pos):
        found = {}
        for slider, front, back in _alignments(pos, us):
            fp, bp = pos.piece_at(front), pos.piece_at(back)
            if bp is None or bp.color == us:
                continue
            ray = chess.SquareSet(chess.ray(slider, front))
            if fp.piece_type == chess.KING:
                pass  # a check always compels the king to move
            else:
                if VALUE[fp.piece_type] <= VALUE[bp.piece_type]:
                    continue
                # the front piece must actually be forced to move ...
                if not (VALUE[fp.piece_type] > VALUE[pos.piece_type_at(slider)]
                        or not is_defended(pos, front)):
                    continue
            # ... and it must have somewhere to go that abandons the line
            if not _has_off_ray_move(pos, front, ray):
                continue
            if VALUE[bp.piece_type] < 3 and is_defended(pos, back):
                continue
            if _front_can_take_slider(pos, front, slider):
                continue  # the front piece can just take the attacker
            if not _bites(pos, us, front, back):
                continue
            found[front] = (slider, back)
        return found

    now, was = skewers(after), skewers(board)
    new = {k: v for k, v in now.items() if k not in was}
    if not new:
        return None
    front = sorted(new)[0]
    slider, back = new[front]
    return _result(
        "skewer",
        _names([front, back]),
        f"{_desc(after, slider)} skewers {_desc(after, front)}; "
        f"{_desc(after, back)} is behind it",
    )


def detect_discovered_attack(board: chess.Board, move: chess.Move):
    """Moving one piece uncovers a friendly slider behind it."""
    after = _after(board, move)
    if after is None:
        return None
    us, them = board.turn, not board.turn
    vacated = _vacated(board, move)
    arrived = _arrived(board, move)
    if not vacated:
        return None

    revealed = []
    detail = []
    for pt in SLIDERS:
        for slider in after.pieces(pt, us):
            if slider in arrived:
                continue  # the slider itself moved -- not a discovery
            new_attacks = set(after.attacks(slider)) - set(board.attacks(slider))
            for t in new_attacks:
                tp = after.piece_at(t)
                if tp is None or tp.color != them:
                    continue
                between = set(chess.SquareSet.between(slider, t))
                if not (vacated & between):
                    continue
                if tp.piece_type == chess.KING:
                    continue  # that is a discovered check, reported separately
                worth = (
                    VALUE[tp.piece_type] > VALUE[pt]
                    or not is_defended(after, t)
                )
                if not worth:
                    continue
                revealed.append(t)
                detail.append(f"{_desc(after, slider)} is uncovered onto {_desc(after, t)}")
    if not revealed:
        return None
    return _result("discovered_attack", _names(sorted(set(revealed))), "; ".join(detail))


def detect_discovered_check(board: chess.Board, move: chess.Move):
    """The check comes from a piece other than the one that moved."""
    after = _after(board, move)
    if after is None:
        return None
    checkers = set(after.checkers())
    if not checkers:
        return None
    hidden = checkers - _arrived(board, move)
    if not hidden:
        return None
    return _result(
        "discovered_check",
        _names(sorted(hidden)),
        f"{_desc(board, move.from_square)} steps aside and "
        + ", ".join(_desc(after, c) for c in sorted(hidden))
        + " gives check",
    )


def detect_double_check(board: chess.Board, move: chess.Move):
    after = _after(board, move)
    if after is None:
        return None
    checkers = sorted(after.checkers())
    if len(checkers) < 2:
        return None
    return _result(
        "double_check",
        _names(checkers),
        "double check from " + " and ".join(_desc(after, c) for c in checkers)
        + " -- only a king move answers it",
    )


def detect_x_ray(board: chess.Board, move: chess.Move):
    """A slider newly bears on an enemy piece through exactly one blocker.

    The loosest of the geometric detectors: pins and skewers are special cases
    of it, so `detect_all` reports those first.
    """
    after = _after(board, move)
    if after is None:
        return None
    us, them = board.turn, not board.turn

    def rays(pos):
        found = {}
        sliders = chess.SquareSet()
        for pt in SLIDERS:
            sliders |= pos.pieces(pt, us)
        for s in sliders:
            for front in pos.attacks(s):
                if pos.piece_at(front) is None:
                    continue
                back = _behind(pos, s, front)
                if back is None:
                    continue
                bp = pos.piece_at(back)
                if bp.color != them:
                    continue
                if bp.piece_type != chess.KING and VALUE[bp.piece_type] < 3:
                    continue
                if not _bites(pos, us, front, back):
                    continue  # geometrically an x-ray, tactically nothing
                found[(s, back)] = front
        return found

    now, was = rays(after), rays(board)
    new = {k: v for k, v in now.items() if k not in was}
    if not new:
        return None
    (slider, back), front = sorted(new.items())[0]
    through = after.piece_at(front)
    kind = "own piece" if through.color == us else "enemy piece"
    return _result(
        "x_ray",
        _names([back, front]),
        f"{_desc(after, slider)} x-rays {_desc(after, back)} through the "
        f"{kind} on {_name(front)}",
    )


def _is_mere_recapture(board: chess.Board, move: chess.Move) -> bool:
    """Is `move` just taking back on the square the opponent just took on?

    Needs a move stack.  Without one this returns False, which means
    ``detect_hanging_piece`` will tag ordinary recaptures -- pass boards that
    still carry their history if you care about that.
    """
    if not board.move_stack:
        return False
    rewound = board.copy()
    prev = rewound.pop()
    if prev.to_square != move.to_square:
        return False
    if not rewound.is_capture(prev):
        return False  # they just put a piece there and it hangs: real
    if rewound.is_en_passant(prev):
        taken = 1
    else:
        taken = VALUE[rewound.piece_type_at(prev.to_square)]
    return taken >= _val(board, move.to_square)


def detect_hanging_piece(board: chess.Board, move: chess.Move):
    """The move grabs a loose enemy piece, or newly attacks one."""
    after = _after(board, move)
    if after is None:
        return None
    us, them = board.turn, not board.turn

    if board.is_capture(move) and not board.is_en_passant(move):
        victim = board.piece_at(move.to_square)
        if (victim and victim.piece_type != chess.PAWN
                and not is_defended(board, move.to_square)
                and capture_see(board, move) > 0
                and not _is_mere_recapture(board, move)):
            return _result(
                "hanging_piece",
                [_name(move.to_square)],
                f"wins the undefended {_desc(board, move.to_square)} "
                f"(+{capture_see(board, move)})",
            )

    return None


def detect_attacks_loose_piece(board: chess.Board, move: chess.Move,
                               min_gain: int = 2):
    """The moved piece newly hits an undefended enemy piece.

    Deliberately a SEPARATE tag from `hanging_piece`.  Attacking a loose queen
    with a knight is an attack, not "there is a piece lying around to take",
    and lumping the two together is what makes motif output unreadable.
    """
    after = _after(board, move)
    if after is None:
        return None
    us, them = board.turn, not board.turn
    threats = []
    for t in after.attacks(move.to_square):
        tp = after.piece_at(t)
        if tp is None or tp.color != them:
            continue
        if tp.piece_type in (chess.PAWN, chess.KING):
            continue  # a king is never "hanging"; a pawn is not worth the tag
        if is_defended(after, t):
            continue
        if move.from_square in board.attackers(us, t):
            continue  # already attacked by this very piece
        gain = exchange_eval(after, us, t)
        if gain < min_gain:
            continue
        threats.append(t)
    if not threats:
        return None
    return _result(
        "attacks_loose_piece",
        _names(sorted(threats)),
        "hits the undefended "
        + ", ".join(_desc(after, t) for t in sorted(threats)),
    )


def detect_leaves_hanging(board: chess.Board, move: chess.Move,
                          min_gain: int = 2):
    """The move leaves one of OUR pieces there for the taking.

    Not on the requested list, but this is the one a blunder classifier
    actually wants.  `min_gain` defaults to 2 because at 1 the tag is 60%
    single-pawn exchange noise (measured over real games).
    """
    after = _after(board, move)
    if after is None:
        return None
    us = board.turn
    loose = []
    for s in chess.SquareSet(after.occupied_co[us]):
        if after.piece_type_at(s) == chess.KING:
            continue
        gain = exchange_eval(after, not us, s)
        if gain < max(1, min_gain):
            continue
        if s != move.to_square and is_hanging(board, s):
            continue  # it was already hanging before the move
        loose.append((gain, s))
    if not loose:
        return None
    loose.sort(reverse=True)
    gain, worst = loose[0]
    return _result(
        "leaves_hanging",
        _names([s for _, s in loose]),
        f"leaves {_desc(after, worst)} en prise (-{gain})",
    )


def detect_trapped_piece(board: chess.Board, move: chess.Move):
    """An enemy piece is newly stuck with no safe square."""
    after = _after(board, move)
    if after is None:
        return None
    them = not board.turn

    if board.is_capture(move) and is_trapped(board, move.to_square):
        return _result(
            "trapped_piece",
            [_name(move.to_square)],
            f"collects the trapped {_desc(board, move.to_square)}",
        )

    newly = []
    for s in chess.SquareSet(after.occupied_co[them]):
        if is_trapped(after, s) and not is_trapped(board, s):
            newly.append(s)
    if not newly:
        return None
    return _result(
        "trapped_piece",
        _names(sorted(newly)),
        ", ".join(f"{_desc(after, s)} has no safe square" for s in sorted(newly)),
    )


def detect_back_rank_weakness(board: chess.Board, move: chess.Move):
    """Enemy king boxed in on its own back rank.

    This is a property of the resulting POSITION, not of the move: it will
    report the weakness on every move while the weakness stands.
    """
    after = _after(board, move)
    if after is None:
        return None
    us, them = board.turn, not board.turn
    king = after.king(them)
    if king is None:
        return None
    back_rank = 7 if them == chess.BLACK else 0
    if chess.square_rank(king) != back_rank:
        return None

    step = -8 if them == chess.BLACK else 8
    flight = [king + step]
    if chess.square_file(king) > 0:
        flight.append(king + step - 1)
    if chess.square_file(king) < 7:
        flight.append(king + step + 1)
    flight = [s for s in flight if 0 <= s <= 63]

    boxed = all(
        after.piece_at(s) is not None and after.piece_at(s).color == them
        for s in flight
    )
    if not boxed:
        return None

    if after.is_checkmate():
        if any(chess.square_rank(c) == back_rank for c in after.checkers()):
            return _result(
                "back_rank_weakness",
                [_name(king)] + _names(sorted(after.checkers())),
                f"back-rank mate: the king on {_name(king)} is walled in by its "
                f"own pawns",
            )

    majors = chess.SquareSet()
    for pt in (chess.ROOK, chess.QUEEN):
        majors |= after.pieces(pt, us)
    entries = []
    for f in range(8):
        s = chess.square(f, back_rank)
        if after.piece_at(s) is not None:
            continue
        if not (after.attackers(us, s) & majors):
            continue
        if after.attackers(them, s):
            continue
        entries.append(s)
    if not entries:
        return None
    return _result(
        "back_rank_weakness",
        [_name(king)] + _names(sorted(entries)),
        f"the king on {_name(king)} has no luft; "
        + ", ".join(_name(s) for s in sorted(entries))
        + " is an undefended entry square on the back rank",
    )


def detect_capturing_defender(board: chess.Board, move: chess.Move):
    """The move takes the piece that was holding something else together."""
    after = _after(board, move)
    if after is None:
        return None
    if not board.is_capture(move) or board.is_en_passant(move):
        return None
    us, them = board.turn, not board.turn
    d = move.to_square
    defender = board.piece_at(d)
    if defender is None or defender.color != them:
        return None
    if capture_see(board, move) < 0:
        return None

    stripped = board.copy(stack=False)
    stripped.remove_piece_at(d)

    freed = []
    for t in chess.SquareSet(board.occupied_co[them]):
        if t == d or board.piece_type_at(t) == chess.KING:
            continue
        if d not in board.attackers(them, t):
            continue  # the captured piece was not defending t
        if not board.attackers(us, t):
            continue  # we do not attack t at all
        if exchange_eval(board, us, t) > 0:
            continue  # t was already winnable; nothing was unlocked
        if exchange_eval(stripped, us, t) <= 0:
            continue  # removing the defender did not change anything
        freed.append(t)

    if not freed:
        return None
    return _result(
        "capturing_defender",
        [_name(d)] + _names(sorted(freed)),
        f"{_desc(board, d)} was the guard for "
        + ", ".join(_desc(board, t) for t in sorted(freed))
        + "; taking it unlocks them",
    )


def detect_promotion_threat(board: chess.Board, move: chess.Move):
    """A pawn promotes, or reaches the seventh with promotion available."""
    after = _after(board, move)
    if after is None:
        return None
    us = board.turn

    if move.promotion is not None:
        if capture_see(board, move) < 0:
            return None
        return _result(
            "promotion_threat",
            [_name(move.to_square)],
            f"promotes to {chess.piece_name(move.promotion)} on {_name(move.to_square)}",
        )

    if after.piece_type_at(move.to_square) != chess.PAWN:
        return None
    seventh = 6 if us == chess.WHITE else 1
    if chess.square_rank(move.to_square) != seventh:
        return None

    probe = _with_side_to_move(after, us)
    if probe is None:
        return None
    promos = [
        m for m in probe.legal_moves
        if m.from_square == move.to_square and m.promotion == chess.QUEEN
    ]
    if not promos:
        return None
    safe = []
    for m in promos:
        landed = probe.copy(stack=False)
        landed.push(m)
        if exchange_eval(landed, not us, m.to_square) <= 0:
            safe.append(m.to_square)
    if not safe:
        return None
    return _result(
        "promotion_threat",
        _names(sorted(set(safe))),
        f"the pawn on {_name(move.to_square)} queens next move on "
        + ", ".join(_name(s) for s in sorted(set(safe))),
    )


def detect_smothered_mate_pattern(board: chess.Board, move: chess.Move):
    """Knight check with the king walled in by its own men."""
    after = _after(board, move)
    if after is None:
        return None
    them = not board.turn
    king = after.king(them)
    if king is None or not after.is_check():
        return None
    knights = [c for c in after.checkers() if after.piece_type_at(c) == chess.KNIGHT]
    if not knights:
        return None
    ring = list(after.attacks(king))
    if not all(
        after.piece_at(s) is not None and after.piece_at(s).color == them
        for s in ring
    ):
        return None
    if after.is_checkmate():
        return _result(
            "smothered_mate_pattern",
            [_name(king)] + _names(sorted(knights)),
            f"smothered mate: {_desc(after, knights[0])} and the king on "
            f"{_name(king)} is buried by its own pieces",
        )
    return _result(
        "smothered_mate_pattern",
        [_name(king)] + _names(sorted(knights)),
        f"smothered pattern: the king on {_name(king)} is buried, but "
        f"{_desc(after, knights[0])} can still be met",
    )


# --------------------------------------------------------------------------- #
#  motifs that need a continuation
# --------------------------------------------------------------------------- #

def _sole_duties(board: chess.Board, defender_sq: chess.Square, us: chess.Color):
    """Squares that `defender_sq` alone defends, and that we already attack."""
    them = not us
    duties = []
    for t in chess.SquareSet(board.occupied_co[them]):
        if t == defender_sq or board.piece_type_at(t) == chess.KING:
            continue
        guards = set(board.attackers(them, t))
        if guards != {defender_sq}:
            continue
        if not board.attackers(us, t):
            continue
        duties.append(t)
    return duties


def detect_overloaded_defender(board: chess.Board, move: chess.Move, pv=None):
    """One enemy piece is the sole guard of two things we attack.

    Needs `pv`; returns None without it.
    """
    if pv is None:
        return None
    cont = _continuation(board, move, pv)
    if cont is None or len(cont) < 2:
        return None
    after = _after(board, move)
    if after is None:
        return None
    us, them = board.turn, not board.turn

    overloaded = []
    for d in chess.SquareSet(board.occupied_co[them]):
        if board.piece_type_at(d) == chess.KING:
            continue
        duties = _sole_duties(board, d, us)
        if len(duties) >= 2:
            overloaded.append((d, duties))
    if not overloaded:
        return None

    hits = []
    for d, duties in overloaded:
        if move.to_square == d or move.to_square in duties:
            hits.append((d, duties))              # we took / hit one of them
        elif any(move.to_square in after.attackers(us, t) for t in duties):
            hits.append((d, duties))              # we added pressure to one
        elif move.to_square in board.attacks(d):
            hits.append((d, duties))              # we offered it a third job
    if not hits:
        return None

    start = material_diff(board, us)
    end = material_diff(_play(board, [move] + cont), us)
    if end - start < 2:
        return None

    d, duties = hits[0]
    return _result(
        "overloaded_defender",
        [_name(d)] + _names(sorted(duties)),
        f"{_desc(board, d)} cannot hold "
        + " and ".join(_desc(board, t) for t in sorted(duties))
        + f" at once (line nets {end - start:+d})",
    )


def detect_deflection(board: chess.Board, move: chess.Move, pv=None):
    """A defender is driven off its post, and then the post falls.

    Needs `pv`; returns None without it.
    """
    if pv is None:
        return None
    cont = _continuation(board, move, pv)
    if cont is None or len(cont) < 2:
        return None
    after = _after(board, move)
    if after is None:
        return None
    us, them = board.turn, not board.turn

    reply, follow = cont[0], cont[1]
    d = reply.from_square
    dp = board.piece_at(d)
    if dp is None or dp.color != them:
        return None
    if reply.to_square == d:
        return None

    # was `move` actually a threat against the defender?
    forcing = after.is_check()
    if not forcing:
        attackers = after.attackers(us, d)
        if not attackers:
            return None
        cheapest = min((_val(after, a) for a in attackers), default=99)
        if not (cheapest < VALUE[dp.piece_type] or not is_defended(after, d)):
            return None

    board2 = _play(board, [move, reply])
    for t in _sole_duties(board, d, us):
        if t != follow.to_square:
            continue
        if board2.piece_at(t) is None or board2.color_at(t) != them:
            continue
        if exchange_eval(board, us, t) > 0:
            continue  # it was already winnable -- no deflection needed
        if exchange_eval(board2, us, t) <= 0:
            continue  # still held
        if not board2.is_capture(follow):
            continue
        return _result(
            "deflection",
            [_name(d), _name(t)],
            f"{_desc(board, d)} is driven off and can no longer hold "
            f"{_desc(board, t)}",
        )
    return None


def detect_interference(board: chess.Board, move: chess.Move, pv=None):
    """We plant a piece on the line between a defender and what it defends.

    Needs `pv`; returns None without it.
    """
    if pv is None:
        return None
    cont = _continuation(board, move, pv)
    if cont is None:
        return None
    after = _after(board, move)
    if after is None:
        return None
    us, them = board.turn, not board.turn
    block = move.to_square

    for d in chess.SquareSet(board.occupied_co[them]):
        if board.piece_type_at(d) not in SLIDERS:
            continue
        if d == block:
            continue
        for t in _sole_duties(board, d, us):
            if block not in chess.SquareSet.between(d, t):
                continue
            if exchange_eval(board, us, t) > 0:
                continue
            if after.piece_at(t) is None or after.color_at(t) != them:
                continue
            if exchange_eval(after, us, t) <= 0:
                continue
            start = material_diff(board, us)
            end = material_diff(_play(board, [move] + cont), us)
            if end - start < 2:
                continue
            return _result(
                "interference",
                [_name(block), _name(d), _name(t)],
                f"{_name(block)} cuts the line from {_desc(board, d)} to "
                f"{_desc(board, t)} (line nets {end - start:+d})",
            )
    return None


def detect_zwischenzug(board: chess.Board, move: chess.Move, pv=None):
    """An in-between move instead of the expected recapture.

    Needs `pv` AND a move stack on `board` (to know what was just captured);
    returns None without either.
    """
    if pv is None or not board.move_stack:
        return None
    cont = _continuation(board, move, pv)
    if cont is None:
        return None

    rewound = board.copy()
    prev = rewound.pop()
    if not rewound.is_capture(prev):
        return None
    c = prev.to_square

    recaptures = [
        m for m in board.legal_moves
        if m.to_square == c and board.is_capture(m)
    ]
    if not recaptures:
        return None
    if move.to_square == c:
        return None  # this IS the recapture

    after = _after(board, move)
    if after is None:
        return None

    forcing = after.is_check()
    if not forcing and board.is_capture(move):
        forcing = (_val(board, move.to_square) >= _val(board, c)
                   and capture_see(board, move) >= 0)
    if not forcing:
        return None

    us = board.turn
    line = [move] + cont
    end_board = _play(board, line)
    returns = any(
        m.to_square == c
        for i, m in enumerate(line) if i % 2 == 0
    )
    if not (returns or end_board.is_checkmate()):
        return None
    if material_diff(end_board, us) < material_diff(board, us):
        return None

    return _result(
        "zwischenzug",
        [_name(move.to_square), _name(c)],
        f"instead of recapturing on {_name(c)}, {_desc(after, move.to_square)} "
        f"is thrown in first"
        + (" (mating)" if end_board.is_checkmate() else ""),
    )


# --------------------------------------------------------------------------- #
#  aggregate
# --------------------------------------------------------------------------- #

_PLAIN = [
    detect_smothered_mate_pattern,
    detect_double_check,
    detect_discovered_check,
    detect_fork,
    detect_skewer,
    detect_absolute_pin,
    detect_relative_pin,
    detect_discovered_attack,
    detect_back_rank_weakness,
    detect_capturing_defender,
    detect_trapped_piece,
    detect_hanging_piece,
    detect_promotion_threat,
    detect_attacks_loose_piece,
    detect_x_ray,
    detect_leaves_hanging,
]

_WITH_PV = [
    detect_deflection,
    detect_interference,
    detect_overloaded_defender,
    detect_zwischenzug,
]

_ORDER = [
    "smothered_mate_pattern", "double_check", "discovered_check", "fork",
    "skewer", "absolute_pin", "relative_pin", "discovered_attack",
    "back_rank_weakness", "deflection", "interference",
    "overloaded_defender", "capturing_defender", "zwischenzug",
    "trapped_piece", "hanging_piece", "promotion_threat",
    "attacks_loose_piece", "x_ray", "leaves_hanging",
]


def detect_all(board: chess.Board, move: chess.Move, pv=None) -> list:
    """Every motif present in `move`, most specific first."""
    if move not in board.legal_moves:
        return []
    found = []
    for fn in _PLAIN:
        try:
            r = fn(board, move)
        except Exception:
            r = None
        if r:
            found.append(r)
    if pv is not None:
        for fn in _WITH_PV:
            try:
                r = fn(board, move, pv)
            except Exception:
                r = None
            if r:
                found.append(r)
    rank = {m: i for i, m in enumerate(_ORDER)}
    found.sort(key=lambda d: rank.get(d["motif"], 99))
    return found


# ==========================================================================  #
#  TESTS
#
#  Every case asserts its own geometric premise with python-chess BEFORE it
#  asserts anything about a detector.  A test that cannot fail is worthless.
# ==========================================================================  #

if __name__ == "__main__":
    import sys
    import traceback

    _RESULTS = []
    _PREMISES = [0]

    class PremiseError(AssertionError):
        pass

    def P(label, cond, extra=""):
        """Assert a premise about the position and print the check."""
        _PREMISES[0] += 1
        ok = bool(cond)
        print(f"      premise {'OK  ' if ok else 'FAIL'} : {label}"
              + (f"  [{extra}]" if extra else ""))
        if not ok:
            raise PremiseError(f"premise failed: {label} {extra}")
        return ok

    def case(name, kind, fn):
        print(f"\n--- {name}  ({kind})")
        try:
            fn()
        except PremiseError as e:
            print(f"      !! {e}")
            _RESULTS.append((name, kind, "PREMISE-FAIL", str(e)))
            return
        except AssertionError as e:
            print(f"      !! {e}")
            _RESULTS.append((name, kind, "FAIL", str(e)))
            return
        except Exception:
            traceback.print_exc()
            _RESULTS.append((name, kind, "ERROR", "exception"))
            return
        _RESULTS.append((name, kind, "PASS", ""))
        print("      => PASS")

    def board_move(fen, uci):
        b = chess.Board(fen)
        P("FEN is a legal position", b.is_valid(), f"status={b.status()}")
        m = chess.Move.from_uci(uci)
        P(f"{uci} is legal here", m in b.legal_moves)
        a = b.copy(stack=False)
        a.push(m)
        return b, m, a

    def expect(res, motif, must_contain=()):
        assert res is not None, f"expected {motif}, got None"
        assert res["motif"] == motif, f"expected {motif}, got {res['motif']}"
        for k in ("motif", "targets", "detail"):
            assert k in res, f"result is missing key {k!r}"
        for sq in must_contain:
            assert sq in res["targets"], \
                f"expected {sq} among targets {res['targets']}"
        print(f"      detector: {res['motif']} {res['targets']} -- {res['detail']}")

    def expect_none(res):
        assert res is None, f"expected None, got {res}"
        print("      detector: None (correct)")

    SS = chess.SquareSet

    # ------------------------------------------------------------------ fork
    def t_fork_pos():
        b, m, a = board_move("3k4/8/8/2q5/5N2/8/8/7K w - - 0 1", "f4e6")
        P("Ne6 attacks the black king on d8", chess.D8 in a.attacks(chess.E6))
        P("Ne6 attacks the black queen on c5", chess.C5 in a.attacks(chess.E6))
        P("d8 really holds the black king",
          a.piece_at(chess.D8) == chess.Piece(chess.KING, chess.BLACK))
        P("c5 really holds the black queen",
          a.piece_at(chess.C5) == chess.Piece(chess.QUEEN, chess.BLACK))
        P("nothing black attacks e6", not a.attackers(chess.BLACK, chess.E6),
          f"attackers={_names(a.attackers(chess.BLACK, chess.E6))}")
        P("the knight did not already hit both from f4",
          not (chess.D8 in b.attacks(chess.F4) and chess.C5 in b.attacks(chess.F4)))
        expect(detect_fork(b, m), "fork", ["d8", "c5"])

    def t_fork_neg():
        # same geometry, but a black pawn guards e6 -- the forker can be taken
        b, m, a = board_move("3k4/3p4/8/2q5/5N2/8/8/7K w - - 0 1", "f4e6")
        P("Ne6 still attacks both king and queen",
          chess.D8 in a.attacks(chess.E6) and chess.C5 in a.attacks(chess.E6))
        P("the d7 pawn attacks e6", chess.D7 in a.attackers(chess.BLACK, chess.E6))
        P("d7 really holds a black pawn",
          a.piece_at(chess.D7) == chess.Piece(chess.PAWN, chess.BLACK))
        expect_none(detect_fork(b, m))

    def t_fork_neg2():
        b = chess.Board()
        for san in ["e4", "e5"]:
            b.push_san(san)
        m = b.parse_san("Nf3")
        a = b.copy(stack=False)
        a.push(m)
        P("Nf3 is an ordinary developing move, no black piece but the e5 pawn "
          "is attacked",
          [s for s in a.attacks(chess.F3)
           if a.piece_at(s) and a.color_at(s) == chess.BLACK] == [chess.E5])
        expect_none(detect_fork(b, m))

    # ------------------------------------------------------------ absolute pin
    def t_abspin_pos():
        b, m, a = board_move("4k3/8/8/4n3/8/8/8/R5K1 w - - 0 1", "a1e1")
        P("nothing is pinned before the move",
          not b.is_pinned(chess.BLACK, chess.E5))
        P("e5 holds a black knight",
          b.piece_at(chess.E5) == chess.Piece(chess.KNIGHT, chess.BLACK))
        P("e8 holds the black king",
          b.piece_at(chess.E8) == chess.Piece(chess.KING, chess.BLACK))
        P("after Re1 the knight IS absolutely pinned",
          a.is_pinned(chess.BLACK, chess.E5))
        expect(detect_absolute_pin(b, m), "absolute_pin", ["e5"])

    def t_abspin_neg():
        b, m, a = board_move("4k3/8/4p3/4n3/8/8/8/R5K1 w - - 0 1", "a1e1")
        occupied = [s for s in SS.between(chess.E1, chess.E8) if a.piece_at(s)]
        P("two black men stand between the rook and the king",
          len(occupied) == 2, _names(occupied))
        P("python-chess agrees the knight is NOT pinned",
          not a.is_pinned(chess.BLACK, chess.E5))
        expect_none(detect_absolute_pin(b, m))

    # ------------------------------------------------------------ relative pin
    def t_relpin_pos():
        b = chess.Board()
        for san in ["d4", "d5", "c4", "e6", "Nc3", "Nf6"]:
            b.push_san(san)
        m = b.parse_san("Bg5")
        a = b.copy(stack=False)
        a.push(m)
        P("f6 holds a black knight",
          a.piece_at(chess.F6) == chess.Piece(chess.KNIGHT, chess.BLACK))
        P("e7 is empty so the line g5-f6-e7-d8 is clear behind the knight",
          a.piece_at(chess.E7) is None)
        P("d8 holds the black queen",
          a.piece_at(chess.D8) == chess.Piece(chess.QUEEN, chess.BLACK))
        P("Bg5 attacks f6", chess.F6 in a.attacks(chess.G5))
        P("the pin is NOT absolute", not a.is_pinned(chess.BLACK, chess.F6))
        expect(detect_relative_pin(b, m), "relative_pin", ["f6", "d8"])

    def t_relpin_neg():
        # Slav move order: the e7 pawn is still home, so Bg5 pins nothing
        b = chess.Board()
        for san in ["d4", "d5", "c4", "c6", "Nc3", "Nf6"]:
            b.push_san(san)
        m = b.parse_san("Bg5")
        a = b.copy(stack=False)
        a.push(m)
        P("f6 holds a black knight",
          a.piece_at(chess.F6) == chess.Piece(chess.KNIGHT, chess.BLACK))
        P("e7 still holds a black PAWN, so the piece behind f6 is worth 1",
          a.piece_at(chess.E7) == chess.Piece(chess.PAWN, chess.BLACK))
        expect_none(detect_relative_pin(b, m))

    # ---------------------------------------------------------------- skewer
    def t_skewer_pos():
        b, m, a = board_move("4q3/8/8/4k3/8/8/8/R5K1 w - - 0 1", "a1e1")
        P("after Re1 black is in check", a.is_check())
        P("the checker is the rook on e1", set(a.checkers()) == {chess.E1})
        P("e5 holds the black king",
          a.piece_at(chess.E5) == chess.Piece(chess.KING, chess.BLACK))
        P("e8 holds the black queen, on the same file, nothing between",
          a.piece_at(chess.E8) == chess.Piece(chess.QUEEN, chess.BLACK)
          and not [s for s in SS.between(chess.E5, chess.E8) if a.piece_at(s)])
        expect(detect_skewer(b, m), "skewer", ["e5", "e8"])

    def t_skewer_neg():
        # identical geometry with the values the other way round: that is a pin
        b, m, a = board_move("4k3/8/8/4n3/8/8/8/R5K1 w - - 0 1", "a1e1")
        P("front piece is the KNIGHT (3), rear piece is the KING",
          a.piece_type_at(chess.E5) == chess.KNIGHT
          and a.piece_type_at(chess.E8) == chess.KING)
        P("so value(front) < value(back): a pin, not a skewer",
          VALUE[chess.KNIGHT] < VALUE[chess.KING])
        expect_none(detect_skewer(b, m))

    # ---------------------------------------------------- discovered attack
    def t_disc_pos():
        b, m, a = board_move("1k5q/8/8/8/3N4/8/8/B5K1 w - - 0 1", "d4f3")
        P("d4 sits between the bishop on a1 and the queen on h8",
          chess.D4 in SS.between(chess.A1, chess.H8))
        P("before the move the bishop does NOT see h8",
          chess.H8 not in b.attacks(chess.A1))
        P("after the move it does", chess.H8 in a.attacks(chess.A1))
        P("h8 holds the black queen, undefended",
          a.piece_at(chess.H8) == chess.Piece(chess.QUEEN, chess.BLACK)
          and not a.attackers(chess.BLACK, chess.H8))
        P("Nf3 itself gives no check", not a.is_check())
        expect(detect_discovered_attack(b, m), "discovered_attack", ["h8"])

    def t_disc_neg():
        # the line opens, but onto a defended pawn -- must NOT fire
        b, m, a = board_move("7k/6p1/8/8/3N4/8/8/B5K1 w - - 0 1", "d4f3")
        P("the line really does open: Ba1 now attacks g7",
          chess.G7 in a.attacks(chess.A1) and chess.G7 not in b.attacks(chess.A1))
        P("g7 is only a PAWN",
          a.piece_at(chess.G7) == chess.Piece(chess.PAWN, chess.BLACK))
        P("and it is defended by the king on h8",
          chess.H8 in a.attackers(chess.BLACK, chess.G7))
        expect_none(detect_discovered_attack(b, m))

    # ----------------------------------------------------- discovered check
    def t_disccheck_pos():
        b, m, a = board_move("7k/8/8/8/3N4/8/8/B5K1 w - - 0 1", "d4b5")
        P("after Nb5 black is in check", a.is_check())
        P("the checker is the BISHOP on a1, not the knight that moved",
          set(a.checkers()) == {chess.A1})
        P("the knight landed on b5, which is not a checker",
          chess.B5 not in a.checkers())
        expect(detect_discovered_check(b, m), "discovered_check", ["a1"])

    def t_disccheck_neg():
        b, m, a = board_move("4q3/8/8/4k3/8/8/8/R5K1 w - - 0 1", "a1e1")
        P("black is in check", a.is_check())
        P("but the checker IS the piece that moved",
          set(a.checkers()) == {m.to_square})
        expect_none(detect_discovered_check(b, m))

    # --------------------------------------------------------- double check
    def t_dblcheck_pos():
        b, m, a = board_move("7k/8/8/4N3/8/8/8/B5K1 w - - 0 1", "e5f7")
        P("e5 sat between the bishop on a1 and the king on h8",
          chess.E5 in SS.between(chess.A1, chess.H8))
        P("after Nf7+ there are exactly TWO checkers",
          len(a.checkers()) == 2, _names(sorted(a.checkers())))
        P("they are the bishop a1 and the knight f7",
          set(a.checkers()) == {chess.A1, chess.F7})
        expect(detect_double_check(b, m), "double_check", ["a1", "f7"])

    def t_dblcheck_neg():
        b, m, a = board_move("7k/8/8/8/3N4/8/8/B5K1 w - - 0 1", "d4b5")
        P("there is a check", a.is_check())
        P("but only one checker", len(a.checkers()) == 1)
        expect_none(detect_double_check(b, m))

    # ---------------------------------------------------------------- x-ray
    def t_xray_pos():
        b, m, a = board_move("4q2k/8/8/8/4R3/8/8/R5K1 w - - 0 1", "a1e1")
        occ = [s for s in SS.between(chess.E1, chess.E8) if a.piece_at(s)]
        P("exactly one piece stands between e1 and e8", len(occ) == 1, _names(occ))
        P("that piece is our own rook on e4", occ == [chess.E4])
        P("e8 holds the black queen",
          a.piece_at(chess.E8) == chess.Piece(chess.QUEEN, chess.BLACK))
        expect(detect_x_ray(b, m), "x_ray", ["e8"])

    def t_xray_neg():
        b, m, a = board_move("4q2k/8/4p3/8/4R3/8/8/R5K1 w - - 0 1", "a1e1")
        occ = [s for s in SS.between(chess.E1, chess.E8) if a.piece_at(s)]
        P("TWO pieces stand between e1 and e8", len(occ) == 2, _names(occ))
        P("the nearest thing behind one blocker is only a pawn on e6",
          a.piece_at(chess.E6) == chess.Piece(chess.PAWN, chess.BLACK))
        expect_none(detect_x_ray(b, m))

    # -------------------------------------------------------- hanging piece
    def t_hang_pos():
        b, m, a = board_move("4k3/8/8/4n3/8/8/8/4R1K1 w - - 0 1", "e1e5")
        P("e5 holds a black knight",
          b.piece_at(chess.E5) == chess.Piece(chess.KNIGHT, chess.BLACK))
        P("nothing black defends e5", not b.attackers(chess.BLACK, chess.E5),
          _names(b.attackers(chess.BLACK, chess.E5)))
        P("Rxe5 is a capture", b.is_capture(m))
        expect(detect_hanging_piece(b, m), "hanging_piece", ["e5"])

    def t_hang_neg():
        # the recapture case that made the old detector over-fire
        b, m, a = board_move("4k3/8/5p2/4n3/8/8/8/4R1K1 w - - 0 1", "e1e5")
        P("Rxe5 is still a capture of a knight", b.is_capture(m)
          and b.piece_type_at(chess.E5) == chess.KNIGHT)
        P("but the f6 pawn defends e5",
          chess.F6 in b.attackers(chess.BLACK, chess.E5))
        P("so the capture loses material", capture_see(b, m) < 0,
          f"see={capture_see(b, m)}")
        expect_none(detect_hanging_piece(b, m))

    # ------------------------------------------------------- leaves hanging
    def t_leaves_pos():
        b, m, a = board_move("6k1/5p2/8/8/8/8/8/4R1K1 w - - 0 1", "e1e6")
        P("f7 holds a black pawn",
          b.piece_at(chess.F7) == chess.Piece(chess.PAWN, chess.BLACK))
        P("that pawn attacks e6", chess.F7 in a.attackers(chess.BLACK, chess.E6))
        P("nothing white defends e6", not a.attackers(chess.WHITE, chess.E6))
        expect(detect_leaves_hanging(b, m), "leaves_hanging", ["e6"])

    def t_leaves_neg():
        b, m, a = board_move("6k1/8/8/8/8/8/8/4R1K1 w - - 0 1", "e1e6")
        P("nothing black attacks e6", not a.attackers(chess.BLACK, chess.E6))
        expect_none(detect_leaves_hanging(b, m))

    def t_leaves_threshold():
        """A one-pawn hang is below the default noise floor but visible at
        min_gain=1 -- proves the threshold is doing real work."""
        b, m, a = board_move("6k1/5p2/8/4P3/8/8/8/6K1 w - - 0 1", "e5e6")
        P("a white PAWN lands on e6",
          a.piece_at(chess.E6) == chess.Piece(chess.PAWN, chess.WHITE))
        P("the black f7 pawn attacks e6",
          chess.F7 in a.attackers(chess.BLACK, chess.E6))
        P("white does not defend e6", not a.attackers(chess.WHITE, chess.E6))
        P("so exactly one pawn is at stake",
          exchange_eval(a, chess.BLACK, chess.E6) == 1,
          f"gain={exchange_eval(a, chess.BLACK, chess.E6)}")
        expect_none(detect_leaves_hanging(b, m))
        expect(detect_leaves_hanging(b, m, min_gain=1), "leaves_hanging", ["e6"])

    # -------------------------------------------------- attacks loose piece
    def t_loose_pos():
        b = chess.Board()
        for san in ["e4", "d5", "exd5", "Qxd5"]:
            b.push_san(san)
        m = b.parse_san("Nc3")
        a = b.copy(stack=False)
        a.push(m)
        P("d5 holds the black queen",
          b.piece_at(chess.D5) == chess.Piece(chess.QUEEN, chess.BLACK))
        P("nothing black defends her", not b.attackers(chess.BLACK, chess.D5),
          _names(b.attackers(chess.BLACK, chess.D5)))
        P("the knight on c3 attacks d5", chess.D5 in a.attacks(chess.C3))
        P("and did not attack d5 from b1",
          chess.D5 not in b.attacks(chess.B1))
        expect(detect_attacks_loose_piece(b, m), "attacks_loose_piece", ["d5"])
        P("but it is NOT tagged as a hanging piece -- nothing is free here",
          detect_hanging_piece(b, m) is None)

    def t_loose_neg():
        b = chess.Board()
        b.push_san("d4")
        b.push_san("d5")
        m = b.parse_san("c4")
        a = b.copy(stack=False)
        a.push(m)
        P("the c4 pawn attacks d5", chess.D5 in a.attacks(chess.C4))
        P("d5 holds a black pawn",
          a.piece_at(chess.D5) == chess.Piece(chess.PAWN, chess.BLACK))
        P("and it is defended by the queen on d8",
          chess.D8 in a.attackers(chess.BLACK, chess.D5))
        expect_none(detect_attacks_loose_piece(b, m))

    # -------------------------------------------------------- trapped piece
    def t_trap_pos():
        b, m, a = board_move("7k/8/8/8/8/1P6/b1P5/3R2K1 w - - 0 1", "d1a1")
        P("a2 holds a black bishop",
          b.piece_at(chess.A2) == chess.Piece(chess.BISHOP, chess.BLACK))
        P("after Ra1 the rook attacks it",
          chess.A1 in a.attackers(chess.WHITE, chess.A2))
        P("nothing defends the bishop", not a.attackers(chess.BLACK, chess.A2))
        P("the bishop is not pinned and black is not in check",
          not a.is_pinned(chess.BLACK, chess.A2) and not a.is_check())
        escapes = [x for x in a.legal_moves if x.from_square == chess.A2]
        P("it has exactly two moves: b1 and Bxb3",
          sorted(x.uci() for x in escapes) == ["a2b1", "a2b3"],
          str(sorted(x.uci() for x in escapes)))
        for e in escapes:
            land = a.copy(stack=False)
            land.push(e)
            P(f"  ...{e.uci()} still lands it under attack and undefended",
              bool(land.attackers(chess.WHITE, e.to_square))
              and not land.attackers(chess.BLACK, e.to_square))
        expect(detect_trapped_piece(b, m), "trapped_piece", ["a2"])

    def t_trap_neg():
        # identical, minus the c2 pawn: Bxb3 is now a safe escape
        b, m, a = board_move("7k/8/8/8/8/1P6/b7/3R2K1 w - - 0 1", "d1a1")
        P("the bishop is attacked by Ra1",
          chess.A1 in a.attackers(chess.WHITE, chess.A2))
        land = a.copy(stack=False)
        land.push(chess.Move(chess.A2, chess.B3))
        P("but Bxb3 lands on a square nothing white attacks",
          not land.attackers(chess.WHITE, chess.B3),
          _names(land.attackers(chess.WHITE, chess.B3)))
        expect_none(detect_trapped_piece(b, m))

    # ----------------------------------------------------------- back rank
    def t_backrank_mate():
        b, m, a = board_move("6k1/5ppp/8/8/8/8/8/4R1K1 w - - 0 1", "e1e8")
        P("black king is on g8, its back rank",
          a.king(chess.BLACK) == chess.G8)
        P("f7, g7, h7 are all black pawns",
          all(a.piece_at(s) == chess.Piece(chess.PAWN, chess.BLACK)
              for s in (chess.F7, chess.G7, chess.H7)))
        P("Re8 is checkmate", a.is_checkmate())
        expect(detect_back_rank_weakness(b, m), "back_rank_weakness", ["g8"])

    def t_backrank_weak():
        b, m, a = board_move("6k1/5ppp/8/8/8/8/8/4R1K1 w - - 0 1", "e1e4")
        P("no check yet", not a.is_check())
        P("king still boxed in by f7/g7/h7",
          all(a.piece_at(s) == chess.Piece(chess.PAWN, chess.BLACK)
              for s in (chess.F7, chess.G7, chess.H7)))
        P("the rook on e4 sees the empty square e8",
          chess.E8 in a.attacks(chess.E4) and a.piece_at(chess.E8) is None)
        P("black does not defend e8", not a.attackers(chess.BLACK, chess.E8))
        expect(detect_back_rank_weakness(b, m), "back_rank_weakness", ["g8", "e8"])

    def t_backrank_neg():
        b, m, a = board_move("6k1/5pp1/7p/8/8/8/8/4R1K1 w - - 0 1", "e1e8")
        P("h7 is empty -- the king has luft", a.piece_at(chess.H7) is None)
        P("so Re8+ is check but NOT mate", a.is_check() and not a.is_checkmate())
        expect_none(detect_back_rank_weakness(b, m))

    # -------------------------------------------------- capturing defender
    def t_capdef_pos():
        b, m, a = board_move("6k1/1p6/2n5/4b3/1N6/8/8/4R1K1 w - - 0 1", "b4c6")
        P("c6 holds a black knight",
          b.piece_at(chess.C6) == chess.Piece(chess.KNIGHT, chess.BLACK))
        P("e5 holds a black bishop",
          b.piece_at(chess.E5) == chess.Piece(chess.BISHOP, chess.BLACK))
        P("the c6 knight is the ONLY defender of e5",
          set(b.attackers(chess.BLACK, chess.E5)) == {chess.C6},
          _names(b.attackers(chess.BLACK, chess.E5)))
        P("our rook on e1 attacks e5",
          chess.E1 in b.attackers(chess.WHITE, chess.E5))
        P("Nxc6 is a capture and is not losing", b.is_capture(m)
          and capture_see(b, m) >= 0, f"see={capture_see(b, m)}")
        P("after Nxc6 nothing defends e5",
          not a.attackers(chess.BLACK, chess.E5))
        expect(detect_capturing_defender(b, m), "capturing_defender", ["c6", "e5"])

    def t_capdef_neg():
        b, m, a = board_move("6k1/1p6/2n2p2/4b3/1N6/8/8/4R1K1 w - - 0 1", "b4c6")
        P("e5 has TWO defenders: c6 and the f6 pawn",
          set(b.attackers(chess.BLACK, chess.E5)) == {chess.C6, chess.F6},
          _names(b.attackers(chess.BLACK, chess.E5)))
        P("after Nxc6 the f6 pawn still holds e5",
          set(a.attackers(chess.BLACK, chess.E5)) == {chess.F6})
        expect_none(detect_capturing_defender(b, m))

    def t_capdef_neg2():
        # a completely ordinary pawn capture in the opening
        b = chess.Board()
        for san in ["e4", "d5"]:
            b.push_san(san)
        m = b.parse_san("exd5")
        P("exd5 is an ordinary capture", b.is_capture(m))
        expect_none(detect_capturing_defender(b, m))

    # -------------------------------------------------------- deflection
    def t_defl_pos():
        b, m, a = board_move("6k1/8/2n5/4b3/1P6/8/8/4R1K1 w - - 0 1", "b4b5")
        P("e5 holds a black bishop, sole defender is the c6 knight",
          b.piece_at(chess.E5) == chess.Piece(chess.BISHOP, chess.BLACK)
          and set(b.attackers(chess.BLACK, chess.E5)) == {chess.C6})
        P("we attack e5 with the rook on e1",
          chess.E1 in b.attackers(chess.WHITE, chess.E5))
        P("b5 attacks the c6 knight", chess.C6 in a.attacks(chess.B5))
        P("the c6 knight is undefended, so it must move",
          not b.attackers(chess.BLACK, chess.C6))
        line = chess.Board("6k1/8/2n5/4b3/1P6/8/8/4R1K1 w - - 0 1")
        pv = []
        for san in ["b5", "Na5", "Rxe5"]:
            mv = line.parse_san(san)
            P(f"pv move {san} is legal", mv in line.legal_moves)
            pv.append(mv)
            line.push(mv)
        P("after the line white is a bishop up",
          material_diff(line, chess.WHITE) - material_diff(b, chess.WHITE) == 3,
          f"delta={material_diff(line, chess.WHITE) - material_diff(b, chess.WHITE)}")
        expect(detect_deflection(b, m, pv), "deflection", ["c6", "e5"])

    def t_defl_neg():
        b, m, a = board_move("6k1/8/2n2p2/4b3/1P6/8/8/4R1K1 w - - 0 1", "b4b5")
        P("e5 has a second defender, the f6 pawn",
          set(b.attackers(chess.BLACK, chess.E5)) == {chess.C6, chess.F6})
        line = chess.Board("6k1/8/2n2p2/4b3/1P6/8/8/4R1K1 w - - 0 1")
        pv = []
        for san in ["b5", "Na5", "Rxe5"]:
            mv = line.parse_san(san)
            P(f"pv move {san} is legal", mv in line.legal_moves)
            pv.append(mv)
            line.push(mv)
        expect_none(detect_deflection(b, m, pv))

    def t_defl_nopv():
        b, m, a = board_move("6k1/8/2n5/4b3/1P6/8/8/4R1K1 w - - 0 1", "b4b5")
        P("this is the position where deflection DOES fire with a pv", True)
        expect_none(detect_deflection(b, m))

    # ------------------------------------------------------- interference
    def t_interf_pos():
        b, m, a = board_move("4k3/8/8/r3n3/1NP5/8/8/4R1K1 w - - 0 1", "b4d5")
        P("the black knight on e5 is absolutely pinned",
          b.is_pinned(chess.BLACK, chess.E5))
        P("its only defender is the rook on a5",
          set(b.attackers(chess.BLACK, chess.E5)) == {chess.A5},
          _names(b.attackers(chess.BLACK, chess.E5)))
        P("d5 lies strictly between a5 and e5",
          chess.D5 in SS.between(chess.A5, chess.E5))
        P("after Nd5 nothing defends e5",
          not a.attackers(chess.BLACK, chess.E5))
        P("the interfering knight is itself defended by the c4 pawn",
          chess.C4 in a.attackers(chess.WHITE, chess.D5))
        line = chess.Board("4k3/8/8/r3n3/1NP5/8/8/4R1K1 w - - 0 1")
        pv = []
        for san in ["Nd5", "Kd7", "Rxe5"]:
            mv = line.parse_san(san)
            P(f"pv move {san} is legal", mv in line.legal_moves)
            pv.append(mv)
            line.push(mv)
        P("the line wins a knight",
          material_diff(line, chess.WHITE) - material_diff(b, chess.WHITE) == 3)
        expect(detect_interference(b, m, pv), "interference", ["d5", "a5", "e5"])

    def t_interf_neg():
        b, m, a = board_move("4k3/8/8/r3n3/1NP5/8/8/4R1K1 w - - 0 1", "b4d3")
        P("d3 is NOT between a5 and e5",
          chess.D3 not in SS.between(chess.A5, chess.E5))
        P("the a5 rook still defends e5",
          chess.A5 in a.attackers(chess.BLACK, chess.E5))
        pv = [m]
        expect_none(detect_interference(b, m, pv))

    def t_interf_nopv():
        b, m, a = board_move("4k3/8/8/r3n3/1NP5/8/8/4R1K1 w - - 0 1", "b4d5")
        expect_none(detect_interference(b, m))

    # ------------------------------------------------- overloaded defender
    OVER = "5r2/3n3k/1b6/8/8/8/8/1R3R1K w - - 0 1"

    def t_over_pos():
        b, m, a = board_move(OVER, "f1f8")
        P("d7 holds a black knight",
          b.piece_at(chess.D7) == chess.Piece(chess.KNIGHT, chess.BLACK))
        P("the d7 knight is the sole defender of the f8 rook",
          set(b.attackers(chess.BLACK, chess.F8)) == {chess.D7},
          _names(b.attackers(chess.BLACK, chess.F8)))
        P("and also the sole defender of the b6 bishop",
          set(b.attackers(chess.BLACK, chess.B6)) == {chess.D7},
          _names(b.attackers(chess.BLACK, chess.B6)))
        P("we attack f8 (Rf1) and b6 (Rb1)",
          chess.F1 in b.attackers(chess.WHITE, chess.F8)
          and chess.B1 in b.attackers(chess.WHITE, chess.B6))
        line = chess.Board(OVER)
        pv = []
        for san in ["Rxf8", "Nxf8", "Rxb6"]:
            mv = line.parse_san(san)
            P(f"pv move {san} is legal", mv in line.legal_moves)
            pv.append(mv)
            line.push(mv)
        P("the line nets a bishop",
          material_diff(line, chess.WHITE) - material_diff(b, chess.WHITE) == 3,
          f"delta={material_diff(line, chess.WHITE) - material_diff(b, chess.WHITE)}")
        expect(detect_overloaded_defender(b, m, pv), "overloaded_defender", ["d7"])

    def t_over_neg():
        fen = "5r2/3n3k/8/8/8/8/8/1R3R1K w - - 0 1"
        b, m, a = board_move(fen, "f1f8")
        P("the d7 knight now guards only f8",
          set(b.attackers(chess.BLACK, chess.F8)) == {chess.D7})
        P("there is no second black piece for it to guard",
          len([s for s in chess.SquareSet(b.occupied_co[chess.BLACK])
               if b.piece_type_at(s) != chess.KING]) == 2)
        line = chess.Board(fen)
        pv = []
        for san in ["Rxf8", "Nxf8"]:
            mv = line.parse_san(san)
            P(f"pv move {san} is legal", mv in line.legal_moves)
            pv.append(mv)
            line.push(mv)
        expect_none(detect_overloaded_defender(b, m, pv))

    def t_over_nopv():
        b, m, a = board_move(OVER, "f1f8")
        expect_none(detect_overloaded_defender(b, m))

    # -------------------------------------------------------- zwischenzug
    ZW = "3qk3/8/8/8/1b6/2N5/1P6/3R2K1 b - - 0 1"

    def _zw_board():
        b = chess.Board(ZW)
        P("start position is legal", b.is_valid())
        mv = b.parse_san("Bxc3")
        P("black's last move Bxc3 is a capture", b.is_capture(mv))
        b.push(mv)
        P("it is now white to move with a move stack",
          b.turn == chess.WHITE and bool(b.move_stack))
        return b

    def t_zw_pos():
        b = _zw_board()
        P("the capture landed on c3", b.peek().to_square == chess.C3)
        P("a recapture bxc3 is available",
          chess.Move.from_uci("b2c3") in b.legal_moves)
        m = b.parse_san("Rxd8+")
        P("Rxd8+ is legal and is NOT the recapture",
          m in b.legal_moves and m.to_square != chess.C3)
        probe = b.copy(stack=False)
        probe.push(m)
        P("Rxd8 gives check", probe.is_check())
        line = b.copy()
        pv = []
        for san in ["Rxd8+", "Kxd8", "bxc3"]:
            mv = line.parse_san(san)
            P(f"pv move {san} is legal", mv in line.legal_moves)
            pv.append(mv)
            line.push(mv)
        P("we do come back and take on c3", pv[2].to_square == chess.C3)
        P("the line gains material",
          material_diff(line, chess.WHITE) > material_diff(b, chess.WHITE),
          f"{material_diff(b, chess.WHITE)} -> {material_diff(line, chess.WHITE)}")
        expect(detect_zwischenzug(b, m, pv), "zwischenzug", ["d8", "c3"])

    def t_zw_neg():
        b = _zw_board()
        m = b.parse_san("bxc3")
        P("this move IS the plain recapture on c3", m.to_square == chess.C3
          and b.is_capture(m))
        expect_none(detect_zwischenzug(b, m, [m]))

    def t_zw_nostack():
        b = chess.Board("3qk3/8/8/8/2b5/8/1P6/3R2K1 w - - 0 1")
        P("this board has no move stack", not b.move_stack)
        m = chess.Move.from_uci("d1d8")
        P("Rxd8 is legal", m in b.legal_moves)
        expect_none(detect_zwischenzug(b, m, [m]))

    # ---------------------------------------------------- promotion threat
    def t_promo_pos():
        b, m, a = board_move("8/P6k/8/8/8/8/8/6K1 w - - 0 1", "a7a8q")
        P("the move carries a queen promotion", m.promotion == chess.QUEEN)
        P("a8 now holds a white queen",
          a.piece_at(chess.A8) == chess.Piece(chess.QUEEN, chess.WHITE))
        P("black cannot take it", not a.attackers(chess.BLACK, chess.A8))
        expect(detect_promotion_threat(b, m), "promotion_threat", ["a8"])

    def t_promo_seventh():
        b, m, a = board_move("7k/8/P7/8/8/8/8/6K1 w - - 0 1", "a6a7")
        P("a white pawn now stands on a7",
          a.piece_at(chess.A7) == chess.Piece(chess.PAWN, chess.WHITE))
        P("a8 is empty", a.piece_at(chess.A8) is None)
        P("and undefended by black", not a.attackers(chess.BLACK, chess.A8))
        expect(detect_promotion_threat(b, m), "promotion_threat", ["a8"])

    def t_promo_neg():
        b, m, a = board_move("7k/8/8/8/P7/8/8/6K1 w - - 0 1", "a4a5")
        P("the pawn lands on rank 5, nowhere near promotion",
          chess.square_rank(m.to_square) == 4)
        expect_none(detect_promotion_threat(b, m))

    def t_promo_neg2():
        b, m, a = board_move("r6k/8/P7/8/8/8/8/6K1 w - - 0 1", "a6a7")
        P("a black rook sits on a8",
          a.piece_at(chess.A8) == chess.Piece(chess.ROOK, chess.BLACK))
        probe = a.copy(stack=False)
        probe.push(chess.Move.null())
        P("so there is NO promotion move available next turn",
          not [x for x in probe.legal_moves
               if x.from_square == chess.A7 and x.promotion])
        expect_none(detect_promotion_threat(b, m))

    # ------------------------------------------------------ smothered mate
    def t_smother_pos():
        b, m, a = board_move("6rk/6pp/3N4/8/8/8/8/6K1 w - - 0 1", "d6f7")
        P("Nf7 is checkmate", a.is_checkmate())
        P("the checker is a knight",
          a.piece_type_at(next(iter(a.checkers()))) == chess.KNIGHT)
        ring = list(a.attacks(a.king(chess.BLACK)))
        P("every square around the king is occupied by black's own men",
          all(a.piece_at(s) and a.color_at(s) == chess.BLACK for s in ring),
          str([(chess.square_name(s), str(a.piece_at(s))) for s in ring]))
        expect(detect_smothered_mate_pattern(b, m), "smothered_mate_pattern", ["h8"])

    def t_smother_neg():
        # a knight check where the king's ring is empty
        b, m, a = board_move("7k/8/8/4N3/8/8/8/B5K1 w - - 0 1", "e5f7")
        P("this is check from a knight",
          a.is_check() and chess.F7 in a.checkers())
        ring = list(a.attacks(a.king(chess.BLACK)))
        P("but the squares around the king are empty, not blocked",
          all(a.piece_at(s) is None for s in ring),
          str(_names(ring)))
        expect_none(detect_smothered_mate_pattern(b, m))

    def t_smother_neg2():
        b, m, a = board_move("6k1/5ppp/8/8/8/8/8/4R1K1 w - - 0 1", "e1e8")
        P("this IS checkmate", a.is_checkmate())
        P("but the checker is a rook, not a knight",
          a.piece_type_at(chess.E8) == chess.ROOK)
        expect_none(detect_smothered_mate_pattern(b, m))

    # -------------------------------------------------------- SEE sanity
    def t_see():
        b = chess.Board("4k3/8/5p2/4n3/8/8/8/4R1K1 w - - 0 1")
        m = chess.Move.from_uci("e1e5")
        P("Rxe5 is legal", m in b.legal_moves)
        P("e5 is defended by the f6 pawn",
          chess.F6 in b.attackers(chess.BLACK, chess.E5))
        see = capture_see(b, m)
        assert see == -2, f"expected SEE -2 (N for R), got {see}"
        print(f"      capture_see(Rxe5) = {see} (correct: knight for rook)")

        b2 = chess.Board("4k3/8/8/4n3/8/8/8/4R1K1 w - - 0 1")
        P("with no defender it is a free knight",
          not b2.attackers(chess.BLACK, chess.E5))
        see2 = capture_see(b2, chess.Move.from_uci("e1e5"))
        assert see2 == 3, f"expected SEE 3, got {see2}"
        print(f"      capture_see(Rxe5) = {see2} (correct: free knight)")

    def t_pinned_attacker_excluded():
        """The subtlety chess-tactics gets right: an absolutely pinned piece is
        not a real attacker."""
        # black knight on d5 is pinned to its king on d8 by our rook on d1,
        # so it does NOT defend the black bishop on e3 -- wait, use a clean one:
        # white bishop on e3, black knight on d5 pinned along the d-file.
        fen = "3k4/8/8/3n4/8/4B3/8/3R2K1 b - - 0 1"
        b = chess.Board(fen)
        P("FEN is legal", b.is_valid())
        P("the black knight on d5 is absolutely pinned by Rd1",
          b.is_pinned(chess.BLACK, chess.D5))
        P("python-chess still lists it as an attacker of e3",
          chess.D5 in b.attackers(chess.BLACK, chess.E3))
        P("but Nxe3 is NOT a legal move",
          chess.Move(chess.D5, chess.E3) not in b.legal_moves)
        la = legal_attackers(b, chess.BLACK, chess.E3)
        assert chess.D5 not in la, f"pinned knight leaked into {_names(la)}"
        print(f"      legal_attackers(black, e3) = {_names(la)} (pinned knight excluded)")
        assert not is_hanging(b, chess.E3), \
            "bishop counted as capturable by a pinned knight"
        print("      is_hanging(Be3) = False (correct)")

    def t_frozen_by_check_excluded():
        """A piece cannot capture while its own king is in check elsewhere."""
        # black king h8 in check from Ra8; the black rook on b2 attacks the
        # white bishop on b6 but is frozen.
        fen = "R6k/8/1B6/8/8/8/1r6/6K1 b - - 0 1"
        b = chess.Board(fen)
        P("FEN is legal", b.is_valid())
        P("black is in check from the a8 rook",
          b.is_check() and set(b.checkers()) == {chess.A8})
        P("the black rook on b2 does attack b6",
          chess.B2 in b.attackers(chess.BLACK, chess.B6))
        P("but Rxb6 is illegal while in check",
          chess.Move(chess.B2, chess.B6) not in b.legal_moves)
        la = legal_attackers(b, chess.BLACK, chess.B6)
        assert chess.B2 not in la, f"frozen rook leaked into {_names(la)}"
        print(f"      legal_attackers(black, b6) = {_names(la)} (frozen rook excluded)")

    def t_checker_capture_allowed():
        """...but capturing the checker itself is still available."""
        # black king e8 in check from a white knight on d6; the black rook on
        # a6 can take it along the sixth rank.
        fen = "4k3/8/r2N4/8/8/8/8/R5K1 b - - 0 1"
        b = chess.Board(fen)
        P("FEN is legal", b.is_valid())
        P("black is in check, and the checker is the knight on d6",
          b.is_check() and set(b.checkers()) == {chess.D6},
          _names(b.checkers()))
        P("Rxd6 is legal (it captures the checker)",
          chess.Move(chess.A6, chess.D6) in b.legal_moves)
        la = legal_attackers(b, chess.BLACK, chess.D6)
        assert chess.A6 in la, f"legal capture of the checker dropped: {_names(la)}"
        print(f"      legal_attackers(black, d6) = {_names(la)} (checker capture kept)")

    def t_check_block_allowed():
        """An interposing square on the check line is reachable; other squares
        the same piece 'attacks' are not."""
        # black king e8 in check from Re1 down the open e-file; the black rook
        # on a5 may interpose on e5 but may not go to a1.
        fen = "4k3/8/8/r7/8/8/8/4R1K1 b - - 0 1"
        b = chess.Board(fen)
        P("FEN is legal", b.is_valid())
        P("black is in check from the rook on e1",
          b.is_check() and set(b.checkers()) == {chess.E1})
        P("e5 lies between the checker and the king",
          chess.E5 in SS.between(chess.E1, chess.E8))
        P("Ra5-e5 (an interposition) IS legal",
          chess.Move(chess.A5, chess.E5) in b.legal_moves)
        P("Ra5-a1 is NOT legal -- it ignores the check",
          chess.Move(chess.A5, chess.A1) not in b.legal_moves)
        P("but python-chess still calls a5 an attacker of a1",
          chess.A5 in b.attackers(chess.BLACK, chess.A1))
        la_e5 = legal_attackers(b, chess.BLACK, chess.E5)
        la_a1 = legal_attackers(b, chess.BLACK, chess.A1)
        assert chess.A5 in la_e5, f"interposition square dropped: {_names(la_e5)}"
        assert chess.A5 not in la_a1, f"frozen rook leaked onto a1: {_names(la_a1)}"
        print(f"      legal_attackers(black, e5) = {_names(la_e5)} (blocking square kept)")
        print(f"      legal_attackers(black, a1) = {_names(la_a1)} (frozen elsewhere)")

    # ------------------------------------------------------------ detect_all
    def t_detect_all():
        b, m, a = board_move("4q3/8/8/4k3/8/8/8/R5K1 w - - 0 1", "a1e1")
        P("Re1 gives check", a.is_check())
        got = detect_all(b, m)
        names = [d["motif"] for d in got]
        print(f"      detect_all -> {names}")
        assert "skewer" in names, f"skewer missing from {names}"
        assert names.index("skewer") < names.index("x_ray") if "x_ray" in names else True
        for d in got:
            assert set(d) == {"motif", "targets", "detail"}, f"bad shape: {d}"

    def t_detect_all_quiet():
        """An ordinary opening move should be quiet or nearly so."""
        b = chess.Board()
        m = b.parse_san("e4")
        P("1.e4 is legal from the start position", m in b.legal_moves)
        got = detect_all(b, m)
        names = [d["motif"] for d in got]
        print(f"      detect_all(1.e4) -> {names}")
        for bad in ("fork", "skewer", "absolute_pin", "relative_pin",
                    "discovered_check", "double_check", "back_rank_weakness",
                    "hanging_piece", "trapped_piece", "leaves_hanging",
                    "smothered_mate_pattern", "promotion_threat",
                    "capturing_defender", "x_ray", "discovered_attack",
                    "attacks_loose_piece"):
            assert bad not in names, f"{bad} fired on 1.e4: {got}"

    def t_detect_all_recapture():
        """The failure mode that bit the old code: firing on a plain recapture."""
        b = chess.Board()
        for san in ["d4", "d5", "c4", "e6", "Nc3", "Nf6", "cxd5"]:
            b.push_san(san)
        m = b.parse_san("exd5")
        P("exd5 is an ordinary pawn recapture on the square white just took on",
          b.is_capture(m) and b.peek().to_square == chess.D5)
        got = detect_all(b, m)
        print(f"      detect_all(exd5) -> {[d['motif'] for d in got]}")
        assert got == [], f"something fired on a plain recapture: {got}"

    def t_detect_all_recapture2():
        """A piece recapture: taking back a bishop that took a knight is NOT
        'winning a hanging piece'."""
        b = chess.Board()
        for san in ["e4", "e5", "Nf3", "Nc6", "Bb5", "a6", "Bxc6"]:
            b.push_san(san)
        m = b.parse_san("dxc6")
        P("white's bishop on c6 captured a knight there",
          b.peek().to_square == chess.C6)
        rew = b.copy()
        prev = rew.pop()
        P("that previous move was itself a capture of a knight",
          rew.is_capture(prev)
          and rew.piece_type_at(chess.C6) == chess.KNIGHT)
        P("the bishop on c6 is genuinely undefended by white",
          not b.attackers(chess.WHITE, chess.C6),
          _names(b.attackers(chess.WHITE, chess.C6)))
        P("so a naive SEE test WOULD call it hanging",
          capture_see(b, m) > 0, f"see={capture_see(b, m)}")
        got = detect_all(b, m)
        print(f"      detect_all(dxc6) -> {[d['motif'] for d in got]}")
        assert "hanging_piece" not in [d["motif"] for d in got], \
            f"hanging_piece fired on a recapture: {got}"

    def t_scandinavian_pin_is_real():
        """1.e4 d5 2.exd5 Qxd5 really does pin g2 to h1 -- a TRUE positive that
        looks like over-firing until you check the geometry."""
        b = chess.Board()
        for san in ["e4", "d5", "exd5"]:
            b.push_san(san)
        m = b.parse_san("Qxd5")
        a = b.copy(stack=False)
        a.push(m)
        P("the queen on d5 attacks g2", chess.G2 in a.attacks(chess.D5))
        P("e4 and f3 are empty, so the diagonal d5-g2 is clear",
          a.piece_at(chess.E4) is None and a.piece_at(chess.F3) is None)
        P("g2 holds a white pawn, h1 a white rook, on that same diagonal",
          a.piece_at(chess.G2) == chess.Piece(chess.PAWN, chess.WHITE)
          and a.piece_at(chess.H1) == chess.Piece(chess.ROOK, chess.WHITE)
          and chess.H1 in SS(chess.ray(chess.D5, chess.G2)))
        P("nothing white defends h1", not a.attackers(chess.WHITE, chess.H1))
        # the load-bearing claim: if the g2 pawn moves, the rook falls
        P("it is white to move after Qxd5", a.turn == chess.WHITE)
        for uci in ("g2g3", "g2g4"):
            step = a.copy(stack=False)
            mv = chess.Move.from_uci(uci)
            P(f"{uci} is available to white", mv in step.legal_moves)
            step.push(mv)
            P(f"  ...and after it Qxh1 wins the rook",
              chess.Move.from_uci("d5h1") in step.legal_moves)
        got = detect_all(b, m)
        names = [d["motif"] for d in got]
        print(f"      detect_all(Qxd5) -> {names}")
        assert "relative_pin" in names, f"real pin missed: {got}"
        for bad in ("fork", "skewer", "absolute_pin", "double_check",
                    "discovered_check", "trapped_piece", "hanging_piece",
                    "smothered_mate_pattern", "back_rank_weakness",
                    "capturing_defender", "promotion_threat",
                    "leaves_hanging", "attacks_loose_piece"):
            assert bad not in names, f"{bad} also fired: {got}"

    # ------------------------------------------------------------------ run
    print("=" * 74)
    print("motifs.py self-test -- every case asserts its geometry first")
    print("=" * 74)

    TESTS = [
        ("fork",                  "positive", t_fork_pos),
        ("fork",                  "negative", t_fork_neg),
        ("fork",                  "negative", t_fork_neg2),
        ("absolute_pin",          "positive", t_abspin_pos),
        ("absolute_pin",          "negative", t_abspin_neg),
        ("relative_pin",          "positive", t_relpin_pos),
        ("relative_pin",          "negative", t_relpin_neg),
        ("skewer",                "positive", t_skewer_pos),
        ("skewer",                "negative", t_skewer_neg),
        ("discovered_attack",     "positive", t_disc_pos),
        ("discovered_attack",     "negative", t_disc_neg),
        ("discovered_check",      "positive", t_disccheck_pos),
        ("discovered_check",      "negative", t_disccheck_neg),
        ("double_check",          "positive", t_dblcheck_pos),
        ("double_check",          "negative", t_dblcheck_neg),
        ("x_ray",                 "positive", t_xray_pos),
        ("x_ray",                 "negative", t_xray_neg),
        ("hanging_piece",         "positive", t_hang_pos),
        ("hanging_piece",         "negative", t_hang_neg),
        ("leaves_hanging",        "positive", t_leaves_pos),
        ("leaves_hanging",        "negative", t_leaves_neg),
        ("leaves_hanging",        "threshold", t_leaves_threshold),
        ("attacks_loose_piece",   "positive", t_loose_pos),
        ("attacks_loose_piece",   "negative", t_loose_neg),
        ("trapped_piece",         "positive", t_trap_pos),
        ("trapped_piece",         "negative", t_trap_neg),
        ("back_rank_weakness",    "positive (mate)", t_backrank_mate),
        ("back_rank_weakness",    "positive (weak)", t_backrank_weak),
        ("back_rank_weakness",    "negative", t_backrank_neg),
        ("capturing_defender",    "positive", t_capdef_pos),
        ("capturing_defender",    "negative", t_capdef_neg),
        ("capturing_defender",    "negative", t_capdef_neg2),
        ("deflection",            "positive", t_defl_pos),
        ("deflection",            "negative", t_defl_neg),
        ("deflection",            "negative (no pv)", t_defl_nopv),
        ("interference",          "positive", t_interf_pos),
        ("interference",          "negative", t_interf_neg),
        ("interference",          "negative (no pv)", t_interf_nopv),
        ("overloaded_defender",   "positive", t_over_pos),
        ("overloaded_defender",   "negative", t_over_neg),
        ("overloaded_defender",   "negative (no pv)", t_over_nopv),
        ("zwischenzug",           "positive", t_zw_pos),
        ("zwischenzug",           "negative", t_zw_neg),
        ("zwischenzug",           "negative (no stack)", t_zw_nostack),
        ("promotion_threat",      "positive", t_promo_pos),
        ("promotion_threat",      "positive (7th)", t_promo_seventh),
        ("promotion_threat",      "negative", t_promo_neg),
        ("promotion_threat",      "negative", t_promo_neg2),
        ("smothered_mate_pattern", "positive", t_smother_pos),
        ("smothered_mate_pattern", "negative", t_smother_neg),
        ("smothered_mate_pattern", "negative", t_smother_neg2),
        ("SEE",                   "sanity",   t_see),
        ("legal_attackers",       "pin",      t_pinned_attacker_excluded),
        ("legal_attackers",       "check",    t_frozen_by_check_excluded),
        ("legal_attackers",       "checker capture", t_checker_capture_allowed),
        ("legal_attackers",       "block",    t_check_block_allowed),
        ("detect_all",            "ordering", t_detect_all),
        ("detect_all",            "quiet move", t_detect_all_quiet),
        ("detect_all",            "plain recapture", t_detect_all_recapture),
        ("detect_all",            "piece recapture", t_detect_all_recapture2),
        ("relative_pin",          "true positive",   t_scandinavian_pin_is_real),
    ]

    for nm, kind, fn in TESTS:
        case(nm, kind, fn)

    print("\n" + "=" * 74)
    print(f"{'DETECTOR':<24} {'CASE':<20} RESULT")
    print("-" * 74)
    for nm, kind, res, msg in _RESULTS:
        flag = "ok " if res == "PASS" else "XX "
        print(f"{flag}{nm:<21} {kind:<20} {res}" + (f"  -- {msg}" if msg else ""))
    passed = sum(1 for r in _RESULTS if r[2] == "PASS")
    print("-" * 74)
    print(f"{passed}/{len(_RESULTS)} cases passed, {_PREMISES[0]} premise checks run")
    print("=" * 74)
    sys.exit(0 if passed == len(_RESULTS) else 1)
