"""
Error taxonomy. An engine tells you *where* you went wrong; this tries to say
*why* -- which pattern you missed, and whether it was the kind of thing a player
at your level is expected to see.
"""

import chess

VALUE = {chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330,
         chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 20000}


# ------------------------------------------------------- static exchange eval

def see(board, move):
    """Static exchange evaluation: material swing if the square is fought over."""
    if board.is_en_passant(move):
        captured = 100
    else:
        vic = board.piece_at(move.to_square)
        captured = VALUE[vic.piece_type] if vic else 0
    if captured == 0 and not move.promotion:
        return 0

    sq = move.to_square
    b = board.copy(stack=False)
    gains = [captured]
    attacker = b.piece_at(move.from_square)
    on_square = VALUE[attacker.piece_type] if attacker else 0
    b.push(move)

    side = b.turn
    while True:
        atk = _least_valuable_attacker(b, sq, side)
        if atk is None:
            break
        gains.append(on_square - gains[-1])
        on_square = VALUE[b.piece_at(atk).piece_type]
        try:
            b.push(chess.Move(atk, sq))
        except Exception:
            break
        side = b.turn
        if len(gains) > 32:
            break

    # negamax back down the capture sequence; either side can stand pat
    for i in range(len(gains) - 2, -1, -1):
        gains[i] = -max(-gains[i], gains[i + 1])
    return gains[0]


def _least_valuable_attacker(board, square, color):
    best, best_val = None, 10 ** 9
    for sq in board.attackers(color, square):
        p = board.piece_at(sq)
        if p and VALUE[p.piece_type] < best_val:
            # make sure the capture is actually legal (pins)
            if chess.Move(sq, square) in board.legal_moves or \
               chess.Move(sq, square, promotion=chess.QUEEN) in board.legal_moves:
                best, best_val = sq, VALUE[p.piece_type]
    return best


# --------------------------------------------------------------- motif checks

def hanging_pieces(board, color):
    """Pieces of `color` the opponent can win material on right now.

    Works for either side. It used to silently return [] whenever
    `color == board.turn`, because `_least_valuable_attacker` and the loop
    below filter candidate captures through `board.legal_moves`, which holds
    only the side-to-move's moves. The caller could not distinguish "nothing
    is hanging" from "you asked about the wrong side" -- so `harvest.mine`'s
    quiet gate was structurally incapable of noticing that the MOVER's own
    material was en prise, and `ingest_hanging`'s promise of material to be
    "saved" was unsatisfiable. Flip the turn on a copy first.
    """
    out = []
    opp = not color
    if board.turn == color:
        if board.is_check():
            return []          # the check has to be answered first: different question
        board = board.copy(stack=False)
        board.turn = opp
        board.ep_square = None
    for sq in chess.SQUARES:
        p = board.piece_at(sq)
        if not p or p.color != color or p.piece_type == chess.KING:
            continue
        if not board.attackers(opp, sq):
            continue
        best = 0
        for a in board.attackers(opp, sq):
            mv = chess.Move(a, sq)
            if mv not in board.legal_moves:
                mv = chess.Move(a, sq, promotion=chess.QUEEN)
                if mv not in board.legal_moves:
                    continue
            best = max(best, see(board, mv))
        if best >= 150:
            out.append((sq, p.piece_type, best))
    return sorted(out, key=lambda x: -x[2])


def is_fork(board_before_reply, reply):
    """After `reply`, does the moved piece hit two or more worthwhile targets?"""
    b = board_before_reply.copy(stack=False)
    mover = b.piece_at(reply.from_square)
    if mover is None:
        return None
    b.push(reply)
    victim_color = b.turn                      # side now to move is the victim
    piece = b.piece_at(reply.to_square)
    if piece is None:
        return None
    targets = []
    for sq in b.attacks(reply.to_square):
        t = b.piece_at(sq)
        if not t or t.color != victim_color:
            continue
        if t.piece_type == chess.KING:
            targets.append(("king", sq))
        elif VALUE[t.piece_type] > VALUE[piece.piece_type] or not b.attackers(victim_color, sq):
            targets.append((chess.piece_name(t.piece_type), sq))
    return targets if len(targets) >= 2 else None


def is_discovered(board_before_reply, reply):
    """Did moving that piece open a line for something behind it?"""
    b = board_before_reply.copy(stack=False)
    before = {sq: set(b.attackers(not b.turn, sq)) for sq in chess.SQUARES if b.piece_at(sq)}
    mover_from = reply.from_square
    b.push(reply)
    victim_color = b.turn
    revealed = []
    for sq in chess.SQUARES:
        p = b.piece_at(sq)
        if not p or p.color != victim_color:
            continue
        now = set(b.attackers(not victim_color, sq))
        new = now - before.get(sq, set()) - {reply.to_square}
        # attackers that only exist because the mover vacated its square
        for a in new:
            if a != reply.to_square and a != mover_from:
                ap = b.piece_at(a)
                if ap and ap.piece_type in (chess.BISHOP, chess.ROOK, chess.QUEEN):
                    if VALUE[p.piece_type] >= 320 or p.piece_type == chess.KING:
                        revealed.append((chess.piece_name(p.piece_type), sq))
                        break
    return revealed or None


def is_back_rank(board_after_reply, color):
    """Is `color`'s king stuck on its back rank behind its own pawns?"""
    ksq = board_after_reply.king(color)
    if ksq is None:
        return False
    home = 0 if color == chess.WHITE else 7
    if chess.square_rank(ksq) != home:
        return False
    escapes = 0
    for sq in board_after_reply.attacks(ksq):
        p = board_after_reply.piece_at(sq)
        if p and p.color == color:
            continue
        if not board_after_reply.attackers(not color, sq):
            escapes += 1
    return escapes == 0


def pins_against(board, color):
    """Pieces of `color` that are pinned (absolutely, against the king)."""
    out = []
    for sq in chess.SQUARES:
        p = board.piece_at(sq)
        if p and p.color == color and board.is_pinned(color, sq):
            out.append((chess.piece_name(p.piece_type), sq))
    return out


def king_pressure(board, color):
    """Crude count of enemy attacks landing in the ring around our king."""
    ksq = board.king(color)
    if ksq is None:
        return 0
    zone = set(board.attacks(ksq)) | {ksq}
    return sum(len(board.attackers(not color, sq)) for sq in zone)


# ------------------------------------------------------------ classification

def classify_error(fen_before, played_uci, best_uci, refutation_pv,
                   cp_before, cp_after, is_mate_after=False):
    """
    Returns (tag, human_sentence). Order matters: the most specific, most
    teachable diagnosis wins.
    """
    board = chess.Board(fen_before)
    us = board.turn
    played = chess.Move.from_uci(played_uci)
    after = board.copy(stack=False)
    after.push(played)

    reply = None
    if refutation_pv:
        try:
            reply = refutation_pv[0]
            if reply not in after.legal_moves:
                reply = None
        except Exception:
            reply = None

    # ---- did we throw away a win that was already there?
    if cp_before >= 300 and cp_after < cp_before - 200:
        if cp_after < 50:
            return ("missed_win",
                    "You had a winning position and let it slip. The advantage was "
                    "already on the board -- this is a conversion problem, not a "
                    "problem of finding ideas.")
        return ("let_slip",
                "A large advantage shrank here. Converting is its own skill: "
                "trade pieces, not pawns, and keep the opponent passive.")

    if is_mate_after:
        return ("allowed_mate", "This allowed a forced mate.")

    # ---- concrete material loss
    hung = hanging_pieces(after, us)
    if hung:
        sq, pt, gain = hung[0]
        return ("hung_piece",
                f"This leaves the {chess.piece_name(pt)} on {chess.square_name(sq)} "
                f"available -- {gain/100:.1f} pawns for the taking. Before committing, "
                f"scan every one of your pieces the opponent can capture.")

    if reply is not None:
        fork = is_fork(after, reply)
        if fork:
            names = " and ".join(sorted({t[0] for t in fork}))
            return ("fork",
                    f"The reply forks your {names}. Two targets on one piece -- "
                    f"look for the squares that touch several of your men at once.")

        disc = is_discovered(after, reply)
        if disc:
            return ("discovered",
                    "The reply is a discovered attack: moving one piece uncovers a "
                    "second one behind it. Watch the lines your opponent's pieces sit on, "
                    "not just where they are.")

        line = after.copy(stack=False)
        line.push(reply)
        if line.is_check() and is_back_rank(line, us):
            return ("back_rank",
                    "Your king has no escape square on the back rank. Giving the king "
                    "air is cheap insurance; being mated there is not.")

    pinned = pins_against(after, us)
    if pinned and cp_before - cp_after >= 150:
        n, sq = pinned[0]
        return ("pin",
                f"Your {n} on {chess.square_name(sq)} is pinned and cannot move. "
                f"Pinned pieces defend nothing -- count them out when you check "
                f"whether a square is protected.")

    kp_before, kp_after = king_pressure(board, us), king_pressure(after, us)
    if kp_after - kp_before >= 3 and cp_before - cp_after >= 100:
        return ("king_safety",
                "This lets the attack on your king gather force. When pieces start "
                "pointing at your king, defensive moves are worth more than they look.")

    if board.is_capture(played) and cp_before - cp_after >= 100:
        return ("bad_trade",
                "The exchange favoured your opponent. Before every capture, ask what "
                "the position looks like *after* the recaptures, not just the count.")

    return ("positional",
            "No single tactic here -- the move quietly worsened the position. "
            "Compare it against the engine's choice and ask what that move does "
            "that yours does not.")


TAXONOMY_LABEL = {
    "missed_win": "Missed win", "let_slip": "Advantage let slip",
    "allowed_mate": "Allowed mate", "hung_piece": "Hung material",
    "fork": "Missed fork", "discovered": "Discovered attack",
    "back_rank": "Back rank", "pin": "Pin", "king_safety": "King safety",
    "bad_trade": "Bad exchange", "positional": "Positional drift",
}
