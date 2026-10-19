"""
Positional decomposition.

"Positional drift" was 24% of the last report and it means nothing -- it was the
bucket for errors the tactic detector could not name. This computes real
positional features for the position your move reached and the one the engine's
move reached, then reports which named features actually differ.
"""

import chess

VAL = {chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330,
       chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 0}
CENTER = [chess.D4, chess.E4, chess.D5, chess.E5]
BIG_CENTER = CENTER + [chess.C4, chess.C5, chess.F4, chess.F5,
                       chess.D3, chess.E3, chess.D6, chess.E6]


def _pawns(board, color):
    return [chess.square_file(s) for s in board.pieces(chess.PAWN, color)]


def pawn_structure(board, color):
    files = _pawns(board, color)
    fset = set(files)
    doubled = sum(files.count(f) - 1 for f in fset)
    isolated = sum(files.count(f) for f in fset
                   if (f - 1) not in fset and (f + 1) not in fset)
    passed, adv = 0, 0
    for sq in board.pieces(chess.PAWN, color):
        f, r = chess.square_file(sq), chess.square_rank(sq)
        blocked = False
        for df in (-1, 0, 1):
            nf = f + df
            if not 0 <= nf <= 7:
                continue
            for es in board.pieces(chess.PAWN, not color):
                if chess.square_file(es) != nf:
                    continue
                er = chess.square_rank(es)
                if (color == chess.WHITE and er > r) or (color == chess.BLACK and er < r):
                    blocked = True
                    break
            if blocked:
                break
        if not blocked:
            passed += 1
            adv += (r if color == chess.WHITE else 7 - r)
    islands = 0
    prev = -9
    for f in sorted(fset):
        if f != prev + 1:
            islands += 1
        prev = f
    return {"doubled": doubled, "isolated": isolated, "passed": passed,
            "passed_adv": adv, "islands": islands}


def king_safety(board, color):
    ksq = board.king(color)
    if ksq is None:
        return {"attackers": 0, "escapes": 0, "shield": 0}
    zone = set(board.attacks(ksq)) | {ksq}
    attackers = sum(1 for s in zone if board.attackers(not color, s))
    escapes = 0
    for s in board.attacks(ksq):
        p = board.piece_at(s)
        if p and p.color == color:
            continue
        if not board.attackers(not color, s):
            escapes += 1
    shield = 0
    kf, kr = chess.square_file(ksq), chess.square_rank(ksq)
    step = 1 if color == chess.WHITE else -1
    for df in (-1, 0, 1):
        f = kf + df
        if not 0 <= f <= 7:
            continue
        for dr in (1, 2):
            r = kr + step * dr
            if 0 <= r <= 7:
                p = board.piece_at(chess.square(f, r))
                if p and p.piece_type == chess.PAWN and p.color == color:
                    shield += 1
                    break
    return {"attackers": attackers, "escapes": escapes, "shield": shield}


def safe_mobility(board, color):
    """Legal moves that don't just drop the piece on arrival."""
    total = 0
    mask = (board.pieces_mask(chess.KNIGHT, color) | board.pieces_mask(chess.BISHOP, color)
            | board.pieces_mask(chess.ROOK, color) | board.pieces_mask(chess.QUEEN, color))
    for sq in chess.SquareSet(mask):
        p = board.piece_at(sq)
        if not p:
            continue
        for t in board.attacks(sq):
            occ = board.piece_at(t)
            if occ and occ.color == color:
                continue
            atk = board.attackers(not color, t)
            if atk:
                cheapest = min((VAL[board.piece_at(a).piece_type] for a in atk), default=10 ** 9)
                if cheapest < VAL[p.piece_type] and not board.attackers(color, t):
                    continue
            total += 1
    return total


def rooks(board, color):
    own = set(_pawns(board, color))
    opp = set(_pawns(board, not color))
    seventh = 6 if color == chess.WHITE else 1
    open_f = semi = on7 = 0
    for sq in board.pieces(chess.ROOK, color):
        f = chess.square_file(sq)
        if f not in own and f not in opp:
            open_f += 1
        elif f not in own:
            semi += 1
        if chess.square_rank(sq) == seventh:
            on7 += 1
    return {"rook_open": open_f, "rook_semi": semi, "rook_7th": on7}


def outposts(board, color):
    """Knights/bishops on squares no enemy pawn can ever attack, backed by a pawn."""
    n = 0
    for pt in (chess.KNIGHT, chess.BISHOP):
        for sq in board.pieces(pt, color):
            r = chess.square_rank(sq)
            depth = r if color == chess.WHITE else 7 - r
            if depth < 3:
                continue
            f = chess.square_file(sq)
            attackable = False
            for df in (-1, 1):
                nf = f + df
                if not 0 <= nf <= 7:
                    continue
                for es in board.pieces(chess.PAWN, not color):
                    if chess.square_file(es) == nf:
                        er = chess.square_rank(es)
                        if (color == chess.WHITE and er > r) or (color == chess.BLACK and er < r):
                            attackable = True
            if not attackable and (int(board.attackers(color, sq)) & board.pieces_mask(chess.PAWN, color)):
                n += 1
    return n


def loose_pieces(board, color):
    n = 0
    for sq in chess.SQUARES:
        p = board.piece_at(sq)
        if not p or p.color != color or p.piece_type in (chess.PAWN, chess.KING):
            continue
        if board.attackers(not color, sq) and not board.attackers(color, sq):
            n += 1
    return n


def space(board, color):
    half = range(4, 8) if color == chess.WHITE else range(0, 4)
    n = 0
    for f in range(8):
        for r in half:
            s = chess.square(f, r)
            if board.attackers(color, s):
                n += 1
    return n


def features(board, color):
    ps = pawn_structure(board, color)
    ks = king_safety(board, color)
    rk = rooks(board, color)
    f = {
        "material": sum(VAL[p.piece_type] * (1 if p.color == color else -1)
                        for p in board.piece_map().values()),
        "mobility": safe_mobility(board, color),
        "space": space(board, color),
        "outposts": outposts(board, color),
        "loose": loose_pieces(board, color),
        "bishop_pair": 1 if len(board.pieces(chess.BISHOP, color)) >= 2 else 0,
        "center": sum(1 for s in BIG_CENTER if board.attackers(color, s)),
    }
    f.update(ps)
    f.update({"k_attackers": ks["attackers"],
              "k_airless": 1 if (ks["escapes"] == 0 and ks["attackers"] > 0) else 0,
              "k_shield": ks["shield"]})
    f.update(rk)
    return f


# rough centipawn weight per unit, only used to rank which differences matter
WEIGHT = {
    # material is measured in centipawns while every other feature is a unit
    # COUNT, so at 1.0 it outranked everything: it led 8 of 17 "positional"
    # explanations in the shipped report, i.e. the module built to replace the
    # "positional drift" bucket spent half its time saying "gives up material",
    # which the evaluation already said. Scaled so a whole pawn ranks near a
    # bishop pair rather than 3x above it.
    "material": 0.30, "mobility": 1.2, "space": 0.9, "outposts": 26, "loose": -18,
    "bishop_pair": 32, "center": 4, "doubled": -11, "isolated": -14,
    "passed": 25, "passed_adv": 7, "islands": -8, "k_attackers": -14,
    "k_airless": -35, "k_shield": 11, "rook_open": 22, "rook_semi": 11, "rook_7th": 20,
}

# Phrased from the player's point of view: what your move gave up relative to
# the engine's. The old two-tuple form described the alternative and read as if
# it described your move.
LOST = {
    "material": "gives up material",
    "mobility": "leaves your pieces with fewer safe squares",
    "space": "concedes space",
    "outposts": "gives up an outpost square",
    "loose": "leaves a piece undefended",
    "bishop_pair": "gives up the bishop pair",
    "center": "concedes the centre",
    "doubled": "doubles your pawns",
    "isolated": "leaves you an isolated pawn",
    "passed": "gives up a passed pawn",
    "passed_adv": "lets your passed pawn stall",
    "islands": "splits your pawns",
    "k_attackers": "lets more enemy pieces near your king",
    "k_airless": "leaves your king with no escape square",
    "k_shield": "breaks up the pawns in front of your king",
    "rook_open": "gives up the open file",
    "rook_semi": "gives up the half-open file",
    "rook_7th": "gives up the seventh rank",
}
GAINED = {
    "material": "wins material", "mobility": "frees your pieces",
    "space": "takes space", "outposts": "plants a piece on an outpost",
    "loose": "leaves nothing loose", "bishop_pair": "keeps the bishop pair",
    "center": "takes the centre", "doubled": "avoids doubled pawns",
    "isolated": "avoids an isolated pawn", "passed": "creates a passed pawn",
    "passed_adv": "advances a passed pawn", "islands": "keeps your pawns connected",
    "k_attackers": "keeps attackers off your king", "k_airless": "gives your king air",
    "k_shield": "keeps your king's pawn cover", "rook_open": "takes the open file",
    "rook_semi": "takes the half-open file", "rook_7th": "reaches the seventh",
}

# mobility is the least interpretable term, so it needs a bigger delta to speak up
MIN_WEIGHT = {"mobility": 20, "space": 14, "center": 14}
DEFAULT_MIN = 11

def explain(board_played, board_best, color, top=3):
    """
    What your move cost relative to the engine's, in named terms.

    Returns {"lost": [...], "gained": [...]} -- `lost` is what the engine's move
    achieves that yours does not, phrased about your move; `gained` is where your
    move was actually the better one, which is worth saying out loud.
    """
    fp, fb = features(board_played, color), features(board_best, color)
    lost, gained = [], []
    for k in fb:
        d = fb[k] - fp[k]
        if d == 0:
            continue
        w = d * WEIGHT[k]
        if abs(w) < MIN_WEIGHT.get(k, DEFAULT_MIN):
            continue
        rec = {"feature": k, "delta": d, "weighted": abs(w)}
        if w > 0:
            lost.append({**rec, "text": LOST[k]})
        else:
            gained.append({**rec, "text": GAINED[k]})
    lost.sort(key=lambda x: -x["weighted"])
    gained.sort(key=lambda x: -x["weighted"])
    return {"lost": lost[:top], "gained": gained[:2]}


def quiescent(board, moves, max_plies=8):
    """
    Play out a principal variation so exchanges resolve before features are read.
    Comparing raw one-ply positions miscounts material mid-capture-sequence.
    """
    b = board.copy(stack=False)
    for mv in moves[:max_plies]:
        if mv not in b.legal_moves:
            break
        b.push(mv)
    return b


def explain_lines(board, played_line, best_line, color, top=3):
    """Compare the position each line actually reaches, not the position one ply in."""
    return explain(quiescent(board, played_line), quiescent(board, best_line), color, top)
