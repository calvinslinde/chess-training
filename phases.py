"""
Game phase detection, ported from Lichess's scalachess Divider (MIT).

My previous rule was "ply <= 20 is opening, <= 12 pieces is endgame", which is
crude enough to misplace most of the middlegame. Lichess uses three signals:
piece count, whether the back ranks have emptied (a development proxy), and
`mixedness` -- a 2x2 sliding window measuring how interlocked the two armies
are, which is the only real piece-tension primitive in their stack.
"""
import chess

# score for a 2x2 window given (white_count, black_count) and the window's rank
def _score(white, black, rank):
    if white == 0 and black == 0:
        return 0
    if white == 1 and black == 0:
        return 1 + (8 - rank)
    if white == 2 and black == 0:
        return 2 + (rank - 2) if rank > 2 else 0
    if white == 3 and black == 0:
        return 3 + (rank - 1) if rank > 1 else 0
    if white == 4 and black == 0:
        return 3 + (rank - 1) if rank > 1 else 0
    if white == 0 and black == 1:
        return 1 + rank
    if white == 1 and black == 1:
        return 5 + abs(3 - rank)
    if white == 2 and black == 1:
        return 4 + rank
    if white == 3 and black == 1:
        return 5 + rank
    if white == 0 and black == 2:
        return 2 + (6 - rank) if rank < 6 else 0
    if white == 1 and black == 2:
        return 4 + (6 - rank)
    if white == 2 and black == 2:
        return 7
    if white == 0 and black == 3:
        return 3 + (7 - rank) if rank < 7 else 0
    if white == 1 and black == 3:
        return 5 + (6 - rank)
    if white == 0 and black == 4:
        return 3 + (7 - rank) if rank < 7 else 0
    return 0


def majors_and_minors(board):
    return chess.popcount(board.occupied & ~(board.kings | board.pawns))


def backrank_sparse(board):
    first = chess.BB_RANK_1 & board.occupied_co[chess.WHITE]
    last = chess.BB_RANK_8 & board.occupied_co[chess.BLACK]
    return chess.popcount(first) < 4 or chess.popcount(last) < 4


def mixedness(board):
    """Sum over the 49 overlapping 2x2 windows of how interlocked the armies are."""
    total = 0
    for y in range(7):
        for x in range(7):
            w = b = 0
            for dy in (0, 1):
                for dx in (0, 1):
                    p = board.piece_at(chess.square(x + dx, y + dy))
                    if p:
                        if p.color == chess.WHITE:
                            w += 1
                        else:
                            b += 1
            total += _score(w, b, y + 1)
    return total


def phase_of(board):
    """'opening' | 'middlegame' | 'endgame' for a single position."""
    mm = majors_and_minors(board)
    if mm <= 6:
        return "endgame"
    if mm <= 10 or backrank_sparse(board) or mixedness(board) > 150:
        return "middlegame"
    return "opening"


def divide(game):
    """Ply indices where middlegame and endgame begin, Lichess's definition."""
    b = game.board()
    mid = end = None
    for ply, mv in enumerate(game.mainline_moves(), start=1):
        b.push(mv)
        mm = majors_and_minors(b)
        if mid is None and (mm <= 10 or backrank_sparse(b) or mixedness(b) > 150):
            mid = ply
        if end is None and mm <= 6:
            end = ply
    return {"midgame": mid, "endgame": end}
