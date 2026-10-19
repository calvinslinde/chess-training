"""Compact, chat-friendly board SVG. Unicode glyphs, no external assets."""
import chess

GLYPH = {chess.PAWN: "♟", chess.KNIGHT: "♞", chess.BISHOP: "♝",
         chess.ROOK: "♜", chess.QUEEN: "♛", chess.KING: "♚"}
LIGHT, DARK = "#e9dcc3", "#ad8264"
HL = "#c9a227"


def board_svg(fen, flip=False, size=440, x0=120, y0=52, highlight=(), title="", desc=""):
    b = chess.Board(fen)
    s = size / 8
    out = [f'<svg width="100%" viewBox="0 0 680 {y0 + size + 34}" role="img">',
           f'<title>{title or "Chess position"}</title>',
           f'<desc>{desc or fen}</desc>']
    for rank in range(8):
        for file in range(8):
            f_, r_ = (7 - file, rank) if flip else (file, 7 - rank)
            sq = chess.square(f_, r_)
            x, y = x0 + file * s, y0 + rank * s
            fill = LIGHT if (f_ + r_) % 2 else DARK
            out.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{s:.1f}" height="{s:.1f}" fill="{fill}"/>')
            if sq in highlight:
                out.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{s:.1f}" height="{s:.1f}" '
                           f'fill="{HL}" opacity="0.42"/>')
            p = b.piece_at(sq)
            if p:
                col = "#ffffff" if p.color == chess.WHITE else "#17150f"
                edge = "#2b2620" if p.color == chess.WHITE else "#d8d2c4"
                out.append(
                    f'<text x="{x + s/2:.1f}" y="{y + s*0.76:.1f}" font-size="{s*0.82:.1f}" '
                    f'text-anchor="middle" fill="{col}" stroke="{edge}" stroke-width="1.1" '
                    f'paint-order="stroke" font-family="Segoe UI Symbol,Apple Symbols,'
                    f'DejaVu Sans,sans-serif">{GLYPH[p.piece_type]}</text>')
    files = "hgfedcba" if flip else "abcdefgh"
    ranks = "12345678" if flip else "87654321"
    for i, ch in enumerate(files):
        out.append(f'<text class="ts" x="{x0 + i*s + s/2:.1f}" y="{y0 + size + 22:.1f}" '
                   f'font-size="13" text-anchor="middle">{ch}</text>')
    for i, ch in enumerate(ranks):
        out.append(f'<text class="ts" x="{x0 - 14:.1f}" y="{y0 + i*s + s/2 + 5:.1f}" '
                   f'font-size="13" text-anchor="middle">{ch}</text>')
    turn = "White" if b.turn == chess.WHITE else "Black"
    out.append(f'<text class="th" x="340" y="30" font-size="15" text-anchor="middle">'
               f'{turn} to move</text>')
    out.append("</svg>")
    return "".join(out)
