# Third-party material

- **python-chess** (GPL-3.0-or-later), by Niklas Fiekas. Every tool here is built on it, which is why this toolkit is licensed under the GPL-3.0 as well.
- **data/eco_all.tsv**: the lichess chess-openings database (CC0-1.0), github.com/lichess-org/chess-openings, used by book.py.
- **data/pawn_structures.json** and the classifier in pawn_structures.py: ported from lichessTools by Siderite Zackwehdex (MIT), github.com/Siderite/lichessTools.
- **phases.py**: a port of the game-phase divider from lichess-org/scalachess (MIT).
- **accuracy.py**: an implementation of the accuracy formula Lichess published (a formula, not their code).
- **Maia** networks (GPL-3.0), github.com/CSSLab/maia-chess, and **lc0** (GPL-3.0), lczero.org: not included; you download them yourself (see the README).
- **Stockfish** (GPL-3.0): not included.
- endgame.py calls the public lichess tablebase API (tablebase.lichess.ovh).
