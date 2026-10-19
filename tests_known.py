"""
Adversarial test set: positions where the correct answer is known independently,
so the analyzer can be caught being wrong rather than merely being self-consistent.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import chess, chess.engine
import classify as CL, accuracy as AC, tactics as T

# (name, fen, move, expected_label, why)
CASES = [
    ("forced recapture is not Great",
     "rnbqkb1r/ppp2ppp/5n2/3p4/3P4/2N5/PPP2PPP/R1BQKBNR w KQkq - 0 5", "c3d5",
     {"not": ["great", "brilliant"]}, "recapturing when it is the only sane move"),

    ("hanging the queen is a Blunder",
     "rnb1kbnr/pppp1ppp/8/4p3/6q1/5P2/PPPPP1PP/RNBQKBNR b KQkq - 0 3", "g4g2",
     {"is": "blunder"}, "Qxg2 loses the queen to Bxg2"),

    ("mate in one available, quiet move played = Miss",
     "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1", "g1h1",
     {"is": "miss"}, "Ra8# was mate; Kh1 throws it"),

    ("only defensive move that holds = Great",
     "r4rk1/ppp2ppp/8/8/8/8/PPP2PPP/R4RK1 w - - 0 1", "f1e1",
     {"any": ["best", "excellent", "great", "good"]}, "quiet equal position"),

    ("real sacrifice that works = Brilliant",
     "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/2N2N2/PPPP1PPP/R1BQK2R w KQkq - 0 1", "c4f7",
     {"maybe": "brilliant"}, "Bxf7+ offers the bishop"),
]


def run():
    sf = chess.engine.SimpleEngine.popen_uci("stockfish")
    sf.configure({"Threads": 4, "Hash": 256})
    fails = []
    try:
        for name, fen, uci, exp, why in CASES:
            b = chess.Board(fen)
            mv = chess.Move.from_uci(uci)
            if mv not in b.legal_moves:
                fails.append((name, "ILLEGAL TEST MOVE", "")); continue
            infos = sf.analyse(b, chess.engine.Limit(time=0.6), multipv=5)
            if isinstance(infos, dict): infos = [infos]
            cpb = CL.MATE_CP if infos[0]["score"].pov(b.turn).is_mate() and infos[0]["score"].pov(b.turn).mate()>0 \
                  else (-CL.MATE_CP if infos[0]["score"].pov(b.turn).is_mate() else infos[0]["score"].pov(b.turn).score())
            b2 = b.copy(); b2.push(mv)
            if b2.is_game_over():
                o=b2.outcome(); cpa = 0 if o.winner is None else (CL.MATE_CP if o.winner==b.turn else -CL.MATE_CP)
            else:
                ai = sf.analyse(b2, chess.engine.Limit(time=0.6))
                s = ai["score"].pov(b2.turn)
                cpa = -(CL.MATE_CP if s.is_mate() and s.mate()>0 else (-CL.MATE_CP if s.is_mate() else s.score()))
            drop = max(0.0, AC.win_prob(cpb) - AC.win_prob(cpa))
            comp = CL.complexity(b, infos, None)
            lab = CL.classify(b, mv, drop, cpb, cpa, infos, in_book=False, comp=comp)
            ok = True
            if "is" in exp and lab != exp["is"]: ok = False
            if "not" in exp and lab in exp["not"]: ok = False
            if "any" in exp and lab not in exp["any"]: ok = False
            mark = "PASS" if ok else "FAIL"
            print(f"  [{mark}] {name}\n         got '{lab}'  drop={drop:.1f} cp {cpb}->{cpa} "
                  f"n_good={comp.get('n_good')} only={comp.get('only_move')} legal={b.legal_moves.count()}")
            if not ok: fails.append((name, lab, why))
    finally:
        sf.quit()
    print(f"\n{len(CASES)-len(fails)}/{len(CASES)} passed")
    for n,g,w in fails: print(f"   FAIL {n}: got {g} ({w})")
    return fails


if __name__ == "__main__":
    run()
