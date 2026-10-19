#!/usr/bin/env python3
"""
The drilling loop: spaced repetition over your own mistakes.

Everything else in this project produces a static report. Reports are not
teaching: you read them, agree, and forget. This turns your own errors into
positions you have to solve, and schedules them.

Two design decisions worth defending:

1. **Only your own mistakes, never generic puzzles.** The one intervention with
   published support here is Riedl & Bogert (n=52,000 over five years): feedback
   on FAILURES improves later performance, feedback on wins makes it worse. So
   the queue is built from losses and errors, and it will not let you review a
   game you won cleanly.

2. **Spaced repetition, with its limits stated.** There is no published evidence
   that spaced repetition of one's own chess mistakes raises rating.
   SM-2 is used here because forgetting curves are well established in general,
   not because anyone has shown it works for chess. That caveat is printed with
   the stats rather than buried.

Positions are ranked by what they are worth: win probability lost, weighted up
when a player at your own rating routinely finds the move you missed. Those are
the cheapest points on the board.
"""
import json, os, time
from paths import P  # data folder; see paths.py
import chess

STATE = P("reports/drill_state.json")


def _load_state():
    if os.path.exists(STATE):
        try:
            return json.load(open(STATE))
        except Exception:
            pass
    return {"cards": {}, "history": []}


def _save(state):
    os.makedirs(os.path.dirname(STATE), exist_ok=True)
    json.dump(state, open(STATE, "w"), indent=1)


def card_id(fen, uci):
    return f"{fen.split(' ')[0]}|{uci}"


def build(review_paths, min_drop=8.0, losses_only=False):
    """
    Turn analysed games into drillable cards, worst-value first.

    `losses_only` applies the Riedl & Bogert asymmetry (feedback on failures
    helps, feedback on wins hurts) by keeping only cards from games you lost.
    It is off by default because a player who wins most of their games would be
    left with almost no cards; the option exists and the caveat is explicit.
    """
    state = _load_state()
    added = 0
    for p in review_paths:
        if not os.path.exists(p):
            continue
        for g in json.load(open(p)):
            side = g["side"]
            res = g.get("result")
            won = (res != "1/2-1/2") and ((res == "1-0") == (side == "white"))
            for mv in g.get("moves", []):
                if not mv.get("mine") or mv.get("drop", 0) < min_drop:
                    continue
                if not mv.get("best"):
                    continue
                cid = card_id(mv["fen"], mv["uci"])
                if cid in state["cards"]:
                    continue
                if losses_only and won:
                    continue
                maia = mv.get("maia") or {}
                p_best = maia.get("p_best")
                # a move your own band finds is worth more to drill than one nobody finds
                value = mv["drop"] * (1.0 + 2.0 * (p_best or 0.0))
                state["cards"][cid] = {
                    "fen": mv["fen"], "played": mv["san"], "best": mv["best"],
                    "best_pv": mv.get("best_pv", ""), "refutation": mv.get("refutation", ""),
                    "drop": mv["drop"], "p_best": p_best, "label": mv.get("label"),
                    "side": side, "move_no": mv["move_no"],
                    "from_win": won, "value": value,
                    "reps": 0, "ease": 2.5, "interval": 0, "due": 0, "lapses": 0,
                }
                added += 1
    _save(state)
    return added, len(state["cards"])


def due_cards(now=None, include_new=True):
    state = _load_state()
    now = now or time.time()
    out = []
    for cid, c in state["cards"].items():
        if c["reps"] == 0 and include_new:
            out.append((cid, c))
        elif c["due"] <= now:
            out.append((cid, c))
    # hardest-hitting and most findable first
    out.sort(key=lambda kv: -kv[1]["value"])
    return out


def next_card():
    d = due_cards()
    return d[0] if d else (None, None)


def grade(cid, quality):
    """
    SM-2. quality 0-5: 5 found the move instantly, 3 found it with effort,
    <3 failed. Failing resets the interval, which is the point.
    """
    state = _load_state()
    c = state["cards"].get(cid)
    if not c:
        return None
    if quality < 3:
        c["reps"] = 0
        c["interval"] = 0
        c["lapses"] += 1
    else:
        c["reps"] += 1
        if c["reps"] == 1:
            c["interval"] = 1
        elif c["reps"] == 2:
            c["interval"] = 4
        else:
            c["interval"] = round(c["interval"] * c["ease"])
        c["ease"] = max(1.3, c["ease"] + 0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02))
    c["due"] = time.time() + c["interval"] * 86400
    state["history"].append({"cid": cid, "q": quality, "t": time.time()})
    _save(state)
    return c


def stats():
    state = _load_state()
    cards = state["cards"]
    if not cards:
        return {"cards": 0}
    seen = [c for c in cards.values() if c["reps"] > 0]
    return {
        "cards": len(cards), "seen": len(seen),
        "due_now": len(due_cards()),
        "lapses": sum(c["lapses"] for c in cards.values()),
        "reviews": len(state["history"]),
        "from_losses": sum(1 for c in cards.values() if not c["from_win"]),
        "caveat": ("spaced repetition of one's own chess mistakes has no published "
                   "evidence of raising rating; the scheduling is a reasonable prior, "
                   "not a demonstrated intervention"),
    }


def prompt_for(card):
    """What to say when showing the position -- never reveal the answer."""
    b = chess.Board(card["fen"])
    side = "White" if b.turn == chess.WHITE else "Black"
    stake = ("You are winning here." if card.get("drop", 0) and
             b.turn == (chess.WHITE if card["side"] == "white" else chess.BLACK) else "")
    return (f"{side} to move. This position cost you {card['drop']:.0f}% win probability "
            f"in a real game. What would you play?")


def reveal(card):
    lines = [f"You played **{card['played']}** — {card.get('label','')}, "
             f"{card['drop']:.0f}% win probability lost."]
    if card.get("best"):
        lines.append(f"Best was **{card['best']}**: {card.get('best_pv','')}")
    if card.get("refutation"):
        lines.append(f"Punished by: {card['refutation']}")
    if card.get("p_best") is not None:
        lines.append(f"{card['p_best']*100:.0f}% of players at your level find it.")
    return "\n".join(lines)


if __name__ == "__main__":
    import sys, glob
    # Feed it the JSON that review.py writes (--out). With no arguments it
    # picks up every reports/review*.json.
    paths = sys.argv[1:] or sorted(glob.glob(P("reports/review*.json")))
    if not paths:
        sys.exit("usage: python3 drill.py REVIEW_JSON [REVIEW_JSON ...]  (the --out file of review.py)")
    added, total = build(paths)
    print(f"built {added} new cards ({total} total)")
    s = stats()
    print({k: v for k, v in s.items() if k != "caveat"})
    cid, c = next_card()
    if c:
        print("\nnext card:", c["move_no"], c["played"], "->", c["best"],
              f"(value {c['value']:.1f})")
