"""Pull games from chess.com or lichess. Both public APIs, no auth needed."""

import argparse, datetime, os, sys, re
from paths import P  # data folder; see paths.py
import requests

UA = {"User-Agent": "chess-training-toolkit/1.0 (+https://github.com/calvinslinde/chess-training)"}
GAMES_DIR = P("games")


def chesscom(username, count=20, time_class=None, rated_only=True,
             color=None, result=None, months_back=6):
    u = username.strip().lower()
    r = requests.get(f"https://api.chess.com/pub/player/{u}/games/archives",
                     headers=UA, timeout=30)
    if r.status_code == 404:
        sys.exit(f"chess.com has no public player '{username}'.")
    r.raise_for_status()
    archives = r.json().get("archives", [])
    if not archives:
        sys.exit(f"No games found for '{username}' on chess.com.")

    picked = []
    for url in reversed(archives[-months_back:]):
        games = requests.get(url, headers=UA, timeout=60).json().get("games", [])
        for g in reversed(games):
            if rated_only and not g.get("rated"):
                continue
            if g.get("rules") != "chess":
                continue
            if time_class and g.get("time_class") != time_class:
                continue
            side = chess_side(g, u)
            if side is None:
                continue
            if color and side != color:
                continue
            if result and g[side].get("result") != result:
                continue
            picked.append(normalize_cc(g, u, side))
            if len(picked) >= count:
                return picked
    return picked


def chess_side(g, username):
    if g.get("white", {}).get("username", "").lower() == username:
        return "white"
    if g.get("black", {}).get("username", "").lower() == username:
        return "black"
    return None


def normalize_cc(g, username, side):
    return {
        "pgn": g["pgn"], "url": g.get("url", ""), "site": "chess.com",
        "time_class": g.get("time_class"), "time_control": g.get("time_control"),
        "rated": g.get("rated"), "eco": g.get("eco", ""),
        "user_side": side, "user_result": g[side].get("result"),
        "user_rating": g[side].get("rating"),
        "opp_rating": g["black" if side == "white" else "white"].get("rating"),
        "opp": g["black" if side == "white" else "white"].get("username"),
        "end_time": g.get("end_time"),
        "cc_accuracy": (g.get("accuracies") or {}).get(side),
    }


def lichess(username, count=20, time_class=None, rated_only=True, color=None):
    params = {"max": count * 3, "pgnInJson": "false", "clocks": "true",
              "evals": "false", "opening": "true"}
    if rated_only:
        params["rated"] = "true"
    if time_class:
        params["perfType"] = time_class
    if color:
        params["color"] = color
    r = requests.get(f"https://lichess.org/api/games/user/{username}",
                     params=params, headers={**UA, "Accept": "application/x-chess-pgn"},
                     timeout=90)
    if r.status_code == 404:
        sys.exit(f"lichess has no player '{username}'.")
    r.raise_for_status()
    blocks = [b.strip() for b in re.split(r"\n\n(?=\[Event)", r.text) if b.strip()]
    out = []
    for b in blocks[:count]:
        side = "white" if re.search(rf'\[White "{re.escape(username)}"\]', b, re.I) else "black"
        out.append({"pgn": b, "site": "lichess", "user_side": side,
                    "url": "", "time_class": time_class, "rated": rated_only,
                    "eco": "", "user_result": None, "user_rating": None,
                    "opp_rating": None, "opp": None, "end_time": None,
                    "cc_accuracy": None})
    return out


def write_pgn(games, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for g in games:
            fh.write(g["pgn"].strip() + "\n\n")
    return path


def main():
    ap = argparse.ArgumentParser(description="Fetch games from chess.com or lichess.")
    ap.add_argument("username")
    ap.add_argument("--site", choices=["chess.com", "lichess"], default="chess.com")
    ap.add_argument("-n", "--count", type=int, default=20)
    ap.add_argument("--time-class", help="bullet | blitz | rapid | daily")
    ap.add_argument("--color", choices=["white", "black"])
    ap.add_argument("--result", help="chess.com only: win | checkmated | resigned | timeout | ...")
    ap.add_argument("--include-unrated", action="store_true")
    ap.add_argument("--months-back", type=int, default=6)
    ap.add_argument("-o", "--out")
    args = ap.parse_args()

    fn = chesscom if args.site == "chess.com" else lichess
    kw = dict(count=args.count, time_class=args.time_class,
              rated_only=not args.include_unrated, color=args.color)
    if args.site == "chess.com":
        kw.update(result=args.result, months_back=args.months_back)
    games = fn(args.username, **kw)

    if not games:
        sys.exit("No games matched those filters.")
    stamp = datetime.datetime.now().strftime("%Y%m%d")
    out = args.out or os.path.join(
        GAMES_DIR, f"{args.username}-{args.time_class or 'all'}-{stamp}.pgn")
    write_pgn(games, out)
    print(f"{len(games)} games -> {out}")
    ratings = [g["user_rating"] for g in games if g.get("user_rating")]
    if ratings:
        print(f"rating range {min(ratings)}-{max(ratings)}, latest {ratings[0]}")


if __name__ == "__main__":
    main()
