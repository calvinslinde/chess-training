"""
Lichess's published accuracy method.

Per-move accuracies are not averaged arithmetically; that runs about 11 points
generous, because one catastrophic move barely dents an arithmetic mean.
Lichess combines a volatility-weighted mean with a harmonic mean, and the
harmonic term is what makes a blunder hurt.
"""

import math, statistics
from paths import P  # data folder; see paths.py

MATE_CP = 10000


def win_prob(cp):
    if cp >= MATE_CP:
        return 100.0
    if cp <= -MATE_CP:
        return 0.0
    return 50 + 50 * (2 / (1 + math.exp(-0.00368208 * cp)) - 1)


def move_accuracy(wp_before, wp_after, k=0.04354415386753951, bonus=1.0):
    """
    Per-move accuracy, matching lila's AccuracyPercent.scala exactly.

    Two details I originally missed, taken from the Lichess source: a move that
    improves the position scores exactly 100, and there is a "+1" uncertainty
    bonus on the curve. The constants are their curve_fit values in full.
    """
    if wp_after >= wp_before:
        return 100.0
    drop = wp_before - wp_after
    return max(0.0, min(100.0,
               103.1668100711649 * math.exp(-k * drop)
               - 3.166924740191411 + bonus))


def harmonic_mean(xs):
    xs = [max(x, 0.5) for x in xs]          # guard against a zero annihilating it
    return _harmonic(xs)


def _harmonic(xs):
    xs = [max(x, 0.5) for x in xs]
    return len(xs) / sum(1.0 / x for x in xs)


_FITTED = None


def _fitted_params():
    """
    The constants fitted against chess.com on 392 held-out player-games
    (TEST rmse 8.72, bias +0.55). They lived in data/acc_params.json and were
    referenced by the fitting script and nothing else -- every accuracy number
    ever reported came from the uncalibrated path at rmse 13.7.
    """
    global _FITTED
    if _FITTED is None:
        import json, os
        p = P("data/acc_params.json")
        try:
            _FITTED = tuple(json.load(open(p))["params"])
        except Exception:
            _FITTED = ()
    return _FITTED


def game_accuracy(win_percents, mover_is_white, calibrated=True):
    """
    win_percents: white-perspective Win% after every ply, with the starting
    position prepended. Returns accuracy for one side.

    Lichess's five steps: sliding windows sized by game length, volatility from
    the standard deviation of Win% in each window, a volatility-weighted mean of
    move accuracies, a harmonic mean of the same, then the average of the two.
    """
    n = len(win_percents)
    if n < 2:
        return 0.0

    p = _fitted_params() if calibrated else ()
    k, blend, wmin, wmax, wdiv = p if len(p) == 5 else (0.04354415386753951, 0.5, 0.5, 12.0, 10.0)

    win_size = max(2, min(8, int((n - 1) / wdiv)))
    windows = []
    padded = ([win_percents[0]] * (win_size - 1)) + win_percents
    for i in range(n - 1):
        windows.append(padded[i:i + win_size])

    accs, weights = [], []
    for i in range(n - 1):
        before_w, after_w = win_percents[i], win_percents[i + 1]
        is_white_move = (i % 2 == 0)
        if is_white_move != mover_is_white:
            continue
        # flip to the mover's own point of view
        wb = before_w if mover_is_white else 100 - before_w
        wa = after_w if mover_is_white else 100 - after_w
        # The fitted constants were fitted against the curve WITHOUT lichess's
        # "+1 uncertainty bonus", so applying them with it double-counts and
        # shifts every game by ~2.7 points. The integration test caught this on
        # its first run, which is the entire argument for having one.
        accs.append(move_accuracy(wb, wa, k, bonus=0.0 if p else 1.0))
        sd = statistics.pstdev(windows[i]) if len(windows[i]) > 1 else 0.0
        weights.append(max(wmin, min(wmax, sd)))

    if not accs:
        return 0.0
    wmean = sum(a * w for a, w in zip(accs, weights)) / sum(weights)
    hmean = _harmonic(accs)
    return max(0.0, min(100.0, blend * wmean + (1 - blend) * hmean))
