"""Where this toolkit keeps its files.

Everything the tools read or write (downloaded games, reports, fitted
parameters, engine weights) lives under one folder. By default that is the
folder this file is in, so a fresh checkout works as is. Set CHESS_HOME to
put the data somewhere else.
"""
import os

HOME = os.path.abspath(os.environ.get("CHESS_HOME") or os.path.dirname(os.path.abspath(__file__)))


def P(rel):
    """Absolute path of a file under HOME. Creates the parent folder."""
    path = os.path.join(HOME, rel)
    os.makedirs(os.path.dirname(path) or HOME, exist_ok=True)
    return path
