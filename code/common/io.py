"""Result-table writing and resumption.

Every experiment writes one row per (task, support size, repetition, arm) and
can be stopped and restarted at any point. Two helpers carry that contract:

  already_done  the keys a part file already holds, so a rerun skips them
  append_rows   append, retrying while the file is briefly locked

The retry exists because these tables are written from a Windows session where
a reader holding the file open makes the append fail; the loop is harmless
elsewhere.
"""
import time

import pandas as pd


def already_done(path, keys):
    """Set of key tuples already present in `path`, empty if it does not exist."""
    from pathlib import Path
    if not Path(path).exists():
        return set()
    prev = pd.read_csv(path)
    return set(zip(*(prev[k] for k in keys)))


def append_rows(path, rows, retries=5):
    """Append `rows` to a CSV, writing the header only if the file is new."""
    from pathlib import Path
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    for attempt in range(retries):
        try:
            df.to_csv(path, mode="a", header=not path.exists(), index=False)
            return
        except PermissionError:
            time.sleep(2 * (attempt + 1))
    raise PermissionError(f"still locked after {retries} retries: {path}")


def write_table(path, rows):
    """Write a whole table at once, replacing any earlier copy."""
    from pathlib import Path
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
