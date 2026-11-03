"""Net votes per post AS OF a date, rebuilt from Votes.xml.

Why this exists: the `score` column in posts.parquet is a 2024 total. It also counts the votes an
original question collects AFTER a duplicate is closed onto it, which a live system could not know.
Here a vote only counts if it was cast strictly before the query's day.

Input : $DATA_DIR/xml/Votes.xml (VoteTypeId 2 = up, 3 = down; CreationDate has day resolution)
Output: {post_id: (sorted vote days, running net total)}, cached as a pickle in $DATA_DIR/cache.
"""
import bisect
import os
import pickle
from pathlib import Path

import numpy as np
from lxml import etree
DATA = Path(os.environ.get("DATA_DIR", ""))
if not DATA.is_dir():
    raise SystemExit("set DATA_DIR to the folder that holds gold/, parquet/, xml/ and the embedding folders (see README)")


VOTES_XML = DATA / "xml" / "Votes.xml"
CACHE = DATA / "cache" / "votes_asof.pkl"
UP, DOWN = "2", "3"


def build(tids):
    """Return {post_id: (days, running_net)} for the given ticket ids (loaded from the cache when present)."""
    if CACHE.exists():
        return pickle.load(open(CACHE, "rb"))

    wanted = set(int(i) for i in tids)
    events = {}

    # iterparse streams the 275 MB file row by row; clear() frees each row so memory stays flat
    for _, row in etree.iterparse(VOTES_XML, tag="row"):
        vote_type = row.get("VoteTypeId")
        if vote_type in (UP, DOWN):
            post = int(row.get("PostId"))
            if post in wanted:
                day = row.get("CreationDate")[:10]
                events.setdefault(post, []).append((day, 1 if vote_type == UP else -1))
        row.clear()

    table = {}
    for post, votes in events.items():
        votes.sort()
        days = [day for day, _ in votes]
        running_net = np.cumsum([sign for _, sign in votes]).tolist()
        table[post] = (days, running_net)

    CACHE.parent.mkdir(exist_ok=True)
    pickle.dump(table, open(CACHE, "wb"))
    return table


def asof(table, post, day):
    """Net votes (never below 0) cast strictly before `day` (YYYY-MM-DD)."""
    entry = table.get(int(post))
    if not entry:
        return 0

    days, running_net = entry
    k = bisect.bisect_left(days, day)        # number of votes dated before `day`
    return max(running_net[k - 1], 0) if k else 0
