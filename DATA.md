# Data

The repository contains **no data**. The benchmark uses the public Stack Exchange data dumps:

| Site | Dump | Used for |
|---|---|---|
| serverfault.com | `serverfault.com.7z` | main benchmark (320,296 tickets in the search base) |
| dba.stackexchange.com | `dba.stackexchange.com.7z` | replication (100,949 tickets) |

Source: the Stack Exchange data dump hosted on the Internet Archive (`https://archive.org/details/stackexchange`).
Download with `scripts/download_data.sh`.

**Licence and attribution.** The posts are user contributions licensed under
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/), attributed to the Stack Exchange contributors.
Anything derived from the posts (parquet files, vectors, rankings, example tickets quoted in text) carries the same licence
and must keep the attribution. The code in this repository is MIT licensed (see `LICENSE`); the data is not covered by it.

**Gold set.** "Duplicate" links created by Stack Exchange moderators (`PostLinks.xml`, link type 3) are the answer key: a new
question closed as a duplicate of an older one means the older one is the right search result.
