"""F1: build the ticket base and the duplicate-question gold set from the Parquet tables made by f0_parse.py.

Idea: moderators close a new question as a duplicate of an older one (PostLinks type 3). That gives free,
human-made labels: the new question is the query, the older question is the right answer.

Input : posts.parquet, postlinks.parquet, comments.parquet in PARQUET_DIR.
Output (in OUT_DIR):
  tickets.parquet        one row per ticket = title + question + accepted answer + comments (closed duplicates are left out)
  gold_dup_dev.jsonl     gold queries used for tuning (about 70%)
  gold_dup_heldout.jsonl gold queries kept for one final scoring (about 30%)
  f1_stats.json          counts

Leakage: old closed duplicates carry a "Possible Duplicate: <original title>" banner, or a link to the original, inside
their own text. Used as a query, that hands the system the answer. So every query has two versions:
query_raw keeps that text, query_clean removes it. The report scores both.

usage: f1_gold.py PARQUET_DIR OUT_DIR
"""
import collections
import json
import os
import random
import re
import sys

import pyarrow as pa
import pyarrow.parquet as pq
from lxml import html as lh

SEED = 13
HELDOUT_SHARE = 0.3
DUPLICATE_LINK = 3

# Phrases that appear in duplicate banners and in comments announcing the duplicate.
BANNER = re.compile(r'possible duplicate|already has an answer|already has answers here|duplicate of|closed as (a )?duplicate', re.I)

pdir, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)


def read(name):
    return pq.read_table(os.path.join(pdir, name + '.parquet')).to_pylist()


# ----------------------------------------------------------------------------- load
posts = read('posts')
links = [l for l in read('postlinks') if l['type'] == DUPLICATE_LINK]
comments = read('comments')
post_by_id = {p['id']: p for p in posts}


# ----------------------------------------------------------------------------- duplicate pairs
def find_duplicate_pairs():
    """Map each duplicate question id to the set of original question ids (both must be questions that exist)."""
    pairs = collections.defaultdict(set)
    for link in links:
        dup, original = post_by_id.get(link['post']), post_by_id.get(link['related'])
        both_exist = dup and original
        if both_exist and dup['type'] == 1 and original['type'] == 1 and dup['id'] != original['id']:
            pairs[dup['id']].add(original['id'])
    return pairs


duplicates = find_duplicate_pairs()
closed_duplicate_ids = set(duplicates)


# ----------------------------------------------------------------------------- cleaning the query text
def clean_text(body_html, original_ids):
    """Plain text of a duplicate question with the banner and every pointer to its original(s) removed."""
    if not body_html:
        return ''

    root = lh.fromstring('<div>' + body_html + '</div>')

    # banner blocks: short paragraphs / quotes that mention the duplicate (the length cap avoids deleting real content)
    for element in list(root.iter('blockquote', 'p', 'div')):
        text = element.text_content() or ''
        if element.getparent() is not None and BANNER.search(text) and len(text) < 400:
            element.getparent().remove(element)

    # links that point at an original
    for anchor in list(root.iter('a')):
        href = anchor.get('href') or ''
        points_at_original = any(re.search(r'/(questions|q|a)/%d\b' % o, href) for o in original_ids)
        if points_at_original and anchor.getparent() is not None:
            anchor.getparent().remove(anchor)

    # keep code readable as code (same convention as f0_parse.py)
    for block in root.iter('pre'):
        code = block.text_content()
        block.clear()
        block.text = '\n```\n' + code.strip('\n') + '\n```\n'
    for inline in root.iter('code'):
        code = inline.text_content()
        inline.clear()
        inline.text = '`' + code + '`'

    text = root.text_content()
    for o in original_ids:  # URLs typed as plain text, not as links
        text = re.sub(r'\S*/(questions|q|a)/%d\S*' % o, '', text)
    return re.sub(r'\n{3,}', '\n\n', text).strip()


# ----------------------------------------------------------------------------- the ticket base
def comments_by_post():
    """Comments per post, skipping the ones that announce a duplicate (they would leak the label)."""
    by_post = collections.defaultdict(list)
    for c in comments:
        if c['text'] and not BANNER.search(c['text']):
            by_post[c['post']].append(c['text'])
    return by_post


def tag_list(raw):
    """Tags as a list. Newer dumps write |a|b|, older ones <a><b>."""
    return [t for t in (raw or '').replace('><', '|').strip('<>').split('|') if t]


def build_tickets():
    comment_text = comments_by_post()
    tickets = []

    for p in posts:
        is_question = p['type'] == 1
        if not is_question or p['id'] in closed_duplicate_ids:
            continue  # closed duplicates stay out, so siblings of the same original cannot compete with it

        answer = post_by_id.get(p['accepted']) if p['accepted'] else None
        comment_block = '\n'.join(comment_text.get(p['id'], []))

        text = f"{p['title']}\n{p['text']}"
        if answer:
            text += f"\nAccepted answer:\n{answer['text']}"
        if comment_block:
            text += f"\nComments:\n{comment_block}"

        tickets.append(dict(
            id=p['id'], title=p['title'], tags=' '.join(tag_list(p['tags'])), created=p['created'], has_accepted=bool(answer),
            question=p['text'], answer=answer['text'] if answer else '', comments=comment_block, ticket_text=text,
        ))
    return tickets


tickets = build_tickets()
pq.write_table(pa.Table.from_pylist(tickets), os.path.join(out, 'tickets.parquet'), compression='zstd')
ticket_ids = {t['id'] for t in tickets}


# ----------------------------------------------------------------------------- the gold queries
def build_queries():
    queries = []
    for dup_id, originals in duplicates.items():
        originals = sorted(o for o in originals if o in ticket_ids)  # an original must be searchable
        if not originals:
            continue

        p = post_by_id[dup_id]
        raw = f"{p['title']}\n{p['text']}"
        clean = f"{p['title']}\n{clean_text(p['body_html'], originals)}"

        queries.append(dict(
            qid=dup_id, originals=originals, query_raw=raw, query_clean=clean, leaked=raw != clean,
            orig_has_accepted=all(post_by_id[o]['accepted'] for o in originals), created=p['created'],
        ))
    return queries


queries = build_queries()


# ----------------------------------------------------------------------------- split by original
# Queries that share an original must land in the same split, otherwise the held-out set would
# contain questions whose answer the tuning set already saw. Union-find groups them.
parent = {}


def find(x):
    parent.setdefault(x, x)
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


for q in queries:
    first = q['originals'][0]
    for other in q['originals'][1:]:
        parent[find(other)] = find(first)
    find(first)

groups = sorted({find(q['originals'][0]) for q in queries})
random.Random(SEED).shuffle(groups)
heldout_groups = set(groups[:round(len(groups) * HELDOUT_SHARE)])

for q in queries:
    q['split'] = 'heldout' if find(q['originals'][0]) in heldout_groups else 'dev'


# ----------------------------------------------------------------------------- write
for split in ('dev', 'heldout'):
    with open(os.path.join(out, f'gold_dup_{split}.jsonl'), 'w') as fh:
        for q in queries:
            if q['split'] == split:
                fh.write(json.dumps(q, ensure_ascii=False) + '\n')

per_split = collections.Counter(q['split'] for q in queries)
stats = dict(
    tickets=len(tickets), tickets_with_accepted=sum(t['has_accepted'] for t in tickets), gold_queries=len(queries),
    dev=per_split['dev'], heldout=per_split['heldout'], leaked_queries=sum(q['leaked'] for q in queries),
    orig_with_accepted=sum(q['orig_has_accepted'] for q in queries), seed=SEED,
)
json.dump(stats, open(os.path.join(out, 'f1_stats.json'), 'w'), indent=1)
print(stats)
