"""F0: turn a Stack Exchange XML dump into Parquet tables.

Input : a folder with Posts.xml, PostLinks.xml, Comments.xml, Tags.xml
        (the public dump of dba.stackexchange.com or serverfault.com, CC BY-SA 4.0).
Output: posts.parquet, postlinks.parquet, comments.parquet, tags.parquet in OUT_DIR.
        posts.parquet keeps the original HTML body and a plain-text version of it.

usage: f0_parse.py XML_DIR OUT_DIR
"""
import os
import re
import sys

import pyarrow as pa
import pyarrow.parquet as pq
from lxml import etree, html as lh

xml_dir, out_dir = sys.argv[1], sys.argv[2]
os.makedirs(out_dir, exist_ok=True)


def text_of(body_html):
    """Plain text of a post body. Code stays visible as code (backticks / fenced blocks)."""
    if not body_html:
        return ''

    root = lh.fromstring('<div>' + body_html + '</div>')  # wrapper: the body may have several top-level tags

    for block in root.iter('pre'):
        code = block.text_content()
        block.clear()
        block.text = '\n```\n' + code.strip('\n') + '\n```\n'

    for inline in root.iter('code'):
        code = inline.text_content()
        inline.clear()
        inline.text = '`' + code + '`'

    text = root.text_content()
    return re.sub(r'\n{3,}', '\n\n', text).strip()


def rows(path, tag):
    """Yield one dict of attributes per XML element.

    iterparse + clear() keeps memory flat: Posts.xml is over 1 GB, and building the full tree would not fit.
    """
    for _, element in etree.iterparse(path, tag=tag):
        yield dict(element.attrib)
        element.clear()
        while element.getprevious() is not None:  # also drop already-processed siblings
            del element.getparent()[0]


def as_int(record, key):
    return int(record[key]) if key in record else None


def write(name, records, schema):
    table = pa.Table.from_pylist(records, schema=schema)
    # zstd: the text columns are large and repetitive; it compresses them well and reads fast
    pq.write_table(table, os.path.join(out_dir, name + '.parquet'), compression='zstd')
    print(name, len(records), flush=True)


# --- posts: questions (type 1), answers (type 2) and a few other kinds
posts = []
for d in rows(os.path.join(xml_dir, 'Posts.xml'), 'row'):
    posts.append(dict(
        id=as_int(d, 'Id'), type=as_int(d, 'PostTypeId'), parent=as_int(d, 'ParentId'), accepted=as_int(d, 'AcceptedAnswerId'),
        created=d.get('CreationDate'), score=as_int(d, 'Score'), views=as_int(d, 'ViewCount'), title=d.get('Title'),
        body_html=d.get('Body'), tags=d.get('Tags'), text=text_of(d.get('Body')),
    ))

posts_schema = pa.schema([
    ('id', pa.int64()), ('type', pa.int8()), ('parent', pa.int64()), ('accepted', pa.int64()), ('created', pa.string()),
    ('score', pa.int32()), ('views', pa.int32()), ('title', pa.string()), ('body_html', pa.string()), ('tags', pa.string()), ('text', pa.string()),
])
write('posts', posts, posts_schema)

# --- post links: type 1 = related, type 3 = duplicate (the labels used for the gold set)
links = [
    dict(id=as_int(d, 'Id'), created=d.get('CreationDate'), post=as_int(d, 'PostId'), related=as_int(d, 'RelatedPostId'), type=as_int(d, 'LinkTypeId'))
    for d in rows(os.path.join(xml_dir, 'PostLinks.xml'), 'row')
]
links_schema = pa.schema([('id', pa.int64()), ('created', pa.string()), ('post', pa.int64()), ('related', pa.int64()), ('type', pa.int8())])
write('postlinks', links, links_schema)

# --- comments
comments = [
    dict(id=as_int(d, 'Id'), post=as_int(d, 'PostId'), score=as_int(d, 'Score'), created=d.get('CreationDate'), text=d.get('Text'))
    for d in rows(os.path.join(xml_dir, 'Comments.xml'), 'row')
]
comments_schema = pa.schema([('id', pa.int64()), ('post', pa.int64()), ('score', pa.int32()), ('created', pa.string()), ('text', pa.string())])
write('comments', comments, comments_schema)

# --- tags
tags = [dict(id=as_int(d, 'Id'), name=d.get('TagName'), count=as_int(d, 'Count')) for d in rows(os.path.join(xml_dir, 'Tags.xml'), 'row')]
tags_schema = pa.schema([('id', pa.int64()), ('name', pa.string()), ('count', pa.int32())])
write('tags', tags, tags_schema)
