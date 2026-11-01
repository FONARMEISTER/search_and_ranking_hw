#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ДЗ 1. Индексация и булев поиск.

Запуск:
    ./solution.py --build_index --index_dir=index PATH-TO-DATA
    ./solution.py --submission_file=submission.csv --index_dir=index PATH-TO-DATA
"""

import argparse
from array import array
import csv
from functools import lru_cache
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
from timeit import default_timer as timer

from nltk import tokenize


def preprocess(text):
    """Токенизация и нормализация. НЕ МЕНЯЙТЕ эту функцию:
    ей же размечен эталон на Kaggle."""
    tokenizer = tokenize.RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text)
    return [token.lower() for token in tokens]


def build_index(data_dir, index_dir):
    """Store sorted postings as little-endian uint32 blobs, keyed by term."""
    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    fd, temporary_path = tempfile.mkstemp(dir=index_dir, suffix='.sqlite3')
    os.close(fd)
    connection = None
    try:
        connection = sqlite3.connect(temporary_path)
        connection.execute('CREATE TABLE documents (id TEXT PRIMARY KEY, ordinal INTEGER)')
        connection.execute(
            'CREATE TABLE postings (term TEXT PRIMARY KEY, frequency INTEGER, data BLOB)'
            ' WITHOUT ROWID'
        )
        postings = {}
        with (Path(data_dir) / 'vkmarco-docs.tsv').open(encoding='utf-8') as source:
            for ordinal, line in enumerate(source):
                document_id, _url, title, body = line.rstrip('\r\n').split('\t', 3)
                connection.execute('INSERT INTO documents VALUES (?, ?)', (document_id, ordinal))
                for term in set(preprocess(title + ' ' + body)):
                    if term not in postings:
                        postings[term] = array('I')
                    postings[term].append(ordinal)
        for term, posting in postings.items():
            if sys.byteorder != 'little':
                posting.byteswap()
            connection.execute(
                'INSERT INTO postings VALUES (?, ?, ?)',
                (term, len(posting), posting.tobytes()),
            )
        connection.execute('PRAGMA user_version = 1')
        connection.commit()
        connection.close()
        connection = None
        os.replace(temporary_path, index_dir / 'index.sqlite3')
    finally:
        if connection is not None:
            connection.close()
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)


def _matching_documents(connection, terms, document_count):
    """Intersect posting bitmaps; only one posting list is loaded at a time."""
    ordered_terms = []
    for term in terms:
        row = connection.execute(
            'SELECT frequency FROM postings WHERE term = ?', (term,)
        ).fetchone()
        if row is None:
            return 0
        ordered_terms.append((row[0], term))

    matches = (1 << document_count) - 1
    for _, term in sorted(ordered_terms):
        blob = connection.execute('SELECT data FROM postings WHERE term = ?', (term,)).fetchone()[0]
        posting = array('I')
        posting.frombytes(blob)
        if sys.byteorder != 'little':
            posting.byteswap()

        # Создаем массив байтов. Каждый байт это 8 документов.
        # Функционально та же битмаска, но не создает новое огромное число на каждое изменение
        bitmap = bytearray((document_count + 7) // 8)
        for ordinal in posting:
            bitmap[ordinal // 8] |= 1 << (ordinal % 8)
        matches &= int.from_bytes(bitmap, 'little')
        if not matches:
            break
    return matches


def make_submission(data_dir, index_dir, submission_file):
    """Stream object pairs, caching at most 4096 compact query result bitmaps."""
    data_dir = Path(data_dir)
    index_path = (Path(index_dir) / 'index.sqlite3').resolve()
    connection = sqlite3.connect(index_path.as_uri() + '?mode=ro', uri=True)
    try:
        connection.execute('PRAGMA cache_size = -4096')
        if connection.execute('PRAGMA user_version').fetchone()[0] != 1:
            raise ValueError('Unsupported index format; rebuild the index')
        documents = dict(connection.execute('SELECT id, ordinal FROM documents'))
        queries = {}
        with (data_dir / 'vkmarco-doceval-queries.tsv').open(encoding='utf-8') as source:
            for line in source:
                query_id, text = line.rstrip('\r\n').split('\t', 1)
                if query_id in queries:
                    raise ValueError(f'Duplicate QueryId: {query_id}')
                queries[query_id] = tuple(sorted(set(preprocess(text))))

        # All 3000 homework queries fit; 4096 results use about 32 MB at 60k docs.
        @lru_cache(maxsize=4096)
        def find_matches(terms):
            return _matching_documents(connection, terms, len(documents))

        with (data_dir / 'objects.csv').open(encoding='utf-8', newline='') as source, \
                Path(submission_file).open('w', encoding='utf-8', newline='') as output:
            reader = csv.DictReader(source)
            if not {'ObjectId', 'QueryId', 'DocumentId'}.issubset(reader.fieldnames or []):
                raise ValueError('objects.csv must contain ObjectId,QueryId,DocumentId')
            writer = csv.writer(output)
            writer.writerow(['ObjectId', 'Label'])
            for row in reader:
                query_id = row['QueryId']
                if query_id not in queries:
                    raise ValueError(f'Unknown QueryId: {query_id}')
                ordinal = documents.get(row['DocumentId'])
                matches = find_matches(queries[query_id])
                label = int(ordinal is not None and (matches >> ordinal) & 1)
                writer.writerow([row['ObjectId'], label])
    finally:
        connection.close()


def main():
    parser = argparse.ArgumentParser(description='Boolean retrieval homework solution')
    parser.add_argument('--submission_file', help='куда записать сабмишн для Kaggle')
    parser.add_argument('--build_index', action='store_true', help='режим построения индекса')
    parser.add_argument('--index_dir', required=True, help='папка с индексом')
    parser.add_argument('data_dir', help='папка с данными соревнования')
    args = parser.parse_args()

    start = timer()
    if args.build_index:
        build_index(args.data_dir, args.index_dir)
    else:
        if not args.submission_file:
            parser.error('в режиме генерации сабмишна нужен --submission_file')
        make_submission(args.data_dir, args.index_dir, args.submission_file)
    print(f'finished, elapsed = {timer() - start:.3f}')


if __name__ == '__main__':
    main()
