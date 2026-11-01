"""Run with: python -m unittest discover -s homeworks/hw1-boolean-retrieval."""

import csv
from pathlib import Path
import random
import sqlite3
import subprocess
import sys
import tempfile
import unittest

import solution


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.index = self.root / 'index'
        self.output = self.root / 'submission.csv'

    def write_data(self, documents, queries, objects):
        for name, rows in [('vkmarco-docs.tsv', documents),
                           ('vkmarco-doceval-queries.tsv', queries)]:
            (self.root / name).write_text(
                ''.join('\t'.join(row) + '\n' for row in rows), encoding='utf-8'
            )
        with (self.root / 'objects.csv').open('w', newline='', encoding='utf-8') as output:
            writer = csv.writer(output)
            writer.writerow(['ObjectId', 'QueryId', 'DocumentId'])
            writer.writerows(objects)

    def read_output(self):
        with self.output.open(newline='', encoding='utf-8') as source:
            return list(csv.reader(source))

    def test_cli_unicode_title_body_and_identifiers(self):
        documents = [
            ('001', 'https://urlonly', 'МОСКВА, Cat cat', '"dog" 123 foo_bar'),
            ('90000000000', '', 'cat', 'Москва'),
            ('empty', '', '', ''),
        ]
        queries = [('q1', 'CAT dog cat'), ('q2', 'москва'), ('q3', 'urlonly'),
                   ('q4', '!!!'), ('q5', '123 foo_bar'), ('q6', 'absent')]
        objects = [('0001', 'q1', '001'), ('2', 'q1', '90000000000'),
                   ('3', 'q2', '90000000000'), ('4', 'q3', '001'),
                   ('5', 'q4', 'empty'), ('6', 'q4', 'missing'),
                   ('7', 'q5', '001'), ('8', 'q6', '001'), ('9,quoted', 'q1', '001')]
        self.write_data(documents, queries, objects)
        command = [sys.executable, str(Path(solution.__file__)), '--index_dir', str(self.index)]
        subprocess.run(command + ['--build_index', str(self.root)], check=True, capture_output=True)
        subprocess.run(command + ['--submission_file', str(self.output), str(self.root)],
                       check=True, capture_output=True)
        self.assertEqual(self.read_output(), [['ObjectId', 'Label']] + [
            [obj[0], str(label)] for obj, label in zip(objects, [1, 0, 1, 0, 1, 0, 1, 0, 1])
        ])

    def test_randomized_against_direct_matching(self):
        rng = random.Random(42)
        vocabulary = ['Кот', 'DOG', 'word_1', '123', 'café', 'other', 'Ёж']
        documents = [(str(i * 13), 'urlonly',
                      ' '.join(rng.choices(vocabulary, k=rng.randrange(8))),
                      ' '.join(rng.choices(vocabulary, k=rng.randrange(20))))
                     for i in range(150)]
        queries = [(str(i), ' '.join(rng.choices(vocabulary + ['missing'], k=rng.randrange(5))))
                   for i in range(200)]
        objects = [(str(i), rng.choice(queries)[0], rng.choice(documents)[0]) for i in range(4000)]
        self.write_data(documents, queries, objects)
        solution.build_index(self.root, self.index)
        solution.make_submission(self.root, self.index, self.output)
        doc_tokens = {key: set(solution.preprocess(title + ' ' + body))
                      for key, _, title, body in documents}
        query_tokens = {key: set(solution.preprocess(text)) for key, text in queries}
        self.assertEqual(self.read_output()[1:], [
            [obj_id, str(int(query_tokens[query_id] <= doc_tokens[doc_id]))]
            for obj_id, query_id, doc_id in objects
        ])

    def test_rebuild_and_failed_rebuild(self):
        self.write_data([('a', '', 'old', '')], [('q', 'new')], [('o', 'q', 'a')])
        solution.build_index(self.root, self.index)
        self.write_data([('a', '', 'new', '')], [('q', 'new')], [('o', 'q', 'a')])
        solution.build_index(self.root, self.index)
        with sqlite3.connect(self.index / 'index.sqlite3') as connection:
            self.assertEqual(connection.execute('SELECT term FROM postings').fetchall(), [('new',)])
        (self.root / 'vkmarco-docs.tsv').write_text('broken\n', encoding='utf-8')
        with self.assertRaises(ValueError):
            solution.build_index(self.root, self.index)
        solution.make_submission(self.root, self.index, self.output)
        self.assertEqual(self.read_output()[1:], [['o', '1']])
        self.assertEqual([p.name for p in self.index.iterdir()], ['index.sqlite3'])

    def test_empty_corpus_and_unknown_query(self):
        self.write_data([], [('q', '')], [('o', 'q', 'missing')])
        solution.build_index(self.root, self.index)
        solution.make_submission(self.root, self.index, self.output)
        self.assertEqual(self.read_output()[1:], [['o', '0']])
        self.write_data([], [], [('o', 'unknown', 'missing')])
        with self.assertRaisesRegex(ValueError, 'Unknown QueryId'):
            solution.make_submission(self.root, self.index, self.output)


if __name__ == '__main__':
    unittest.main()
