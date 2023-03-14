#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
This goes through the corpus,
select sentences that have the discourse marker
filtering based on length of sentence (so we can discard ill-formed sentences) 100 words
and save them as intermediate files
shuffle within each discourse marker
"""

import os
import io
import json
import argparse

import logging
import nltk
from util import rephrase
from os.path import join as pjoin

from parser import depparse_ssplit, setup_corenlp
from cfg import DISCOURSE_MARKER_SET_TAG, EN_DISCOURSE_MARKERS, EN_FIVE_DISCOURSE_MARKERS, EN_EIGHT_DISCOURSE_MARKERS

import sys

"""
This file contains WikiText-specific information

1. Sentence tokenization (make it config)
2. Grab pairs of sentences where the 2nd sentence has one of the discourse markers
3. Save them, a file for each discourse markers (a json file with [,] is good enough)
for_example.txt, in side it's [{prev: "", sent: ""}]
"""


parser = argparse.ArgumentParser(description='DisExtract BookCorpus')

parser.add_argument("--json", type=str, default="example_config.json", help="corpus parameter setting to load")

parser.add_argument("--filter", action='store_true',
                    help="Stage 1: run filtering on the corpus, collect sentence pairs (sentence and previous sentence)")
parser.add_argument("--max_seq_len", default=50, type=int)
parser.add_argument("--min_seq_len", default=5, type=int)
parser.add_argument("--max_ratio", default=5.0, type=float)
parser.add_argument("--filter_print_every", default=10000, type=int)

parser.add_argument("--parse", action='store_true',
                    help="Stage 2: run parsing on filtered sentences, collect sentence pairs (S1 and S2)")
# parser.add_argument("--no_dep_cache", action='store_true', help="not caching dependency parsed result")

parser.add_argument("--split", action='store_true',
                    help="Stage 3: load in parsed sentences pairs and split into discourse marker set based groups")
parser.add_argument("--tag", type=str, default="discourse_EN", help="the tag of the generated file such as discourse_EN_FIVE_and_but_because_if_when_2017dec12.tsv")

args, _ = parser.parse_known_args()
args.min_ratio = 1 / args.max_ratio  # auto-generate min-ratio

"""
Logging
"""

logging.basicConfig(format='[%(asctime)s] %(levelname)s: %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger(__name__)

"""
Default json file loading
"""
with open(args.json, 'rb') as f:
    json_config = json.load(f)

wikitext_dir = json_config['wikitext_dir']
wikitext_files = ['wiki.train.tokens', 'wiki.valid.tokens', 'wiki.test.tokens']


def collect_raw_sentences(source_dir, filenames, marker_set_tag, discourse_markers):
    """
    This function needs to be implemented differently for each corpus
    since it contains crucial corpus-specific functions, though
    the main logic remains the same

    :param source_dir:
    :param filenames:
    :param marker_set_tag:
    :param discourse_markers:
    :return:
    """
    markers_dir = pjoin(source_dir, "markers_" + marker_set_tag)
    output_dir = pjoin(markers_dir, "sentences")

    if not os.path.exists(markers_dir):
        os.makedirs(markers_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    sentences = {marker: {"sentence": [], "previous": []} for marker in discourse_markers}

    for filename in filenames:
        logger.info("reading {}".format(filename))
        file_path = pjoin(source_dir, filename)

        previous_sentence = ""
        previous_sentence_split = None
        FIRST = True
        with io.open(file_path, 'rU', encoding="utf-8") as f:
            for i, line in enumerate(f):

                # this is wikitext-103, so we need to split the paragraph
                # we also need to ignore the header of each paragraph
                if len(line.strip()) == 0:
                    continue

                if line.split()[0] == "=" and line.split()[-1] == "=":
                    continue

                sentence_list = nltk.sent_tokenize(line)

                for sentence in sentence_list:

                    words = rephrase(sentence).split()  # replace "for example"
                    for marker in discourse_markers:
                        if marker == "for example":
                            proxy_marker = "for_example"
                        else:
                            proxy_marker = marker

                        # [min_len, max_len) like [5, 10)
                        if len(words) >= args.max_seq_len or len(words) < args.min_seq_len:
                            continue

                        # length-based filtering
                        if not FIRST:
                            # because parser might request previous sentence
                            # we are here to control the balance. This is not s1 / s2 ratio.
                            len2 = len(previous_sentence_split)
                            ratio = float(len2) / len(words)

                            if ratio <= args.min_ratio or ratio >= args.max_ratio:
                                continue

                        # all bookcorpus text are lower case
                        if proxy_marker in words:
                            sentences[marker]["sentence"].append(sentence)
                            sentences[marker]["previous"].append(previous_sentence)

                    previous_sentence = sentence
                    previous_sentence_split = words

                if i % args.filter_print_every == 0:
                    logger.info("processed {}".format(i))

        logger.info("{} file finished".format(filename))

    logger.info('writing files')

    with open(pjoin(output_dir, "{}.json".format(marker_set_tag)), 'wb') as f:
        json.dump(sentences, f)

    logger.info('file writing complete')

    statistics_lines = []
    for marker in sentences:
        n_sentences = len(sentences[marker]["sentence"])
        statistics_lines.append("{}\t{}".format(marker, n_sentences))

    statistics_report = "\n".join(statistics_lines)
    with open(pjoin(markers_dir, "VERSION.txt"), "wb") as f:
        f.write(
            "commit: \n\ncommand: \n\nmarkers:\n" + statistics_report
        )


def parse_filtered_sentences(source_dir, filenames, marker_set_tag, discourse_markers):
    """
    This function can be the same for each corpus

    :param source_dir:
    :param filenames:
    :param marker_set_tag:
    :param discourse_markers:
    :return:
    """

    markers_dir = pjoin(source_dir, "markers_" + marker_set_tag)
    input_dir = pjoin(markers_dir, "sentences")
    input_file_path = pjoin(input_dir, "{}.json".format(marker_set_tag))
    output_dir = pjoin(markers_dir, "parsed_sentence_pairs")

    if not os.path.exists(markers_dir):
        raise Exception("{} does not exist".format(markers_dir))
    if not os.path.exists(input_dir):
        raise Exception("{} does not exist".format(input_dir))
    if not os.path.exists(input_file_path):
        raise Exception("{} does not exist".format(input_file_path))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger.info("setting up parser (actually just testing atm)")
    setup_corenlp()

    # parsed_sentence_pairs = {marker: {"s1": [], "s2": []} for marker in discourse_markers}
    with open(pjoin(output_dir, "{}_parsed_sentence_pairs.txt".format(marker_set_tag)), 'a') as w:
        # header = "{}\t{}\t{}\n".format("s1", "s2", "marker")
        # w.write(header)

        with open(input_file_path, 'rb') as f:
            logger.info("reading {}".format(input_file_path))
            sentences = json.load(f)
            logger.info("total sentences: {}".format(
                sum([len(sentences[marker]["sentence"]) for marker in sentences])
            ))
            for marker, slists in sentences.iteritems():
                i = 0
                if marker in discourse_markers:
                    # if marker == "because":
                    for sentence, previous in set(zip(slists["sentence"], slists["previous"])):
                        i += 1
                        if True:
                            parsed_output = dependency_parsing(sentence, previous, marker)
                            if parsed_output:
                                s1, s2 = parsed_output

                                line_to_print = "{}\t{}\t{}\n".format(s1, s2, marker)
                                w.write(line_to_print)

                            if i % args.filter_print_every == 0:
                                logger.info("processed {}".format(i))

    logger.info('file writing complete')


def dependency_parsing(sentence, previous_sentence, marker):
    try:
        return depparse_ssplit(sentence, previous_sentence, marker)
    except:
        return None

from collections import defaultdict

def split_parsed_sentences(source_dir, marker_set_tag):
    markers_dir = pjoin(source_dir, "markers_" + marker_set_tag)
    input_dir = pjoin(markers_dir, "sentences")
    input_file_path = pjoin(input_dir, "{}.json".format(marker_set_tag))
    output_dir = pjoin(markers_dir, "parsed_sentence_pairs")

    five_sents = []
    eight_sents = []
    all_sents = []

    sent_hash_set = set()  # we expect repeating entries
    marker_stats = defaultdict(int)

    with open(pjoin(output_dir, "{}_parsed_sentence_pairs.txt".format(marker_set_tag)), 'r') as f:
        # this is a tsv file
        for line in f:
            if line not in sent_hash_set:
                sent_hash_set.add(line)
                row = line.strip().split('\t')
                marker_stats[row[2]] += 1
                all_sents.append(row)
                if row[2] in EN_FIVE_DISCOURSE_MARKERS:
                    five_sents.append(row)
                if row[2] in EN_EIGHT_DISCOURSE_MARKERS:
                    eight_sents.append(row)

    for k, v in marker_stats.iteritems():
        print "{}: {}".format(k, v)

    with open(pjoin(output_dir, "stats.txt"), 'w') as f:
        for k, v in marker_stats.iteritems():
            f.write("{}: {}\n".format(k, v))

    with open(pjoin(output_dir, args.tag + "_FIVE_{}".format("_".join(EN_FIVE_DISCOURSE_MARKERS))), 'w') as f:
        for row in five_sents:
            f.write("{}\t{}\t{}\n".format(row[0], row[1], row[2]))

    with open(pjoin(output_dir, args.tag + "_EIGHT_{}".format("_".join(EN_EIGHT_DISCOURSE_MARKERS))), 'w') as f:
        for row in eight_sents:
            f.write("{}\t{}\t{}\n".format(row[0], row[1], row[2]))

    with open(pjoin(output_dir, args.tag + "_ALL_{}".format("_".join(EN_DISCOURSE_MARKERS))), 'w') as f:
        for row in all_sents:
            f.write("{}\t{}\t{}\n".format(row[0], row[1], row[2]))

if __name__ == '__main__':
    if args.filter:
        collect_raw_sentences(wikitext_dir, wikitext_files, DISCOURSE_MARKER_SET_TAG, EN_DISCOURSE_MARKERS)
    elif args.parse:
        parse_filtered_sentences(wikitext_dir, wikitext_files, DISCOURSE_MARKER_SET_TAG, EN_DISCOURSE_MARKERS)
    elif args.split:
        split_parsed_sentences(wikitext_dir, DISCOURSE_MARKER_SET_TAG)
