# -*- coding: utf-8 -*-

"""
Preprocess gigaword chinese 5th edition

We are only process text type of "story", and ignore the rest.
"story" is the most frequent type in this corpus.
"""

import os
import re
import io
import sys
import json
import gzip
import argparse

import logging
from util import rephrase
from os.path import join as pjoin

from parser import depparse_ssplit, setup_corenlp
from cfg import SP_DISCOURSE_MARKERS

"""
Stats:
Raw gigaword es file has: X (XM) sentences

Unlike bookcorpus.py, we are not filtering anything (due to difficulty in tokenization for raw string)
"""

parser = argparse.ArgumentParser(description='DisExtract Gigaword Spanish')

parser.add_argument("--json", type=str, default="example_config.json", help="corpus parameter setting to load")
parser.add_argument("--extract", action='store_true')

parser.add_argument("--filter", action='store_true',
                    help="Stage 2: run filtering on the corpus, collect sentence pairs (sentence and previous sentence)")
parser.add_argument("--filter_print_every", default=10000, type=int)
parser.add_argument("--max_seq_len", default=50, type=int)
parser.add_argument("--min_seq_len", default=5, type=int)

parser.add_argument("--parse", action='store_true',
                    help="Stage 3: run parsing on filtered sentences, collect sentence pairs (S1 and S2)")
parser.add_argument("--exclude_list", action='store_true', help="use exclusion list defined in this file")
parser.add_argument("--no_dep_cache", action='store_false', help="not caching dependency parsed result")

args, _ = parser.parse_known_args()

"""
Logging
"""

logging.basicConfig(format='[%(asctime)s] %(levelname)s: %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger(__name__)

with open(args.json, 'rb') as f:
    json_config = json.load(f)

gigaword_sp_dir = json_config['gigaword_sp_dir']
gigaword_sp_file = 'gigaword_sp.txt'

import re

import nltk
spanish_tokenizer = nltk.data.load('tokenizers/punkt/spanish.pickle')


def process_sent(sent, lang="sp"):
    #sent = re.sub(r"\(.+\)", "", sent)  # get rid of parentheses (many content inside are English/other languages)

    sent = sent.replace("&amp;gt;", "")

    # HTML entities
    sent = sent.replace("&lt;", '<')
    sent = sent.replace("&gt;", '>')
    sent = sent.replace("&amp;", '&')
    sent = sent.replace("&apos;", '\'')
    sent = sent.replace("&quot;", '"')

    return sent


def extract_stories(lines):
    # pass in all lines from a gigaword xml file
    sentences = []

    story_doc = False
    paragraph = False
    paragraph_text = []
    for line in lines:
        if 'DOC' in line and 'type="story"' in line:
            story_doc = True
        if '<P>' in line and story_doc:
            paragraph = True
            continue
        if '</P>' in line and story_doc:
            paragraph = False
            sentence = "".join(paragraph_text).strip()
            # preprocess the sentence
            sentence = process_sent(sentence)
            sentences.append(sentence)
            paragraph_text = []
        if '</DOC>' in line and story_doc:
            story_doc = False

        if paragraph:
            paragraph_text.append(line)

    return sentences


def sent_tokenize(p):
    sents = spanish_tokenizer.tokenize(p)
    return sents


def extrat_raw_gigaword():
    news_sources = os.listdir(pjoin(gigaword_sp_dir, 'data'))
    articles_processed = 0
    sentences = []
    for news_source in news_sources:
        files = os.listdir(pjoin(gigaword_sp_dir, 'data', news_source))
        files = filter(lambda s: '.gz' in s, files)
        for file in files:
            with gzip.open(pjoin(gigaword_sp_dir, 'data', news_source, file), 'rb') as f:
                file_content = f.read()
                lines = file_content.split('\n')
                sents = extract_stories(lines)
                sentences.extend(sents)
            articles_processed += 1
            if articles_processed % 20 == 0:
                print("processed {} articles".format(articles_processed))
                print("{} paragraphs are collected".format(len(sentences)))

    with open(pjoin(gigaword_sp_dir, gigaword_sp_file), 'wb') as f:
        for sent in sentences:
            f.write(sent + '\n')

# def generate_pairs(sent_splits, marker_sent_list, marker_prev_list):
#     # successive pairs
#     for s1, s2 in zip(sent_splits, sent_splits[1:]):
#         if len(s1.decode("utf-8")) > args.max_seq_len or len(s1.decode("utf-8")) < args.min_seq_len:
#             return
#         elif len(s2.decode("utf-8")) > args.max_seq_len or len(
#                 s2.decode("utf-8")) < args.min_seq_len:
#             return
#         marker_sent_list.append(sent)
#         marker_prev_list.append(previous_sentence)

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

    sentences = {marker: [] for marker in discourse_markers}

    for filename in filenames:
        logger.info("reading {}".format(filename))
        file_path = pjoin(source_dir, filename)

        previous_sentence = ""
        with io.open(file_path, 'rU', encoding="utf-8") as f:
            for i, sentence in enumerate(f):

                # sentence tokenization here!!! <P> is not sentence.
                sents = sent_tokenize(sentence) # these are already preprocessed

                for sent in sents:
                    sent = sent.replace("\t", "")
                    # a single marker match, so continue is fine
                    # we match the sentence length condition, but when there are multiple
                    # markers, we sync with spanish, and defer the decision to parser!
                    words = sent.lower().split(" ")
                    for marker in discourse_markers:
                      if len(marker.split(" ")) > 1:
                        if marker in sent:
                          line_to_write = "{}\t{}\t{}".format(sent, previous_sentence, marker)
                          sentences[marker].append(line_to_write)
                      else:
                        if marker in words:
                          line_to_write = "{}\t{}\t{}".format(sent, previous_sentence, marker)
                          sentences[marker].append(line_to_write)

                    previous_sentence = sent

                    if i % args.filter_print_every == 0:
                        logger.info("processed {}".format(i))

        logger.info("{} file finished".format(filename))

    logger.info('writing files')

    with open(pjoin(output_dir, "{}.tsv".format(marker_set_tag)), 'a') as f:
      for marker in discourse_markers:
        for line in sentences[marker]:
          f.write(line + "\n")

    logger.info('file writing complete')

    statistics_lines = []
    for marker in sentences:
        n_sentences = len(sentences[marker])
        statistics_lines.append("{}\t{}".format(marker, n_sentences))

    statistics_report = "\n".join(statistics_lines)
    with open(pjoin(markers_dir, "VERSION.txt"), "wb") as f:
        f.write(
            "commit: \n\ncommand: \n\nmarkers:\n" + statistics_report
        )


def parse_filtered_sentences(source_dir, marker_set_tag):
    """
    This function can be the same for each corpus

    :param source_dir:
    :param marker_set_tag:
    :return:
    """

    markers_dir = pjoin(source_dir, "markers_" + marker_set_tag)
    input_dir = pjoin(markers_dir, "sentences")
    input_file_path = pjoin(input_dir, "{}.tsv".format(marker_set_tag))
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

        with open(input_file_path, 'rb') as f:
            logger.info("reading {}".format(input_file_path))
            i = 0
            for line in f:
              sentence, previous, marker = line[:-1].split("\t")
              i+=1
              if i > 0:
                #try:
                  parsed_output = dependency_parsing(sentence, previous, marker)
                  if parsed_output:
                    s1, s2 = parsed_output
                    line_to_print = "{}\t{}\t{}\n".format(s1, s2, marker)
                    w.write(line_to_print)
                #except:
                #  print i, marker, sentence
              if i % args.filter_print_every == 0:
                logger.info("processed {}".format(i))
              #stop
            #logger.info("total sentences: {}".format(
            #    sum([len(sentences[marker]["sentence"]) for marker in sentences])
            #))

    logger.info('file writing complete')


def dependency_parsing(sentence, previous_sentence, marker):
    return depparse_ssplit(sentence, previous_sentence, marker, lang='sp')


if __name__ == '__main__':
    if args.extract:
        extrat_raw_gigaword()
    elif args.filter:
        collect_raw_sentences(gigaword_sp_dir, [gigaword_sp_file], "ALL", SP_DISCOURSE_MARKERS)
    elif args.parse:
        setup_corenlp("sp")
        parse_filtered_sentences(gigaword_sp_dir, "ALL")
