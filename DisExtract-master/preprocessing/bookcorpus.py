"""
Each corpus-specific files handle
reading of files as well as
calling functions from Filter and Parser
"""

import os
import io
import json
import argparse

import logging
from util import rephrase
from os.path import join as pjoin

from parser import depparse_ssplit, setup_corenlp
from cfg import DISCOURSE_MARKER_SET_TAG, EN_DISCOURSE_MARKERS

import sys

reload(sys)
sys.setdefaultencoding('utf8')

"""
Unlike Wikitext, we don't have sentence tokenization, and don't need to cache that.
But we do need to cache dependency parses.

This does filtering on max, min, ratio already (ratio should not be there...), 
to save dependency parsing time
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

books_dir = json_config['books_dir']
book_files = ['books_large_p1.txt', 'books_large_p2.txt']


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
            for i, sentence in enumerate(f):
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
                        # this part might be uncalled for...
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

                                # parsed_sentence_pairs[marker]["s1"].append(s1)
                                # parsed_sentence_pairs[marker]["s2"].append(s2)
                                line_to_print = "{}\t{}\t{}\n".format(s1, s2, marker)
                                w.write(line_to_print)

                            if i % args.filter_print_every == 0:
                                logger.info("processed {}".format(i))

    # logger.info('writing files')

    # with open(pjoin(output_dir, "{}_parsed_sentence_pairs.json".format(marker_set_tag)), 'wb') as f:
    #     json.dump(parsed_sentence_pairs, f)

    logger.info('file writing complete')


def dependency_parsing(sentence, previous_sentence, marker):
    try:
        return depparse_ssplit(sentence, previous_sentence, marker)
    except:
        return None


if __name__ == '__main__':
    if args.filter:
        collect_raw_sentences(books_dir, book_files, DISCOURSE_MARKER_SET_TAG, EN_DISCOURSE_MARKERS)
    elif args.parse:
        parse_filtered_sentences(books_dir, book_files, DISCOURSE_MARKER_SET_TAG, EN_DISCOURSE_MARKERS)
