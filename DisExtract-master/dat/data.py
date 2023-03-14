import os
import numpy as np
import torch
import logging
from os.path import join as pjoin
from preprocessing.cfg import EN_FIVE_DISCOURSE_MARKERS, \
    EN_EIGHT_DISCOURSE_MARKERS, EN_DISCOURSE_MARKERS, EN_OLD_FIVE_DISCOURSE_MARKERS, \
    CH_FIVE_DISCOURSE_MARKERS, SP_FIVE_DISCOURSE_MARKERS

def list_to_map(dis_label):
    dis_map = {}
    for i, l in enumerate(dis_label):
        dis_map[l] = i
    return dis_map

def get_dis(data_dir, prefix, discourse_tag="books_5", no_train=False):
    marker_dict = {}

    if discourse_tag == "books_5":
        dis_map = list_to_map(EN_FIVE_DISCOURSE_MARKERS)
    elif discourse_tag == "books_8":
        dis_map = list_to_map(EN_EIGHT_DISCOURSE_MARKERS)
    elif discourse_tag == "books_all" or discourse_tag == "books_perfectly_balanced" or discourse_tag == "books_mostly_balanced":
        dis_map = list_to_map(EN_DISCOURSE_MARKERS)
    elif discourse_tag == "books_old_5":
        dis_map = list_to_map(EN_OLD_FIVE_DISCOURSE_MARKERS)
    elif discourse_tag == "gw_cn_5":
        dis_map = list_to_map(CH_FIVE_DISCOURSE_MARKERS)
    elif discourse_tag == "gw_es_5":
        dis_map = list_to_map(SP_FIVE_DISCOURSE_MARKERS)
    elif discourse_tag == "gw_es_1M_5":
        dis_map = list_to_map(SP_FIVE_DISCOURSE_MARKERS)
    else:
        raise Exception("Corpus/Discourse Tag Set {} not found".format(discourse_tag))

    logging.info(dis_map)

    for dis_marker in dis_map.keys():
        marker_dict[dis_marker] = []

    splits = ['valid', 'test'] if no_train else ['train', 'valid', 'test']

    for data_type in splits:

        text_path = pjoin(data_dir, prefix + "_" + data_type + ".tsv")

        with open(text_path, 'r') as f:
            for line in f:
                columns = line.split('\t')
                # we use this to avoid/skip lines that are empty
                if len(columns) != 3:
                    continue

                dis_marker = columns[2].rstrip('\n')
                marker_dict[dis_marker].append([columns[0], columns[1]])

    dis_stats = [(tup[0], len(tup[1])) for tup in marker_dict.items()]
    logging.info(dis_stats)

    return marker_dict