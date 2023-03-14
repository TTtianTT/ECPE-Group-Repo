"""
Load in data, and interactively filter sentences to give what we want
"""

import json
import argparse
from os.path import join as pjoin
from itertools import izip
from preprocessing.cfg import EN_DISCOURSE_MARKERS
from data import get_dis
import itertools
import random

import numpy as np
import IPython

parser = argparse.ArgumentParser(description='DAT data generation')
parser.add_argument("--corpus", type=str, default='books_all',
                    help="books_5|books_old_5|books_8|books_all|gw_5|gw_8")
parser.add_argument("--hypes", type=str, default='hypes/default.json', help="load in a hyperparameter file")
parser.add_argument("--senteval", type=str, default='~/SentEval/data/senteval_data',
                    help="point to senteval data directory")
parser.add_argument("--seed", type=int, default=1234, help="seed")
parser.add_argument("--train_size", default=0.9, type=float)
parser.add_argument("--gen_senteval", action='store_true', help="generate a dataset to senteval")
parser.add_argument("--gen_dis", action='store_true', help="generate a dataset to DIS training format")

# 6 vs. 8
order_invar_list = ['but', 'and', 'also', 'while', 'as', 'when']
order_dep_list = ['because', 'so', 'if', 'although',
                  'though', 'before', 'after', 'then']

# load in the corpus
params, _ = parser.parse_known_args()

# DAT, and when we add 2nd order operation like substitution, we can call it DATS
DAT_dir = pjoin(params.senteval, "DAT")
random.seed(params.seed)
np.random.seed(params.seed)

"""
Default json file loading
"""
with open(params.hypes, 'rb') as f:
    json_config = json.load(f)

data_dir = json_config['data_dir']
prefix = json_config[params.corpus]

"""
Data
"""
# no_train = True if params.gen_senteval else False
# since it's doing so well...let's try not having everything
marker_dict = get_dis(data_dir, prefix, params.corpus, no_train=True)

split_proportions = {
    "train": params.train_size,
    "valid": (1 - params.train_size) / 2,
    "test": (1 - params.train_size) / 2
}

"""
Search
"""


def display_marker():
    print marker_dict.keys()


def get_sent(marker, idx=-1, rand=True, display=True):
    if idx != -1:
        rand = False

    if rand:
        nums = range(len(marker_dict[marker]))
        np.random.shuffle(nums)
        ex_num = nums[0]

        selection = marker_dict[marker][ex_num]
    else:
        selection = marker_dict[marker][idx]

    if not display:
        return selection
    else:
        print selection[0][:-1] + marker + " " + selection[1]


def get_sents(marker, st=0, en=10, rand=False, display=True):
    if rand:
        nums = range(len(marker_dict[marker]))
        np.random.shuffle(nums)
        choices = nums[st:en]
        selected = [marker_dict[marker][ch] for ch in choices]
    else:
        selected = marker_dict[marker][st:en]

    if not display:
        return selected
    else:
        for s in selected:
            print s[0][:-1] + marker + " " + s[1]


"""
Generate SentEval dataset
"""


def write_to_file(file_name, data, assignments, split_num):
    # split_num=0: s1, split_num=1: s2, split_num=2: label
    with open(pjoin(DAT_dir, file_name), 'w') as f:
        for a in assignments:
            f.write(data[a][split_num] + '\n')


# finish writing this part...
# 35 minutes
# Desired properties:
# 1. Overall, balanced binary class "entailment", "contradict"
# 2. For each discourse marker, equal number per marker
# (in order to achieve 2, we need distribution of each marker)
# this is the train+val+test stats...
#
# and 102003
# then 1652
# because 16713
# though 10370
# after 9564
# when 52629
# as 75113
# but 102682
# also 1595      (fewest)
# while 16178
# so 7693
# although 3759
# before 22015
# if 47289

# Plan: to satisfy (1), let's sample pairs, entail and contradict together
# but since invar markers only have "entail", does not have contradict...
# so actual "pairs" are pairing invar and dep markers...
# To satisfy (2), we sample each discourse marker according to fewest marker (make sure it's even number)

# assume we get 1594 * 6 = total number of "entail"
# then (total number of "entail") / 8

# 19128

def generate_senteval():
    fewest_ex = 1e10
    fewest_key = ""
    for k in order_invar_list + order_dep_list:
        ex = len(marker_dict[k])
        if ex < fewest_ex:
            fewest_ex = ex
            fewest_key = k

    print "we threshold at {}, limited by marker {}".format(fewest_ex, fewest_key)

    print "sampling invariant at {}, dep at {}".format(fewest_ex, fewest_ex * 6 / 8)

    dataset = []
    # we store pairs like [s1, s2, label]

    # I could also swap sent1 and sent2 in final output, but probably not necessary
    for marker in order_invar_list:  # 6 of those

        # fewest_ex / 2 to have half as "neutral"
        sents = get_sents(marker, en=fewest_ex / 2, rand=True, display=False)  # [[s1, s2]]
        for s1, s2 in sents:
            # this is the "flip" action
            sent1 = s1[:-1] + marker + " " + s2[0].lower() + s2[1:]
            sent2 = s2[:-1] + marker + " " + s1[0].lower() + s1[1:]
            dataset.append([sent1, sent2, 'entail'])

            # we randomly sample one to be "neutral" (condition to be different)
            # s1_neutral, s2_neutral = get_sent(marker, rand=True, display=False)
            # while s1_neutral == s1 and s2_neutral == s2:
            #     s1_neutral, s2_neutral = get_sent(marker, rand=True, display=False)
            # s1, s2 = s1_neutral, s2_neutral
            # new_sent1 = s1[:-1] + marker + " " + s2[0].lower() + s2[1:]  # a random sent1 with original sent2
            # new_sent2 = s2[:-1] + marker + " " + s1[0].lower() + s1[1:]
            #
            # coin = random.randint(1, 2)
            # if coin == 1:
            #     dataset.append([new_sent1, sent2, 'neutral'])
            # else:
            #     dataset.append([sent1, new_sent2, 'neutral'])

    contra_ex = fewest_ex * 6 / 8
    for marker in order_dep_list:

        # contra_ex / 2 to have half as "neutral"
        sents = get_sents(marker, en=contra_ex / 2, rand=True, display=False)
        for s1, s2 in sents:
            # this is the "flip" action
            sent1 = s1[:-1] + marker + " " + s2[0].lower() + s2[1:]
            sent2 = s2[:-1] + marker + " " + s1[0].lower() + s1[1:]
            dataset.append([sent1, sent2, 'contradict'])

            # we randomly sample one to be "neutral" (condition to be different)
            # s1_neutral, s2_neutral = get_sent(marker, rand=True, display=False)
            # while s1_neutral == s1 and s2_neutral == s2:
            #     s1_neutral, s2_neutral = get_sent(marker, rand=True, display=False)
            # s1, s2 = s1_neutral, s2_neutral
            # new_sent1 = s1[:-1] + marker + " " + s2[0].lower() + s2[1:]  # a random sent1 with original sent2
            # new_sent2 = s2[:-1] + marker + " " + s1[0].lower() + s1[1:]
            #
            # coin = random.randint(1, 2)
            # if coin == 1:
            #     dataset.append([new_sent1, sent2, 'neutral'])
            # else:
            #     dataset.append([sent1, new_sent2, 'neutral'])

    # shuffle 2 times
    random.shuffle(dataset)
    random.shuffle(dataset)

    num_examples = len(dataset)
    assignments = range(num_examples)
    np.random.shuffle(assignments)

    train_numbers = assignments[:int(np.rint(num_examples * split_proportions['train']))]
    valid_numbers = assignments[int(np.rint(num_examples * split_proportions['train'])): int(
        np.rint(num_examples * (split_proportions['train'] + split_proportions['valid'])))]
    test_numbers = assignments[int(np.rint(num_examples * (split_proportions['train'] + split_proportions['valid']))):]

    print "train {}, dev {}, test {}".format(len(train_numbers), len(valid_numbers), len(test_numbers))

    # everything above here is actually the same for SentEval or DIS training...

    write_to_file("s1.train", dataset, train_numbers, 0)
    write_to_file("s2.train", dataset, train_numbers, 1)
    write_to_file("labels.train", dataset, train_numbers, 2)

    write_to_file("s1.dev", dataset, valid_numbers, 0)
    write_to_file("s2.dev", dataset, valid_numbers, 1)
    write_to_file("labels.dev", dataset, valid_numbers, 2)

    write_to_file("s1.test", dataset, test_numbers, 0)
    write_to_file("s2.test", dataset, test_numbers, 1)
    write_to_file("labels.test", dataset, test_numbers, 2)


"""
Generate DisSent training files
"""


def write_to_dis_file(file_name, dataset, assignments):
    with open(pjoin(data_dir, file_name), 'w') as f:
        for a in assignments:
            f.write("\t".join(dataset[a]) + '\n')


def generate_dis():
    fewest_ex = 1e10
    fewest_key = ""
    for k in order_invar_list + order_dep_list:
        ex = len(marker_dict[k])
        if ex < fewest_ex:
            fewest_ex = ex
            fewest_key = k

    print "we threshold at {}, limited by marker {}".format(fewest_ex, fewest_key)

    print "sampling invariant at {}, dep at {}".format(fewest_ex, fewest_ex * 6 / 8)

    dataset = []
    # we store pairs like [s1, s2, label]

    # I could also swap sent1 and sent2 in final output, but probably not necessary
    for marker in order_invar_list:  # 6 of those
        sents = get_sents(marker, en=fewest_ex, rand=True, display=False)  # [[s1, s2]]
        for s1, s2 in sents:
            # this is the "flip" action
            sent1 = s1[:-1] + marker + " " + s2[0].lower() + s2[1:]
            sent2 = s2[:-1] + marker + " " + s1[0].lower() + s1[1:]
            dataset.append([sent1, sent2, 'entail'])

    contra_ex = fewest_ex * 6 / 8
    for marker in order_dep_list:
        sents = get_sents(marker, en=contra_ex, rand=True, display=False)
        for s1, s2 in sents:
            # this is the "flip" action
            sent1 = s1[:-1] + marker + " " + s2[0].lower() + s2[1:]
            sent2 = s2[:-1] + marker + " " + s1[0].lower() + s1[1:]
            dataset.append([sent1, sent2, 'contradict'])

    # shuffle 2 times
    random.shuffle(dataset)
    random.shuffle(dataset)

    num_examples = len(dataset)
    assignments = range(num_examples)
    np.random.shuffle(assignments)

    train_numbers = assignments[:int(np.rint(num_examples * split_proportions['train']))]
    valid_numbers = assignments[int(np.rint(num_examples * split_proportions['train'])): int(
        np.rint(num_examples * (split_proportions['train'] + split_proportions['valid'])))]
    test_numbers = assignments[int(np.rint(num_examples * (split_proportions['train'] + split_proportions['valid']))):]

    print "train {}, dev {}, test {}".format(len(train_numbers), len(valid_numbers), len(test_numbers))

    # everything above here is actually the same for SentEval or DIS training...
    file_name = "discourse_EN_DAT_ALL_2018feb26"

    write_to_dis_file(file_name + "_train.tsv", dataset, train_numbers)
    write_to_dis_file(file_name + "_valid.tsv", dataset, valid_numbers)
    write_to_dis_file(file_name + "_test.tsv", dataset, test_numbers)


if __name__ == '__main__':
    if not params.gen_senteval and not params.gen_dis:
        # call in console
        IPython.embed()
    elif params.gen_senteval:
        generate_senteval()
    elif params.gen_dis:
        generate_dis()
