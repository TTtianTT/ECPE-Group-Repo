#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""

"""

import numpy as np
import argparse
import io
import nltk
import pickle
import requests
import re

from parser import depparse_ssplit, setup_corenlp
from dep_patterns import en_dependency_patterns, ch_dependency_patterns, sp_dependency_patterns
dependency_patterns = None

import sys
reload(sys)
sys.setdefaultencoding('utf8')

import os
from os.path import join as pjoin

import json
from itertools import izip

from copy import deepcopy as cp

np.random.seed(123)

def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", type=str, default='en', help="en|ch|es")
    return parser.parse_args()


def test(lang):
    if lang == "en":
        data = json.load(open("en_tests.json"))
    elif lang == "sp":
        data = json.load(open("sp_tests.json"))
    elif lang == "ch":
        data = json.load(open("ch_tests.json"))

    test_items = data["test_items"]
    curious_cases = data["curious_cases"]
        
    print("{} cases are weird and I can't figure out how to handle them. :(".format(len(curious_cases)))
    # print("{} of those incorrectly return None".format(len([c for c in curious_cases if depparse_ssplit(c["sentence"], c["previous_sentence"], c["marker"], "en")==None])))
    print("{} parsable cases are being tested".format(len(test_items)))
    curious=False
    if curious:
        print("running curious cases...")
        for item in curious_cases:
            print("====================")
            print(item["sentence"])
            output = depparse_ssplit(item["sentence"], item["previous_sentence"], item["marker"], lang)
            if output:
                output = list(output)
            print(output)
        print("====================")
        print("====================")
        print("====================")
    marker_accuracy=False
    if marker_accuracy:
        markers = set([c["marker"] for c in test_items + curious_cases])
        for marker in markers:
            correct = len([c for c in test_items if c["marker"]==marker])
            incorrect = len([c for c in curious_cases if c["marker"]==marker])
            accuracy = float(correct) / (correct+incorrect)
            print("{} ~ {}".format(marker, accuracy))


    # n_tests = 79
    # i = 0
    failures = 0
    print("running tests...")

    for item in test_items:
        # if i < n_tests:
            output = depparse_ssplit(item["sentence"], item["previous_sentence"], item["marker"], lang)
            if output:
                output = "[\"" + "\", \"".join(list(output)) + "\"]"
            try:
                assert(output == item["output"])
            except AssertionError:
                print("====== TEST FAILED ======" + "\nsentence: " + item["sentence"] + "\nmarker: " + item["marker"] + "\nactual output: " + str(output) + "\ndesired output: " + str(item["output"]))
                failures += 1
        # else:
        #     print("====================")
        #     print(item["sentence"])
        #     output = depparse_ssplit(item["sentence"], item["previous_sentence"], item["marker"])
        #     print(output)
        # i += 1

    if failures==0:
        print("All tests passed.")

if __name__ == '__main__':
    args = setup_args()
    setup_corenlp(args.lang)
    test(args.lang)

