"""
Adapted from
https://github.com/facebookresearch/InferSent/blob/master/mutils.py
"""

import re
import inspect
from torch import optim

from preprocessing.cfg import EN_FIVE_DISCOURSE_MARKERS, EN_EIGHT_DISCOURSE_MARKERS, \
    EN_DISCOURSE_MARKERS, EN_OLD_FIVE_DISCOURSE_MARKERS, CH_FIVE_DISCOURSE_MARKERS, SP_FIVE_DISCOURSE_MARKERS


def get_labels(corpus):
    if corpus == "books_5":
        labels = EN_FIVE_DISCOURSE_MARKERS
    elif corpus == "books_old_5":
        labels = EN_OLD_FIVE_DISCOURSE_MARKERS
    elif corpus == "books_8":
        labels = EN_EIGHT_DISCOURSE_MARKERS
    elif corpus == "books_all" or corpus == "books_perfectly_balanced" or corpus == "books_mostly_balanced":
        labels = EN_DISCOURSE_MARKERS
    elif corpus == "gw_cn_5":
        labels = CH_FIVE_DISCOURSE_MARKERS
    elif corpus == "gw_es_5":
        labels = SP_FIVE_DISCOURSE_MARKERS
    elif corpus == "gw_es_1M_5":
        labels = SP_FIVE_DISCOURSE_MARKERS
    elif corpus == 'dat':
        labels = ['entail', 'contradict']
    else:
        raise Exception("corpus not found")

    return labels


def get_optimizer(s):
    """
    Parse optimizer parameters.
    Input should be of the form:
        - "sgd,lr=0.01"
        - "adagrad,lr=0.1,lr_decay=0.05"
    """
    if "," in s:
        method = s[:s.find(',')]
        optim_params = {}
        for x in s[s.find(',') + 1:].split(','):
            split = x.split('=')
            assert len(split) == 2
            assert re.match("^[+-]?(\d+(\.\d*)?|\.\d+)$", split[1]) is not None
            optim_params[split[0]] = float(split[1])
    else:
        method = s
        optim_params = {}

    if method == 'adadelta':
        optim_fn = optim.Adadelta
    elif method == 'adagrad':
        optim_fn = optim.Adagrad
    elif method == 'adam':
        optim_fn = optim.Adam
    elif method == 'adamax':
        optim_fn = optim.Adamax
    elif method == 'asgd':
        optim_fn = optim.ASGD
    elif method == 'rmsprop':
        optim_fn = optim.RMSprop
    elif method == 'rprop':
        optim_fn = optim.Rprop
    elif method == 'sgd':
        optim_fn = optim.SGD
        assert 'lr' in optim_params
    else:
        raise Exception('Unknown optimization method: "%s"' % method)

    # check that we give good parameters to the optimizer
    expected_args = inspect.getargspec(optim_fn.__init__)[0]
    assert expected_args[:2] == ['self', 'params']
    if not all(k in expected_args[2:] for k in optim_params.keys()):
        raise Exception('Unexpected parameters: expected "%s", got "%s"' % (
            str(expected_args[2:]), str(optim_params.keys())))

    return optim_fn, optim_params
