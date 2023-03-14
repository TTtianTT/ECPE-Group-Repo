"""
Load in InferSent, SkipThought, and other sentence embedding models
and compare their performance interactively
"""

import IPython
import logging
import sys
import os
import torch
import tensorflow as tf
import scipy.spatial.distance as distance
from baselines import AverageEmbedder

# ======= SkipThought Config ========
PATH_TO_SKIPTHOUGHT = '/home/anie/models/research/skip_thoughts'
sys.path.insert(0, PATH_TO_SKIPTHOUGHT)

from skip_thoughts import configuration
from skip_thoughts import encoder_manager

logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

VOCAB_FILE = "/home/anie/skip_thoughts/pretrained/skip_thoughts_bi_2017_02_16/vocab.txt"
EMBEDDING_MATRIX_FILE = "/home/anie/skip_thoughts/pretrained/skip_thoughts_bi_2017_02_16/embeddings.npy"
CHECKPOINT_PATH = "/home/anie/skip_thoughts/pretrained/skip_thoughts_bi_2017_02_16/model.ckpt-500008"

# ======= InferSent Config ========
GLOVE_PATH = '/home/anie/glove/glove.840B.300d.txt'
MODEL_PATH = 'infersent.allnli.pickle'

assert os.path.isfile(MODEL_PATH) and os.path.isfile(GLOVE_PATH), \
    'Set MODEL and GloVe PATHs'

# ======== Can add more models ========
# ...

# ======== Set up individual sentence testing =======

def _compute_sim(batched_emb):
    # (2, hidden-dim)
    sent1 = batched_emb[0]
    sent2 = batched_emb[1]

    # can add more
    cos_dis = distance.cosine(sent1, sent2)  # pearson correlation and cosine similarities are similar
    euc_dis = distance.euclidean(sent1, sent2)

    return [cos_dis, euc_dis], ['cosine', 'euclidean']


def _generate_log(model_name, metrics, names):
    log_str = model_name + " distance - "
    for m, n in zip(metrics, names):
        log_str += "{}: {} ".format(n, m)

    return log_str


def compute_for_sent(sent1, sent2):
    # this is only for a pair of sentences, not batch testing, not default
    infersent.build_vocab([sent1, sent2], tokenize=True)
    inf_emb = infersent.encode([sent1, sent2], bsize=2, tokenize=True)
    inf_sims, sim_names = _compute_sim(inf_emb)

    # SkipThought is tokenize-by-default, so no worries
    st_emb = skipthought.encode([sent1, sent2], verbose=False, use_eos=True)
    st_sims, _ = _compute_sim(st_emb)

    # Average word embedding similarity
    avg_emb.build_vocab([sent1, sent2], tokenize=True)
    ae_emb = avg_emb.encode([sent1, sent2], bsize=2, tokenize=True)
    ae_sims, _ = _compute_sim(ae_emb)

    # traditional measurement like levenstein distance, dynamic time wrapping, jaro, etc.

    print(_generate_log("Average Embedding", ae_sims, sim_names))
    print(_generate_log("InferSent", inf_sims, sim_names))
    print(_generate_log("SkipThought", st_sims, sim_names))



if __name__ == '__main__':
    # Load in InferSent
    infersent = torch.load(MODEL_PATH)  # rely on "models.py" as well
    infersent.set_glove_path(GLOVE_PATH)

    # Load in SkipThought
    config_gpu = tf.ConfigProto()
    config_gpu.gpu_options.allow_growth = True

    with tf.Graph().as_default(), tf.Session(config=config_gpu) as session:
        skipthought = encoder_manager.EncoderManager()

        skipthought.load_model(configuration.model_config(bidirectional_encoder=True),
                               vocabulary_file=VOCAB_FILE,
                               embedding_matrix_file=EMBEDDING_MATRIX_FILE,
                               checkpoint_path=CHECKPOINT_PATH)

    # Load in average embedding
    avg_emb = AverageEmbedder(word_emb_dim=300)
    avg_emb.set_glove_path(GLOVE_PATH)

    IPython.embed()
