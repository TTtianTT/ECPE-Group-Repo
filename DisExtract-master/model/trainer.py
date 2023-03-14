# -*- coding: utf-8 -*-

"""
Code adapted from
https://github.com/facebookresearch/InferSent/blob/master/train_nli.py

with minor modifications
"""

import os
import sys
import csv
import time
import json
import argparse
from os.path import join as pjoin
from itertools import izip

import numpy as np

import torch
from torch.autograd import Variable
import torch.nn as nn

from data import get_dis, get_batch, build_vocab
from dissent import DisSent
from util import get_optimizer, get_labels

import logging

parser = argparse.ArgumentParser(description='NLI training')
# paths
parser.add_argument("--corpus", type=str, default='books_5', help="books_5|books_old_5|books_8|books_all|gw_cn_5|gw_cn_all|gw_es_5|dat")
parser.add_argument("--hypes", type=str, default='hypes/default.json', help="load in a hyperparameter file")
parser.add_argument("--outputdir", type=str, default='sandbox/', help="Output directory")
parser.add_argument("--outputmodelname", type=str, default='dis-model')

# training
parser.add_argument("--n_epochs", type=int, default=10)
parser.add_argument("--cur_epochs", type=int, default=1)
parser.add_argument("--cur_lr", type=float, default=0.1)
parser.add_argument("--cur_valid", type=float, default=-1e10, help="must set this otherwise resumed model will be saved by default")

parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--dpout_model", type=float, default=0., help="encoder dropout")
parser.add_argument("--dpout_emb", type=float, default=0., help="embedding dropout")
parser.add_argument("--dpout_fc", type=float, default=0., help="classifier dropout")
parser.add_argument("--optimizer", type=str, default="sgd,lr=0.1", help="adam or sgd,lr=0.1")
parser.add_argument("--lrshrink", type=float, default=5, help="shrink factor for sgd")
parser.add_argument("--decay", type=float, default=0.99, help="lr decay")
parser.add_argument("--minlr", type=float, default=1e-5, help="minimum lr")
parser.add_argument("--max_norm", type=float, default=5., help="max norm (grad clipping)")
parser.add_argument("--log_interval", type=int, default=100, help="how many batches to log once")

# model
parser.add_argument("--encoder_type", type=str, default='BLSTMEncoder', help="see list of encoders")
parser.add_argument("--enc_lstm_dim", type=int, default=2048, help="encoder nhid dimension")
parser.add_argument("--n_enc_layers", type=int, default=1, help="encoder num layers")
parser.add_argument("--fc_dim", type=int, default=512, help="nhid of fc layers")
parser.add_argument("--pool_type", type=str, default='max', help="max or mean")
parser.add_argument("--tied_weights", action='store_true', help="RNN would share weights on both directions")
parser.add_argument("--reload_val", action='store_true', help="Reload the previous best epoch on validation, should be used with tied weights")
parser.add_argument("--char", action='store_true', help="for Chinese we can train on char-level model")
parser.add_argument("--s1", action='store_true', help="training only on S1")
parser.add_argument("--s2", action='store_true', help="training only on S2")

# gpu
parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID")
parser.add_argument("--seed", type=int, default=1234, help="seed")

params, _ = parser.parse_known_args()

# set gpu device
torch.cuda.set_device(params.gpu_id)

"""
SEED
"""
np.random.seed(params.seed)
torch.manual_seed(params.seed)
torch.cuda.manual_seed(params.seed)

"""
Logging
"""

logging.basicConfig(format='[%(asctime)s] %(levelname)s: %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger(__name__)

if not os.path.exists(params.outputdir):
    os.makedirs(params.outputdir)
file_handler = logging.FileHandler("{0}/log.txt".format(params.outputdir))
logging.getLogger().addHandler(file_handler)

# print parameters passed, and all parameters
logger.info('\ntogrep : {0}\n'.format(sys.argv[1:]))
logger.info(params)

"""
Default json file loading
"""
with open(params.hypes, 'rb') as f:
    json_config = json.load(f)

data_dir = json_config['data_dir']
prefix = json_config[params.corpus]
glove_path = json_config['glove_path']

if params.char and params.corpus == "gw_cn_5":
    prefix = prefix.replace('discourse', 'discourse_char')

"""
DATA
"""
train, valid, test = get_dis(data_dir, prefix, params.corpus)
word_vec = build_vocab(train['s1'] + train['s2'] +
                       valid['s1'] + valid['s2'] +
                       test['s1'] + test['s2'], glove_path)

# unknown words instead of map to <unk>, this directly takes them out
for split in ['s1', 's2']:
    for data_type in ['train', 'valid', 'test']:
        eval(data_type)[split] = np.array([['<s>'] +
                                           [word for word in sent.split() if word in word_vec] +
                                           ['</s>'] for sent in eval(data_type)[split]])

params.word_emb_dim = 300

dis_labels = get_labels(params.corpus)
label_size = len(dis_labels)

"""
MODEL
"""
# model config
config_dis_model = {
    'n_words': len(word_vec),
    'word_emb_dim': params.word_emb_dim,
    'enc_lstm_dim': params.enc_lstm_dim,
    'n_enc_layers': params.n_enc_layers,
    'dpout_emb': params.dpout_emb,
    'dpout_model': params.dpout_model,
    'dpout_fc': params.dpout_fc,
    'fc_dim': params.fc_dim,
    'bsize': params.batch_size,
    'n_classes': label_size,
    'pool_type': params.pool_type,
    'encoder_type': params.encoder_type,
    'tied_weights': params.tied_weights,
    'use_cuda': True,
}

if params.cur_epochs == 1:
    dis_net = DisSent(config_dis_model)
    logger.info(dis_net)
else:
    # if starting epoch is not 1, we resume training
    # 1. load in model
    # 2. resume with the previous learning rate
    model_path = pjoin(params.outputdir, params.outputmodelname + ".pickle")  # this is the best model
    # this might have conflicts with gpu_idx...
    dis_net = torch.load(model_path)

# loss
loss_fn = nn.CrossEntropyLoss()
loss_fn.size_average = False

# optimizer
optim_fn, optim_params = get_optimizer(params.optimizer)
optimizer = optim_fn(dis_net.parameters(), **optim_params)

if params.cur_epochs != 1:
    optimizer.param_groups[0]['lr'] = params.cur_lr

# cuda by default
dis_net.cuda()
loss_fn.cuda()

"""
TRAIN
"""
val_acc_best = -1e10 if params.cur_epochs == 1 else params.cur_valid
adam_stop = False
stop_training = False
lr = optim_params['lr'] if 'sgd' in params.optimizer else None


def trainepoch(epoch):
    logger.info('\nTRAINING : Epoch ' + str(epoch))
    dis_net.train()
    all_costs = []
    logs = []
    words_count = 0

    last_time = time.time()
    correct = 0.
    # shuffle the data
    permutation = np.random.permutation(len(train['s1']))

    s1 = train['s1'][permutation]
    s2 = train['s2'][permutation]
    target = train['label'][permutation]

    optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * params.decay if epoch > 1 \
                                                                                        and 'sgd' in params.optimizer else \
        optimizer.param_groups[0]['lr']
    logger.info('Learning rate : {0}'.format(optimizer.param_groups[0]['lr']))

    for stidx in range(0, len(s1), params.batch_size):
        # prepare batch
        s1_batch, s1_len = get_batch(s1[stidx:stidx + params.batch_size],
                                     word_vec)
        s2_batch, s2_len = get_batch(s2[stidx:stidx + params.batch_size],
                                     word_vec)
        s1_batch, s2_batch = Variable(s1_batch.cuda()), Variable(s2_batch.cuda())
        tgt_batch = Variable(torch.LongTensor(target[stidx:stidx + params.batch_size])).cuda()
        k = s1_batch.size(1)  # actual batch size

        # model forward
        output = dis_net((s1_batch, s1_len), (s2_batch, s2_len))

        pred = output.data.max(1)[1]
        correct += pred.long().eq(tgt_batch.data.long()).cpu().sum()
        assert len(pred) == len(s1[stidx:stidx + params.batch_size])

        # loss
        loss = loss_fn(output, tgt_batch)
        all_costs.append(loss.data[0])
        words_count += (s1_batch.nelement() + s2_batch.nelement()) / params.word_emb_dim

        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient clipping (off by default)
        shrink_factor = 1
        total_norm = 0

        for p in dis_net.parameters():
            if p.requires_grad:
                p.grad.data.div_(k)  # divide by the actual batch size
                total_norm += p.grad.data.norm() ** 2
        total_norm = np.sqrt(total_norm)

        if total_norm > params.max_norm:
            shrink_factor = params.max_norm / total_norm
        current_lr = optimizer.param_groups[0]['lr']  # current lr (no external "lr", for adam)
        optimizer.param_groups[0]['lr'] = current_lr * shrink_factor  # just for update

        # optimizer step
        optimizer.step()
        optimizer.param_groups[0]['lr'] = current_lr

        if len(all_costs) == params.log_interval:
            logs.append('{0} ; loss {1} ; sentence/s {2} ; words/s {3} ; accuracy train : {4}'.format(
                stidx, round(np.mean(all_costs), 2),
                int(len(all_costs) * params.batch_size / (time.time() - last_time)),
                int(words_count * 1.0 / (time.time() - last_time)),
                round(100. * correct / (stidx + k), 2)))
            logger.info(logs[-1])
            last_time = time.time()
            words_count = 0
            all_costs = []
    train_acc = round(100 * correct / len(s1), 2)
    logger.info('results : epoch {0} ; mean accuracy train : {1}'
                .format(epoch, train_acc))
    return train_acc


def get_multiclass_recall(preds, y_label):
    # preds: (label_size), y_label; (label_size)
    label_cat = range(label_size)
    labels_accu = {}

    for la in label_cat:
        # for each label, we get the index of the correct labels
        idx_of_cat = y_label == la
        cat_preds = preds[idx_of_cat]
        if cat_preds.size != 0:
            accu = np.mean(cat_preds == la)
            labels_accu[la] = [accu]
        else:
            labels_accu[la] = []

    return labels_accu


def get_multiclass_prec(preds, y_label):
    label_cat = range(label_size)
    labels_accu = {}

    for la in label_cat:
        # for each label, we get the index of predictions
        idx_of_cat = preds == la
        cat_preds = y_label[idx_of_cat]  # ground truth
        if cat_preds.size != 0:
            accu = np.mean(cat_preds == la)
            labels_accu[la] = [accu]
        else:
            labels_accu[la] = []

    return labels_accu


def evaluate(epoch, eval_type='valid', final_eval=False, save_confusion=False):
    global dis_net

    dis_net.eval()
    correct = 0.
    global val_acc_best, lr, stop_training, adam_stop

    if eval_type == 'valid':
        logger.info('\nVALIDATION : Epoch {0}'.format(epoch))

    s1 = valid['s1'] if eval_type == 'valid' else test['s1']
    s2 = valid['s2'] if eval_type == 'valid' else test['s2']
    target = valid['label'] if eval_type == 'valid' else test['label']

    valid_preds, valid_labels = [], []

    for i in range(0, len(s1), params.batch_size):
        # prepare batch
        s1_batch, s1_len = get_batch(s1[i:i + params.batch_size], word_vec)
        s2_batch, s2_len = get_batch(s2[i:i + params.batch_size], word_vec)
        s1_batch, s2_batch = Variable(s1_batch.cuda()), Variable(s2_batch.cuda())
        tgt_batch = Variable(torch.LongTensor(target[i:i + params.batch_size])).cuda()

        # model forward
        output = dis_net((s1_batch, s1_len), (s2_batch, s2_len))

        pred = output.data.max(1)[1]
        correct += pred.long().eq(tgt_batch.data.long()).cpu().sum()

        # we collect samples
        labels = target[i:i + params.batch_size]
        preds = pred.cpu().numpy()

        valid_preds.extend(preds.tolist())
        valid_labels.extend(labels.tolist())

    mean_multi_recall = get_multiclass_recall(np.array(valid_preds), np.array(valid_labels))
    mean_multi_prec = get_multiclass_prec(np.array(valid_preds), np.array(valid_labels))

    multiclass_recall_msg = 'Multiclass Recall - '
    for k, v in mean_multi_recall.iteritems():
        multiclass_recall_msg += dis_labels[k] + ": " + str(v[0]) + " "

    multiclass_prec_msg = 'Multiclass Precision - '
    for k, v in mean_multi_prec.iteritems():
        if len(v) == 0:
            v = [0.]
        multiclass_prec_msg += dis_labels[k] + ": " + str(v[0]) + " "

    logger.info(multiclass_recall_msg)
    logger.info(multiclass_prec_msg)

    # if params.corpus == "gw_cn_5" or params.corpus == "gw_es_5":
    #     print(multiclass_recall_msg)
    #     print(multiclass_prec_msg)

    # save model
    eval_acc = round(100 * correct / len(s1), 2)
    if final_eval:
        logger.info('finalgrep : accuracy {0} : {1}'.format(eval_type, eval_acc))
    else:
        logger.info('togrep : results : epoch {0} ; mean accuracy {1} :\
              {2}'.format(epoch, eval_type, eval_acc))
        # print out multi-class recall and precision

    if save_confusion:
        with open(pjoin(params.outputdir, 'confusion_test.csv'), 'wb') as csvfile:
            fieldnames = ['preds', 'labels']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for pair in izip(valid_preds, valid_labels):
                writer.writerow({'preds': pair[0], 'labels': pair[1]})

    # there is no re-loading of previous best model btw
    if eval_type == 'valid' and epoch <= params.n_epochs:
        if eval_acc > val_acc_best:
            logger.info('saving model at epoch {0}'.format(epoch))
            if not os.path.exists(params.outputdir):
                os.makedirs(params.outputdir)
            torch.save(dis_net, os.path.join(params.outputdir,
                                             params.outputmodelname + ".pickle"))
            # monitor memory usage
            try:
                torch.save(dis_net, os.path.join(params.outputdir,
                                                 params.outputmodelname + "-" + str(epoch) + ".pickle"))
            except:
                print("saving by epoch error, maybe due to disk space limit")

            val_acc_best = eval_acc
        else:
            # can reload previous best model
            if 'sgd' in params.optimizer:
                optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] / params.lrshrink
                logger.info('Shrinking lr by : {0}. New lr = {1}'
                      .format(params.lrshrink,
                              optimizer.param_groups[0]['lr']))
                if optimizer.param_groups[0]['lr'] < params.minlr:
                    stop_training = True
            if 'adam' in params.optimizer:
                # early stopping (at 2nd decrease in accuracy)
                stop_training = adam_stop
                adam_stop = True

            # now we finished annealing, we can reload
            if params.reload_val:
                del dis_net
                dis_net = torch.load(os.path.join(params.outputdir, params.outputmodelname + ".pickle"))
                logger.info("Load in previous best epoch")
    return eval_acc


"""
Train model on Discourse Classification task
"""
if __name__ == '__main__':
    epoch = params.cur_epochs  # start at 1

    while not stop_training and epoch <= params.n_epochs:
        train_acc = trainepoch(epoch)
        eval_acc = evaluate(epoch, 'valid')
        epoch += 1

    # Run best model on test set.
    del dis_net
    dis_net = torch.load(os.path.join(params.outputdir, params.outputmodelname + ".pickle"))

    logger.info('\nTEST : Epoch {0}'.format(epoch))
    evaluate(1e6, 'valid', True)
    evaluate(0, 'test', True, True)  # save confusion results on test data

    # Save encoder instead of full model
    torch.save(dis_net.encoder,
               os.path.join(params.outputdir, params.outputmodelname + ".pickle" + '.encoder'))
