## Note

This is the official repo that we used to train for DisSent paper. It is not yet cleaned up and not ready for official release.

You can download the trained models from the following AWS S3 link:
https://s3-us-west-2.amazonaws.com/dissent/ (Books 5, Books 8, and Books ALL)

You can also load the models by using the following script:
https://github.com/windweller/SentEval/blob/master/examples/dissent.py
https://github.com/windweller/SentEval/blob/master/examples/dissent_eval.py

Please contact anie@stanford.edu if you have problem using these scripts! Thank you!

We wrote the majority of the code in 2017 when PyTorch was still at version 0.1 and Python 2 was still popular. You might need to adjust your library versions in order to load in the model.

## Predicting Discourse Connectors

Thanks to Stepan Zakharov <stepanz@post.bgu.ac.il> who wrote some easy to use code for loading the complete model for the task of discourse connector prediction.

```python
import torch
from torch.autograd import Variable
model_path = 'C:/Users/vaule/PycharmProjects/Reddit/src/remote/dissent/dis-model.pickle'
GLOVE_PATH = 'C:/Users/vaule/PycharmProjects/Reddit/src/glove/glove.840B.300d.txt'
map_locations = torch.device('cpu')
dissent = torch.load(model_path, map_location=map_locations)
dissent.encoder.set_glove_path(GLOVE_PATH)

s1_raw = ['Hello there.']
s2_raw = ['How are you?']

dissent.encoder.build_vocab(s1_raw+s2_raw)
s1_prepared,s1_len = dissent.encoder.prepare_samples(s1_raw, tokenize=True, verbose=False, no_sort=True)
s2_prepared,s2_len = dissent.encoder.prepare_samples(s2_raw, tokenize=True, verbose=False, no_sort=True)

b1 = Variable(dissent.encoder.get_batch(s1_prepared, no_sort=True))
b2 = Variable(dissent.encoder.get_batch(s2_prepared, no_sort=True))
discourse_preds = dissent((b1, s1_len), (b2, s2_len))

print(discourse_preds)

out_proba = torch.nn.Softmax()(discourse_preds)

print('Output probabilities: ', [ '%.4f' % elem for elem in out_proba[0] ])
```

In this code, he loads the "DIS-ALL" model where we trained to predict 15 discourse markers. The list of markers can be found in: https://github.com/windweller/DisExtract/blob/master/preprocessing/cfg.py

## DisSent Corpus

The links to data is available under the same link: 

https://s3-us-west-2.amazonaws.com/dissent/ (Books 5, Books 8, Books ALL, Books ALL -- perfectly balanced)

If you scroll down, beneath the trained model pickle files, you can find download links for all our data.

We include all the training files (with the original train/valid/test split). We do not provide access to the original raw BookCorpus data at all.

Once you install [AWS-CLI](https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-install.html) (commandline tool from AWS), you can download using commands like below:

```
aws s3 cp s3://dissent/data/discourse_EN_ALL_and_then_because_though_still_after_when_while_but_also_as_so_although_before_if_2017dec21_test.tsv .
```

You can also browse files and folders using:

```
aws s3 ls s3://dissent/
    PRE books_5/
    PRE books_8/
    PRE books_all/
    PRE data/
```

An alternative way to download the data (since all of them are public), is to use `aws s3 ls s3://dissent/data/` to find the correct file name and use the following format:

```
wget https://dissent.s3-us-west-2.amazonaws.com/data/discourse_EN_FIVE_and_but_because_if_when_2017dec12_test.tsv
```

You can copy and paste your the file name in `https://dissent.s3-us-west-2.amazonaws.com/data/{FILE_NAME}`.

## Dataset Metadata

The following table is necessary for this dataset to be indexed by search
engines such as <a href="https://g.co/datasetsearch">Google Dataset Search</a>.
<div itemscope itemtype="http://schema.org/Dataset">
<table>
  <tr>
    <th>property</th>
    <th>value</th>
  </tr>
  <tr>
    <td>name</td>
    <td><code itemprop="name">Large Scale Discourse Marker Prediction Task</code></td>
  </tr>
  <tr>
    <td>alternateName</td>
    <td><code itemprop="alternateName">dissent</code></td>
  </tr>
  <tr>
    <td>url</td>
    <td><code itemprop="url">https://github.com/windweller/DisExtract</code></td>
  </tr>
  <tr>
    <td>description</td>
    <td><code itemprop="description">Learning effective representations of sentences is one of the core missions of natural language understanding. Existing models
either train on a vast amount of text, or require costly, manually curated sentence relation datasets. We show that with dependency
parsing and rule-based rubrics, we can curate
a high quality sentence relation task by leveraging explicit discourse relations. We show
that our curated dataset provides an excellent
signal for learning vector representations of
sentence meaning, representing relations that
can only be determined when the meanings
of two sentences are combined.</code></td>
  </tr>
  <tr>
    <td>provider</td>
    <td>
      <div itemscope itemtype="http://schema.org/Organization" itemprop="provider">
        <table>
          <tr>
            <th>property</th>
            <th>value</th>
          </tr>
          <tr>
            <td>name</td>
            <td><code itemprop="name">Stanford University</code></td>
          </tr>
          <tr>
            <td>sameAs</td>
            <td><code itemprop="sameAs">https://www.stanford.edu/</code></td>
          </tr>
        </table>
      </div>
    </td>
  </tr>
  <tr>
    <td>license</td>
    <td>
      <div itemscope itemtype="http://schema.org/CreativeWork" itemprop="license">
        <table>
          <tr>
            <th>property</th>
            <th>value</th>
          </tr>
          <tr>
            <td>name</td>
            <td><code itemprop="name">CC BY-SA 3.0</code></td>
          </tr>
          <tr>
            <td>url</td>
            <td><code itemprop="url">https://creativecommons.org/licenses/by-sa/3.0/</code></td>
          </tr>
        </table>
      </div>
    </td>
  </tr>
  <tr>
    <td>citation</td>
    <td><code itemprop="citation">Nie, A., Bennett, E. and Goodman, N., 2019, July. DisSent: Learning sentence representations from explicit discourse relations. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics (pp. 4497-4510).</code></td>
  </tr>
</table>
</div>

## PDTB 2 Caveat

In the main portion of our paper, we stated that we "use the same dataset split scheme for this task as for the implicit vs explicit task discussed above. Following Ji and Eisenstein (2015) and Qin et al. (2017)". This line caused confusion. In terms of dataset split scheme, we followed "Patterson and Kehler (2013)’s preprocessing. The dataset contains 25 sections in total. We use sections 0 and 1 as the development set, **sections 23 and 24** for the **test set**, and we train on the remaining sections 2-22.". This is made clear in the Appendix Section A.3.

As far as we know, Ji and Eisenstein (2015) and Qin et al. (2017) both used **sections 21-22** as the test set. It appears that they weren't aware the existence of **sections 23 and 24** at the time. In order to move the field forward, we highly encourage you to follow Patterson and Kehler (2013) processing scheme and use https://github.com/cgpotts/pdtb2 to process.

The code to extract sections 23 and 24 is the following:

```python
from pdtb2 import CorpusReader

corpus = CorpusReader('pdtb2.csv')

test_sents = []

for sent in corpus.iter_data():
    if sent.Relation == 'Implicit' and sent.Section in ['23','24']:
        if len(sent.ConnHeadSemClass1.split('.')) != 1:
            test_sents.append(sent)
```

Patterson, Gary, and Andrew Kehler. "Predicting the presence of discourse connectives." Proceedings of the 2013 Conference on Empirical Methods in Natural Language Processing. 2013. [link](https://www.aclweb.org/anthology/D13-1094)


## Dependency Pattern Instructions

```python
depdency_patterns = [
    "still": [
        {"POS": "RB", "S2": "advmod", "S1": "parataxis", "acceptable_order": "S1 S2"},
        {"POS": "RB", "S2": "advmod", "S1": "dep", "acceptable_order": "S1 S2"},
    ],
    "for example": [
        {"POS": "NN", "S2": "nmod", "S1": "parataxis", "head": "example"}
    ],
    "but": [
        {"POS": "CC", "S2": "cc", "S1": "conj", "flip": True}
    ],
]
  
```

`Acceptable_order: "S1 S2""`: a flag to reject weird discourse term like "then" referring to previous sentence, 
but in itself a S2 S1 situation. In general it's hard to generate a valid (S1, S2) pair.

`flip: True`: a flag that means rather than S2 connecting to marker, S1 to S2; it's S1 connecting to marker, 
then S1 connects to S2.

`head: "example""`: for two words, which one is the one we use to match dependency patterns. 

## Parsing performance

Out of 176 Wikitext-103 examples:
* accuracy = 0.81 (how many correct pairs or rejections overall?)
* precision = 0.89 (given that the parser returns a pair, was that pair correct?)

This excludes all "still" examples, because they were terrible.
