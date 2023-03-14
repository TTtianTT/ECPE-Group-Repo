# -*- coding: utf-8 -*-

import sys
reload(sys)
sys.setdefaultencoding('utf8')

# grab test cases
ch_dependency_patterns = {
  # and
  u"并且": [{"S2": "advmod", "S1": "conj", "POS": "AD"}],
  u"而且": [{"S2": "advmod", "S1": "conj", "POS": "AD"}],
  u"而": [{"S2": "advmod", "S1": "conj", "POS": "AD"}],
  u"因为": [{"S2": "case", "S1": "nmod:prep", "POS": "P"}],
  u"之前": [{"S2": "advmod", "S1": "conj", "POS": "AD"}],
  u"但是": [{"POS": "AD", "S1": "conj","S2": "advmod"}],
  u"可是": [{"POS": "AD", "S1": "conj","S2": "advmod"}],
  u"不过": [{"POS": "AD", "S1": "conj","S2": "advmod"}],
  u"但": [{"POS": "AD", "S1": "conj","S2": "advmod"}],
  u"如果": [{"POS": "CS", "S1": "dep","S2": "advmod"}],
  u"因此": [{"POS": "AD", "S1": "conj","S2": "advmod"}],
  u"所以": [{"POS": "AD", "S1": "conj","S2": "advmod"}],
  # "然后": [{"POS": "AD", "S1": "conj", "S2": "advmod"}, {"POS": "AD", "S1": "dep", "S2": "advmod"}],
  u"虽然": [{"POS": "CS", "S1": "conj","S2": "advmod"}],
  u"尽管": [{"POS": "CS", "S1": "conj","S2": "advmod"}],
  # u"当": [{
  #   "POS": "P",
  #   "S1": "nmod:prep",
  #   "S2": "case",
  #   "enclosing_marker": {
  #     "POS": "LC",
  #     "marker": "时",
  #     "dep_to_S": "case",
  #     "note": "this appears at the end of S2 and would make the classifier's job too easy (even tho it would be a well-formed sentence with the marker)"
  #   }
  # }],
  u"当时": [{"POS": "NT", "S1": "nmod:tmod", "S2": "conj"}],
}

sp_dependency_patterns = {
  "luego": [{"S1": "parataxis", "S2": "advmod", "POS": "ADV"}],
  "después": [{"S1": "advcl", "S2": "advmod", "POS": "ADV"}],
  "también": [{"S1": "parataxis", "S2": "advmod", "POS": "ADV"}],
  "además": [{"S1": "parataxis", "S2": "advmod", "POS": "ADV"}],
  "aunque": [{"S1": "advcl", "S2": "mark", "POS": "SCONJ"}],
  # "a pesar de que": [{"S1": "", "S2": "", "POS": ""}],

  "y": [{"S1": "conj", "S2": "cc", "POS": "CONJ", "flip": True}],
  "porque": [{"S1": "advcl", "S2": "mark", "POS": "SCONJ"}],

  "ya que": [{"S1": "advcl", "S2": "mark", "POS": "ADV", "head": "ya"}],
  # "dado que": [{"S1": "", "S2": "", "POS": "", "head": ""}],
  # "puesto que": [{"S1": "advcl", "S2": "mark", "POS": "nc0s000", "head": "puesto"}],
  "antes": [{"S1": "advcl", "S2": "advmod", "POS": "ADV"}],
  "pero": [{"S1": "conj", "S2": "cc", "POS": "CONJ", "flip": True}],
  "por ejemplo": [{"S1": "advcl", "S2": "advmod", "POS": "ADP", "head": "por"}],
  "sin embargo": [{"S1": "conj", "S2": "cc", "POS": "ADP", "head": "sin"}],
  "si": [{"S1": "advcl", "S2": "mark", "POS": "SCONJ"}],
  "mientras": [{"S1": "advcl", "S2": "mark", "POS": "SCONJ", "alternative": "mientras tanto"}],
  "entonces": [{"S1": "parataxis", "S2": "advmod", "POS": "ADV"}],

  # "por lo tanto": [{"S1": "parataxis", "S2": "nmod", "POS": "vmis000", "head": "por"}],
  # "por eso": [{"S1": "advcl", "S2": "nmod", "POS": "pd000000", "head": "por"}],
  # "por lo cual": [{"S1": "parataxis", "S2": "nmod", "POS": "pr000000", "head": "por"}],
  # "en consecuencia": [{"S1": "advcl", "S2": "nmod", "POS": "nc0s000", "head": "consecuencia"}],

  "todavía": [{"S1": "parataxis", "S2": "advmod", "POS": "ADV"}],
  "cuando": [{"S1": "advcl", "S2": "mark", "POS": "SCONJ"}]
}

en_dependency_patterns = {
  # S2 ~ S2 head (full S head) ---> connective
  "after": [
    {"S1": "advcl", "S2": "mark", "POS": "IN"},
    {"S1": "acl", "S2": "mark", "POS": "IN"},
  ],
  "also": [
    {"S1": "advcl", "S2": "advmod", "POS": "RB"},
  ],
  "although": [
    {"S1": "advcl", "S2": "mark", "POS": "IN"},
  ],
  "and": [
    {"S1": "conj", "S2": "cc", "POS": "CC", "flip": True},
  ],
  "as": [
    {"S1": "advcl", "S2": "mark", "POS": "IN"},
  ],
  "before": [
    {"S1": "advcl", "S2": "mark", "POS": "IN"},
  ],
  "but": [
    {"S1": "conj", "S2": "cc", "POS": "CC", "flip": True},
  ],
  "so": [
    # {"S1": "parataxis", "S2": "dep", "POS": "IN", "flip": True},
    {"S1": "advcl", "S2": "mark", "POS": "IN"},
    {"S1": "advcl", "S2": "mark", "POS": "RB"},
  ],
  "still": [
    {"S1": "parataxis", "S2": "advmod", "POS": "RB", "acceptable_order": "S1 S2"},
    {"S1": "dep", "S2": "advmod", "POS": "RB", "acceptable_order": "S1 S2"},
  ],
  "though": [
    {"S1": "advcl", "S2": "mark", "POS": "IN"},
  ],
  "because": [
    {"S1": "advcl", "S2": "mark", "POS": "IN"},
  ],
  "however": [
    {"S1": "parataxis", "S2": "advmod", "POS": "RB"},
    # {"S1": "ccomp", "S2": "advmod", "POS": "RB"}, ## rejecting in favor of high precision
  ],
  "if": [
    {"S1": "advcl", "S2": "mark", "POS": "IN"},
  ],
  "meanwhile": [
    {"S1": "parataxis", "S2": "advmod", "POS": "RB"},
  ],
  "while": [
    {"S1": "advcl", "S2": "mark", "POS": "IN"},
  ],
  "for example": [
    {"S1": "parataxis", "S2": "nmod", "POS": "NN", "head": "example"},
  ],
  "then": [
    {"S1": "parataxis", "S2": "advmod", "POS": "RB", "acceptable_order": "S1 S2"},
  ],
  "when": [
    {"S1": "advcl", "S2": "advmod", "POS": "WRB"},
  ],
}