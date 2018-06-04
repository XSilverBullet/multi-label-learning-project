#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import gensim
import os
import collections
import smart_open
import random
import sys


def read_corpus(fname, tokens_only=False):

    with open(fname, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if tokens_only:
                yield gensim.utils.simple_preprocess(line)
            else:
                # For training data, add tags
                yield gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(line), [i])

path = os.path.join("","../../suning_data/data.txt")
sents = gensim.models.doc2vec.TaggedLineDocument(path)
#train_corpus = list(read_corpus(path))
#print(train_corpus[0])



print("start train doc2vec...")
model = gensim.models.doc2vec.Doc2Vec(sents, size=500, min_count=8)

# print("start build train_corpus...")
# model.build_vocab(sents)
# print("end build...")

model.train(sents, total_examples=149112, epochs=1)
print("end training...")

model.save("doc2vec500.txt")

corpus = model.docvecs
print(np.asarray(corpus))
#
np.save("vector500.txt", corpus)
# # model = gensim.models.doc2vec.Doc2Vec.load("doc2vec100.model")

# corpus = np.load("vector.txt.npy")
# print(np.asarray(corpus).shape)