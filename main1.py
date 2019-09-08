
#%% imports
from nltk.classify.util import accuracy
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import stopwords
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from nltk.collections import defaultdict
import nltk
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm

from hazm import Normalizer, word_tokenize
import pandas as pd
import numpy as np

def clean_persianText(txt):
    normalizer = Normalizer()
    txt = normalizer.character_refinement(txt)
    txt = normalizer.affix_spacing(txt)
    txt = normalizer.punctuation_spacing(txt)
    txt = txt.replace('.', '')
    txt = normalizer.normalize(txt)
    return txt

def prepare_data(data):
    data = data[['text', 'label']]
    lbl = data['label']
    data['text'] = data['text'].apply(lambda x: clean_persianText(x))
    
    return data['text'], data['label']

def bag_of_words(words):
    return dict([(word, True) for word in words])

def bag_of_words_not_in_set(words, badwords):
    return bag_of_words(set(words)-set(badwords))


def bag_of_non_stopwords(words, stopfile="english"):
    badwords = stopwords.words(stopfile)
    return bag_of_words_not_in_set(words, badwords)


def bag_of_bigrams(words, score_fn=BigramAssocMeasures.chi_sq, n=200):
    bcf = BigramCollocationFinder.from_words(words)
    bigrams = bcf.nbest(score_fn, n)
    return bag_of_words(words+bigrams)


def label_feats_from_corpus(corp, feature_detector=bag_of_words):
    label_feats = defaultdict(list)
    for label in corp.categories():
        for fileid in corp.fileids(categories=[label]):
            feats = feature_detector(corp.words(fileids=[fileid]))
            label_feats[label].append(feats)
    return label_feats

def load_data(file_name='./dataset/fa_2.xlsx'):
    data, labels = prepare_data(pd.read_excel(file_name))
    unique_labels = np.unique(labels)
    data_new = list([bag_of_words(word_tokenize(d)) for d in data])
    lfeats = dict()
    for label in unique_labels:
        idx = np.where(labels == label)[0]
        data_c = data[idx]
        lfeats[label] = list([bag_of_words(word_tokenize(d)) for d in data_c])
    return lfeats

lfeats = load_data()

#%%

def split_label_feats(lfeats, split=0.80):
    train_feats = []
    test_feats = []
    for label, feats in lfeats.items():
        cutoff = int(len(feats)*split)
        train_feats.extend([(feat, label) for feat in feats[:cutoff]])
        test_feats.extend([(feat, label) for feat in feats[cutoff:]])
    return train_feats, test_feats


train_feats, test_feats = split_label_feats(lfeats)
train_feats[0]

print(len(train_feats))
print(len(test_feats))
#%%
# ______________  Bayesian  _______________-

nb = NaiveBayesClassifier.train(train_feats)
nb.labels()
acc = accuracy(nb, test_feats)
print("Bayesian Accuracy: ", acc)

#%%
# ______________  Naive_Bayes  _______________-

sk = SklearnClassifier(MultinomialNB())
sk.train(train_feats)
acc_Naive_Bayes = accuracy(sk, test_feats)
print("Naive Bayes Accuracy: ", acc_Naive_Bayes)

# ______________  K-Neighbors  _______________-

sk_knn = SklearnClassifier(KNeighborsClassifier())
sk_knn.train(train_feats)
acc_knn = accuracy(sk_knn, test_feats)
print("K-NN Accuracy: ", acc_knn)

# ______________  Regression  _______________-
#%%
sk_reg = SklearnClassifier(LogisticRegression())
sk_reg.train(train_feats)
acc_reg=accuracy(sk_reg, test_feats)
print("Regression Accuracy: ", acc_reg)


# ______________  SVM  _______________-
#%%
sk_svm = SklearnClassifier(svm())
sk_svm.train(train_feats)
acc_svm=accuracy(sk_svm, test_feats)
print("SVM Accuracy: ", acc_svm)

#%%
