from __future__ import division
import re
import nltk
import sys

from nltk.collocations import BigramCollocationFinder #for finding most relevant bigrams
from nltk.metrics import BigramAssocMeasures

from nltk import stem
stemmer = stem.PorterStemmer()

from math import log, copysign
import json

def bigram_features_selector(words, n, chi=BigramAssocMeasures.chi_sq):
    bigram_finder = BigramCollocationFinder.from_words(words)
    bigr = bigram_finder.nbest(chi, n)
    return [x for x in bigr]
    
def getNMI(n11, n01, n10, n00, n1dot, n0dot):
    nmi  = (n11/n)*log(n*n11/(n1dot*ndot1))
    nmi += (n01/n)*log(n*n01/(n0dot*ndot1))
    nmi += (n10/n)*log(n*n10/(n1dot*ndot0))
    nmi += (n00/n)*log(n*n00/(n0dot*ndot0))
    return nmi
    
def getNMIparams(feature):
    n11 = 1
    n01 = 1
    n10 = 1
    n00 = 1
    n1dot = 0
    n0dot = 0
    for i,sentences in enumerate(x):
        if feature in sentences:
            if y[i] == 1: 
                n11 += 1 
            else: n10 += 1
        else:
            if y[i] == 1: 
                n01 += 1 
            else: n00 += 1
    n1dot = n11 + n10
    n0dot = n00 + n01
    return n11, n01, n10, n00, n1dot, n0dot
    
def tokenizer(words):
    words = words.replace('\'','').replace(',','').replace('.','')
    temp = re.split("\s|(?<!\d)[^\w']+|[^\w']+(?!\d)", words.lower())
    return [stemmer.stem(i) for i in temp]
    
  

poswords = []
negwords = []
fpos = open('positive-words.txt','r')
for i in fpos:
    poswords.append(stemmer.stem(i.rstrip()))
fpos.close()
fneg = open('negative-words.txt','r')
for i in fneg:
    negwords.append(stemmer.stem(i.rstrip()))
fneg.close()

train = open(sys.argv[1])
train.readline()
y = []
x = []
oneLongSentence = []
all_features = set()

for i in train:
    datarow =  i.split(',')
    y.append(int(datarow[0]))
    tokenized = tokenizer(datarow[1])
    all_features = all_features.union(set(tokenized))
    oneLongSentence.extend(tokenized)
    x.append(tokenized)

stop = set(['',' ','a','an','the','its','it','his'])
all_features = all_features.difference(stop)
sorted_features = []

n = len(y)
ndot1 = y.count(1) 
ndot0 = y.count(0) 

###############################FEATURE SELECTION#####################################################

for features in all_features:
    if isinstance(features, (tuple)) == False:
        n11, n01, n10, n00, n1dot, n0dot =  getNMIparams(features)
        sorted_features.append(( getNMI(n11, n01, n10, n00, n1dot, n0dot),features ))
sorted_features.sort(reverse=True)    



num_features = 2500
sorted_features = sorted_features[0:num_features]
sorted_features = [xx for x,xx in sorted_features]

    
#adding bigram features
num_bigram_features = 200
bigram_features = bigram_features_selector(oneLongSentence,num_bigram_features)
sorted_features.extend([' '.join(x) for x in bigram_features]) 

###############################GETTING TRAINING DATA TOGETHER#####################################################
train = open('train.csv')
train.readline()

x = [ [0 for i in range(num_features + num_bigram_features)] for j in range(n) ]
for i,data in enumerate(train):
    datarow =  data.split(',')
    y.append(int(datarow[0]))
    tokenized = tokenizer(datarow[1])
    for token in tokenized:
        if token in sorted_features:
            x[i][sorted_features.index(token)] = 1
    for j in range(len(tokenized)-1):
        token = ' '.join((tokenized[j], tokenized[j+1]))
        if token in sorted_features:
            x[i][sorted_features.index(token)] = 1

########################training NAIVE BAYES###################################
prob = [ [0 for i in range(2)] for j in range(num_features + num_bigram_features) ]
y_pred = [0 for i in range(len(y))]
for i in range(num_features + num_bigram_features):
    pos = 0
    neg = 0
    for j in range(n):
        if y[j] == 1 and x[j][i] == 1:
            pos += 1
        if y[j] == 0 and x[j][i] == 1:
            neg += 1
    prob[i][1] = (pos + 1) / (y.count(1) + num_features + num_bigram_features)  
    prob[i][0] = (neg + 1) / (y.count(0) + num_features + num_bigram_features) 
    if sorted_features[i].encode('cp1252') in poswords:
        prob[i][1] = 2 * prob[i][1]
    if sorted_features[i].encode('cp1252') in negwords:
        prob[i][0] = 2 * prob[i][0]
        

###cross validation code####################################################
# nCV = 6397 - n 
# yCV = []
# xCV = [ [0 for i in range(num_features + num_bigram_features)] for j in range(nCV) ]
# train = open('train.csv')
# train.readline()
# for i,data in enumerate(train):
    # if i >= n
        # datarow =  data.split(',')
        # yCV.append(int(datarow[0]))
        # tokenized = tokenizer(datarow[1])
        # for token in tokenized:
            # if token in sorted_features:
                # xCV[i-n][sorted_features.index(token)] = 1

# y_predCV = [0 for i in range(6397 - n)]
# for i in range(6397 - n):
    # poslikhood = 0
    # neglikhood = 0
    # for j in range(num_features + num_bigram_features):
        # if xCV[i][j] == 1:
            # poslikhood += log(prob[j][1])
            # neglikhood += log(prob[j][0])
        # if xCV[i][j] == 0:
            # poslikhood += log( 1 - prob[j][1] )
            # neglikhood += log( 1 - prob[j][0] )
            
        # if poslikhood > neglikhood:
            # y_predCV[i] = 1
        # else:
            # y_predCV[i] = 0
# error = 0
# for i,_ in enumerate(y_predCV):
    # if y[i] != y_predCV[i]:
        # error += 1
# print "Number of mislabeled cross validation points :", error/n
################################################################################

#####################################SAVING MODEL#################################
with open(sys.argv[2],  'wb') as fm:
    json.dump([prob,sorted_features], fm) 
#################################################################################
