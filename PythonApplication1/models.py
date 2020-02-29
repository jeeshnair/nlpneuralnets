# models.py
import time
import random
import math
import torch
from torch import nn
import embeddings as E
from sentiment_data import *
from collections import Counter, defaultdict
import numpy as np

class FeatureExtractor:

    def extract_features(self, ex_words: List[str]) -> List[int]:
        raise NotImplementedError()

    def feature_size(self):
        raise NotImplementedError()

class UnigramFeatureExtractor(FeatureExtractor):
    """
    Extracts unigram bag-of-words features from a sentence. It's up to you to decide how you want to handle counts
    and any additional preprocessing you want to do.
    """
    def __init__(self, vocabulary):
        self.vocabulary = vocabulary

    def feature_size(self):
        return len(self.vocabulary)

    def extract_features(self, ex_words):
        """
        Q1: Implement the unigram feature extractor.
        Hint: You may want to use the Counter class.
        """
        ex_words_lower = [x.lower() for x in ex_words]
        # feature of 1 for bias weight.
        features = [1]
        for key in self.vocabulary:
            if(key in ex_words_lower):
                features.append(1)
            else:
                features.append(0)
        return features


class BetterFeatureExtractor(FeatureExtractor):
    def __init__(self, vocabulary , bigrams):
        self.bigrams = bigrams
        self.vocabulary = vocabulary
        pass
    """
    Better feature extractor...try whatever you can think of!
    """
    def extract_features(self, ex_words):
        """
        Q3: Implement the unigram feature extractor.
        Hint: You may want to use the Counter class.
        """
        ex_words_lower = [x.lower() for x in ex_words]
        features = [1]

        for key in self.vocabulary:
            if(key in ex_words_lower):
                features.append(1)
            else:
                features.append(0)

        bigramFeatures = dict()
        for key in self.bigrams:
           for i in range(len(ex_words_lower)):
                if(i == 0):
                    bigramKey = ("<s>",ex_words_lower[i])
                else:
                    bigramKey = (ex_words_lower[i - 1], ex_words_lower[i])
                if (key == bigramKey):
                    bigramFeatures[key] = 1
                    break
                
        for  key in self.bigrams:
            features.append( bigramFeatures.get(key,0))

        return features;

    def feature_size(self):
        return len(self.bigrams) + len(self.vocabulary)

class SentimentClassifier(object):

    def featurize(self, ex):
        raise NotImplementedError()

    def forward(self, feat):
        raise NotImplementedError()

    def extract_pred(self, output):
        raise NotImplementedError()

    def update_parameters(self, output, feat, ex, lr):
        raise NotImplementedError()

    def run_train(self, train_data: List[SentimentExample], dev_data: List[SentimentExample], lr=1e-3, epoch=10):
        """
        Training loop.
        """
        train_data = train_data[:]
        for ep in range(epoch):
            start = time.time()
            random.shuffle(train_data)

            if isinstance(self, nn.Module):
                self.train()

            acc = []
            i = 0
            for ex in train_data:
                feat = self.featurize(ex)
                output = self.forward(feat)
                self.update_parameters(output, feat, ex, lr)
                predicted = self.extract_pred(output)
                acc.append(predicted == ex.label)
                i = i + 1
            acc = sum(acc) / len(acc)

            if isinstance(self, nn.Module):
                self.eval()

            dev_acc = []
            for ex in dev_data:
                feat = self.featurize(ex)
                output = self.forward(feat)
                predicted = self.extract_pred(output)
                dev_acc.append(predicted == ex.label)
            dev_acc = sum(dev_acc) / len(dev_acc)
            print('epoch {}: train acc = {}, dev acc = {}, time = {}'.format(ep, acc, dev_acc, time.time() - start))

    def predict(self, ex: SentimentExample) -> int:
        feat = self.featurize(ex)
        output = self.forward(feat)
        return self.extract_pred(output)


class TrivialSentimentClassifier(SentimentClassifier):
    """
    Sentiment classifier that always predicts the positive class.
    """
    def predict(self, ex: SentimentExample) -> int:
        return 1

    def run_train(self, train_data: List[SentimentExample], dev_data: List[SentimentExample], lr=None, epoch=None):
        pass


class PerceptronClassifier(SentimentClassifier):
    """
    Q1: Implement the perceptron classifier.
    """

    def __init__(self, feat_extractor):
        self.feat_extractor = feat_extractor
        # Add and extra for bias weight
        self.weights = np.zeros(feat_extractor.feature_size() + 1)

    def featurize(self, ex):
        """
        Converts an example into features.
        """
        return self.feat_extractor.extract_features(ex.words)

    def forward(self, feat) -> float:
        # compute the activation of the perceptron
        return np.dot(self.weights,feat)

    def extract_pred(self, output) -> int:
        # compute the prediction of the perceptron given the activation
        #ReLU
        return 1.0 if output >= 0.0 else 0.0

    def update_parameters(self, output, feat, ex, lr):
        # update the weight of the perceptron given its activation, the input
        # features, the example, and the learning rate
        prediction = self.extract_pred(output)
        error = ex.label - prediction
        weightadjustment = np.multiply(lr, feat)
        if(ex.label == 1 and prediction == 0):
            self.weights = self.weights + weightadjustment
        elif(ex.label == 0 and prediction == 1):
            self.weights = self.weights - weightadjustment

class FNNClassifier(SentimentClassifier, nn.Module):
    """
    Q4: Implement the multi-layer perceptron classifier.
    """

    def __init__(self, args):
        super().__init__()
        self.glove = E.GloveEmbedding('wikipedia_gigaword', 300, default='zero')
        self.lossfunction = torch.nn.BCELoss()
        ### Start of your code
        self.fullyConnectedOne = torch.nn.Sequential(torch.nn.Linear(300, 100),
            torch.nn.Tanh())

        self.outputLayer = torch.nn.Sequential(torch.nn.Linear(100, 1),
            torch.nn.Sigmoid())

        # do not touch this line below
        self.optim = torch.optim.Adam(self.parameters(), args.learning_rate)

    def featurize(self, ex):
        # You do not need to change this function
        # return a [T x D] tensor where each row i contains the D-dimensional
        # embedding for the ith word out of T words
        embs = [self.glove.emb(w.lower()) for w in ex.words]
        return torch.Tensor(embs)

    def forward(self, feat) -> torch.Tensor:
        # compute the activation of the FNN
        feat = torch.sum(feat, dim=0)
        feat = feat.unsqueeze(0)
        out = self.fullyConnectedOne(feat)
        out = self.outputLayer(out)

        return out

    def extract_pred(self, output) -> int:
        # compute the prediction of the FNN given the activation
        if(output >= 0.5):
            return 1
        else:
            return 0

    def update_parameters(self, output, feat, ex, lr):
        # update the weight of the perceptron given its activation, the input
        # features, the example, and the learning rate
        target = torch.Tensor([[ex.label]]) 
        loss = self.lossfunction(output,target)
        loss.backward()
        self.optim.step()
        self.optim.zero_grad()


class RNNClassifier(FNNClassifier):

    """
    Q5: Implement the RNN classifier.
    """

    def __init__(self, args):
        super().__init__(args)
        # Start of your code

        self.lstm = nn.LSTM(300, 20, bidirectional=True, batch_first=True)
        self.outputLayer = torch.nn.Sequential(torch.nn.Linear(40, 1),
            torch.nn.Sigmoid())

        # End of your code
        self.optim = torch.optim.Adam(self.parameters(), args.learning_rate)

    def forward(self, feat):
        feat = feat.unsqueeze(0)
        (out, states) = self.lstm(feat)
        (out, indices) = torch.max(out, dim=1)
        out = self.outputLayer(out)

        return out



class MyNNClassifier(FNNClassifier):

    """
    Q6: Implement the your own classifier.
    """

    def __init__(self, args):
        super().__init__(args)
        # Start of your code
        self.conv1 = nn.Sequential(nn.Conv2d(1, 1, (1, 300)), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(1, 1, (2, 300)), nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(1, 1, (3, 300)), nn.ReLU())
        self.conv4 = nn.Sequential(nn.Conv2d(1, 1, (4, 300)), nn.ReLU())
        self.conv5 = nn.Sequential(nn.Conv2d(1, 1, (5, 300)), nn.ReLU())
        self.dropout = nn.Dropout(0.1)
        self.outputLayer = torch.nn.Sequential(torch.nn.Linear(740, 1),
            torch.nn.Sigmoid())

        # End of your code
        self.optim = torch.optim.Adam(self.parameters(), args.learning_rate)

        self.lossFunction = torch.nn.MSELoss(reduction='sum')
        self.optim = torch.optim.SGD(self.parameters(), lr=args.learning_rate, momentum=0.1)

    def featurize(self, ex):
        # You do not need to change this function
        # return a [T x D] tensor where each row i contains the D-dimensional
        if(len(ex.words) < 150):
            for i in range(150 - len(ex.words)):
                ex.words.append("<pad>")
        # embedding for the ith word out of T words
        embs = [self.glove.emb(w.lower()) for w in ex.words]
        return torch.Tensor(embs)

    def forward(self, feat):
        feat = feat.unsqueeze(0)
        feat = feat.unsqueeze(0)
        out1 = self.conv1(feat)
        out2 = self.conv2(feat)
        out3 = self.conv3(feat)
        out4 = self.conv4(feat)
        out5 = self.conv5(feat)
        concatenated = torch.cat([out1,out2,out3,out4,out5],2)
        extracted = concatenated.squeeze(0)
        extracted = extracted.squeeze(0)
        extracted = extracted.squeeze(1)
        extracted = extracted.unsqueeze(0)
        extracted = self.dropout(extracted)
        final = self.outputLayer(extracted)
        
        return final
        
def train_model(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample]) -> SentimentClassifier:
    """
    Main entry point for your modifications. Trains and returns one of several models depending on the args
    passed in from the main method. You don't need to change this.
    """
    # Initialize feature extractor
    if args.model == "TRIVIAL":
        feat_extractor = None
    elif args.feats == "UNIGRAM":
        # Add additional preprocessing code here
        wordcount = dict()
        vocabulary = dict()
        for train_ex in train_exs:
            for word in train_ex.words:
                wordcount[word] = wordcount.get(word,0) + 1
        for key in wordcount:
            if(wordcount[key] > 1):
                 vocabulary[key.lower()] = wordcount[key]
        feat_extractor = UnigramFeatureExtractor(vocabulary)
    elif args.feats == "BETTER":
        wordcount = dict()
        vocabulary = dict()
        for train_ex in train_exs:
            for word in train_ex.words:
                wordcount[word] = wordcount.get(word,0) + 1
        for key in wordcount:
            if(wordcount[key] > 1):
                 vocabulary[key.lower()] = wordcount[key]

        # Add additional preprocessing code here
        bigram = dict()
        finalBigrams = dict()
        for train_ex in train_exs:
            for i in range(len(train_ex.words)):
                if(i == 0):
                    bigram[("<s>",train_ex.words[i].lower())] = bigram.get(("<s>",train_ex.words[i].lower()),0) + 1
                else:
                    bigram[(train_ex.words[i - 1].lower(), train_ex.words[i].lower())] = bigram.get((train_ex.words[i - 1].lower(), train_ex.words[i].lower()),0) + 1
        for key in bigram:
            if(bigram[key]>2):
                finalBigrams[key] = bigram[key]
        feat_extractor = BetterFeatureExtractor(vocabulary, finalBigrams)
    else:
        raise Exception("Pass in UNIGRAM, or BETTER to run the appropriate system")

    # Train the model
    if args.model == "TRIVIAL":
        model = TrivialSentimentClassifier()
    elif args.model == "PERCEPTRON":
        model = PerceptronClassifier(feat_extractor)
    elif args.model == "FNN":
        model = FNNClassifier(args)
    elif args.model == 'RNN':
        model = RNNClassifier(args)
    elif args.model == 'MyNN':
        model = MyNNClassifier(args)
    else:
        raise NotImplementedError()

    model.run_train(train_exs, dev_exs, lr=args.learning_rate, epoch=args.epoch)
    if args.model == "PERCEPTRON":
        # Write weights to file
        f = open("weights.txt", "w")
        f.write('\n'.join(str(elem) for elem in model.weights))
        f.close()
        # Write vocabulary to file
        f = open("vocabulary.txt", "w")
        f.write('\n'.join([str(elem) for elem in feat_extractor.vocabulary]) )
        f.close()
    return model
