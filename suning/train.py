import numpy as np
import os
import scipy.sparse as sp
from skmultilearn.problem_transform import ClassifierChain
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection._split import train_test_split
from skmultilearn.neurofuzzy import MLARAM
from skmultilearn.problem_transform import BinaryRelevance
from skmultilearn.problem_transform import LabelPowerset
from sklearn.svm import SVC
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.metrics import accuracy_score
from skmultilearn.adapt import MLkNN
from sklearn.metrics import hamming_loss
from sklearn.metrics import f1_score, average_precision_score,precision_score,recall_score
from skmultilearn.ensemble.rakelo import RakelO

base_path = os.path.join("","../suning_data/label.txt")

def get_labels():
    num_of_labels = 41
    labels = []
    with open(base_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            temp = str(line).strip('\n').split(" ")
            #print(temp)
            s = np.zeros(num_of_labels)

            for a in temp:
                s[int(a)-1] = 1
            #print(s)
            labels.append(s)

    labels = np.array(labels)
    print(labels.shape)
    return labels

def get_vec():
    # corpus = np.load("./preprocess/vector500.txt.npy")
    # print(np.asarray(corpus).shape)
    # return np.asarray(corpus)
    X_data = []
    with open("./preprocess/mean_word2vec1.txt", encoding="utf-8") as f:
        for i, line in enumerate(f):
            temp = str(line).strip("\n").split(" ")
            vec = [float(a) for a in temp[:-1]]
            X_data.append(vec)
    X_data = np.array(X_data)
    print(X_data.shape)
    return X_data

def get_train():
    corpus = get_vec()
    labels = get_labels()

    X_train,X_test, y_train, y_test = train_test_split(corpus, labels, test_size=0.2)

    return X_train, y_train, X_test, y_test
    # size_of_examples = corpus.shape[0]
    # num = int(size_of_examples*0.8)
    # X_train = corpus[:num+1,:]
    # X_test = corpus[num:,:]
    # y_train = labels[:num+1, :]
    # y_test = labels[num:,:]
    # return sp.csr_matrix(X_train),sp.csr_matrix(y_train),sp.csr_matrix(X_test),sp.csr_matrix(y_test)





class Multi_Learning:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test= y_test

    def binary_relevance(self):
        '''Name: Binary Relevance
           Main Idea: Divide multi-classify into multi binary classfier
           Evaluation Metric: accuracy_score
        '''
        print(self.X_train)
        print(self.y_train)
        classifier = BinaryRelevance(GaussianNB())
        classifier.fit(self.X_train, self.y_train)

        predictions = classifier.predict(self.X_test)
        print(predictions)
        #print(y_test)
        #print("predictions:\n",predictions)

        result = accuracy_score(self.y_test, predictions)

        print(result)

    def classifer_chain(self):


        # initialize classifier chains multi-label classifier
        # with a gaussian naive bayes base classifier
        print("build classifier...")
        classifier = ClassifierChain(RandomForestClassifier())
        #classifier = LabelPowerset(RandomForestClassifier())
        print("end...")

        print("start training...")
        classifier.fit(self.X_train, self.y_train)
        print("end...")

        # predict
        print("start test...")
        predictions = classifier.predict(self.X_test)
        print("end...")

        print("result as following:")

        result = hamming_loss(self.y_test, predictions)
        print("hanming_loss: ", result)

        print("accuracy score: ", accuracy_score(y_test, predictions))

        result = f1_score(self.y_test, predictions, average='micro')
        print("micro-f1_score: ", result)

    def labelSet(self):
        classifier = LabelPowerset(GaussianNB())

        classifier.fit(self.X_train, self.y_train)

        # predict
        predictions = classifier.predict(self.X_test)
        result = accuracy_score(self.y_test, predictions)
        print(result)

    def mlknn(self, number):
        classifier = MLkNN(k=number)

        classifier.fit(self.X_train, self.y_train)

        # predict
        predictions = classifier.predict(self.X_test)
        result = hamming_loss(self.y_test, predictions)

        print("hanming_loss,",result)

        result = f1_score(self.y_test, predictions, average='micro')
        print("micro -f1: ", result)

        result = precision_score(self.y_test, predictions,average='micro')
        print(result)

    def k_random_labelSet(self,k):

        classifier = RakelO(classifier=RandomForestClassifier(), model_count=k, labelset_size=self.y_train.shape[1])
        # classifier.generate_partition(X_train,y_train)
        print("start training...")
        classifier.fit(self.X_train, self.y_train)
        print("end...")

        # predict
        print("start test...")
        predictions = classifier.predict(self.X_test)
        print("end...")

        print("result as following:")

        result = hamming_loss(self.y_test, predictions)
        print("hanming_loss: ", result)

        print("accuracy score: ", accuracy_score(y_test, predictions))

        result = f1_score(self.y_test, predictions, average='micro')
        print("micro-f1_score: ", result)

    def MLARAM(self):

        # initialize classifier chains multi-label classifier
        # with a gaussian naive bayes base classifier
        print("build classifier...")
        classifier = MLARAM()
        print("end...")

        print("start training...")
        classifier.fit(self.X_train, self.y_train)
        print("end...")

        # predict
        print("start test...")
        predictions = classifier.predict(self.X_test)
        print("end...")

        print("result as following:")

        result = hamming_loss(self.y_test, predictions)
        print("hanming_loss: ", result)

        result = f1_score(self.y_test, predictions, average='micro')
        print("micro-f1_score: ", result)


if __name__=="__main__":
    X_train, y_train, X_test, y_test = get_train()
    test = Multi_Learning(X_train, y_train, X_test, y_test)
    #test.binary_relevance()
    #test.classifer_chain()
    #test.MLARAM()
    #test.labelSet()
    #test.mlknn(number=20)
    test.k_random_labelSet(k=50)
