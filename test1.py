__author__ = 'kevintandean'
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from random import shuffle
from itertools import combinations
from collections import Counter
from sklearn.ensemble import RandomForestClassifier


import time

def timeit(method):

    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()

        print '%r %2.2f sec' % \
              (method.__name__, te-ts)
        return result

    return timed

def split_training_file(filename, num):
    f = open(filename, 'r')
    list = []
    validation_set_positive = []
    validation_set_negative = []
    for line in f:
        line = line.split()
        if len(line) == 12:
            line = line[1:]
        list.append(line)
    shuffle(list)
    training_set = list[:]
    for index, item in enumerate(list):
        if item[-1] == '1' and len(validation_set_positive) < num:
            training_set[index] = 0
            validation_set_positive.append(item)
        elif item[-1] == '\xc2\xa11' and len(validation_set_negative) < num:
            training_set[index] = 0
            validation_set_negative.append(item)
    clean_training = []
    for i in training_set:
        if i != 0:
            clean_training.append(i)
    validation_set = validation_set_negative + validation_set_positive
    return clean_training, validation_set



def get_xy(list):
    X = []
    Y = []
    for i, readline in enumerate(list):
        if len(readline)==12:
            readline = readline[1:]
        x_list = []
        for index, item in enumerate(readline):
            if index == 0:
                pass
            elif index == 10:
                if item == '\xc2\xa11':
                    item = -1
                elif item == '1':
                    item = 1
                Y.append(item)
            else:
                if item[0]=='\xc2':
                    item = item[2:]
                    item = -1*float(item)
                x_list.append(float(item))
        X.append(x_list)

    return X,Y

def train_and_validate(training, validation, num):
    X,Y = get_xy(training)
    x_valid, y_valid = get_xy(validation)
    num_list = [1,2,3,4,5,6,7,8,9]
    combination = combinations(num_list, num)
    dict = {}
    for item in combination:
        new_X = []
        new_x_validation = []
        for row in X:
            new_row = []
            for i, col in enumerate(row):
                if i in item:
                    new_row.append(col)
            new_X.append(new_row)
        for row in x_valid:
            new_row = []
            for i, col in enumerate(row):
                if i in item:
                    new_row.append(col)
            new_x_validation.append(new_row)

        # clf = svm.SVC(C=5.0)
        clf = RandomForestClassifier()
        clf.fit(new_X,Y)
        score = clf.score(new_x_validation, y_valid)
        dict[item] = score
    max = 0
    key = 0

    for k in dict:
        if dict[k] > max:
            max = dict[k]
            key = k
    return {key: max}

@timeit
def compute():
    new_list = []
    key_list = []
    cnt = Counter()
    for i in range(0,50):
        training, validation = split_training_file('training_set', 25)
        new_list.append(train_and_validate(training, validation, 7))
    for item in new_list:
        for k in item:
            key_list.append(k)
    for item in key_list:
        cnt[item] += 1

    most_common = cnt.most_common(1)
    most_common = most_common[0][0]
    sum = 0
    n = 0
    for item in new_list:
        for k in item:
            if k == most_common:
                sum += item[k]
                n += 1
    average = float(sum/n)
    print most_common
    print n
    print average
    print '-------------'

@timeit
def main():
    for i in range(1,6):
        print i
        compute()

main()


