import numpy as np
from search import process, scalar_product


def calc_precision2(res):
    nbps = 0
    for i, item in enumerate(res):
        if res[item] != 0:
            nbps += 1
    return nbps / len(res)


def calc_precision(res, k):
    nbps = 0
    for i, item in enumerate(res):
        if res[item] != 0 and int(i) < k:
            nbps += 1
    return nbps / k


def calc_recall(res, judgement):
    dps = 0
    for i, item in enumerate(res):
        if res[item] != 0:
            dps += 1
    return dps / len(judgement)


def calc_fscore(res, judgement):
    p = calc_precision2(res)
    r = calc_recall(res, judgement)
    return 2 * p * r / (p + r)


# read the judgment file and the queries file
with open("Judgements", "r") as file:
    # read line by line
    lines = file.readlines()
    # split each line by space
    lines = [line.split() for line in lines]
    # convert to numpy array
    lines = np.array(lines)
    # convert to int
    lines = lines.astype(int)
    judgements = lines

# print(judgements)

with open("Queries", "r") as file:
    lines = file.readlines()
    lines = [line.split() for line in lines]
    queries = lines

# print(queries)


def rappel_precision(res, judgement):
    tdp = len(judgement)
    print("tdp", tdp)
    # rappel = [0.25, 0.25, 0.5, 0.5, 0.75, 1.0, 1.0, 1.0, 1.0, 1.0]
    rappel = []
    precision = []
    inc = 0
    cpt = 1
    for k in res.keys():
        if res[k] > 0:
            inc += 1
        precision.append(inc / cpt)
        rappel.append(inc / tdp)
        cpt += 1
    precision_interpole = []
    rapple_interpole = []
    # create a list of values of 0.1,0.2 up to 1
    for i in range(0, 11):
        rapple_interpole.append(i / 10)
    for i in range(0, 11):
        prec = []
        for j, x in enumerate(rappel):
            if x >= rapple_interpole[i]:
                prec.append(precision[j])
        precision_interpole.append(max(prec))
    return precision_interpole, rapple_interpole


# rappel_precision(queries, judgements)
