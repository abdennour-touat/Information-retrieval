import nltk
import math
import numpy as np


# read a txt file
def token(data):
    ExpReg = nltk.RegexpTokenizer(
        # "(?:[A-Za-z]\.)+|[A-Za-z]+[\-@]\d+(?:\.\d+)?|\d+[A-Za-z]+|\d+(?:[\.\,]\d+)?%?|\w+(?:[\-/]\w+)*"
        r"(?:[A-Za-z]\.)+|[A-Za-z]+[\-@]\d+(?:\.\d+)?|\d+[A-Za-z]+|\d+(?:[\.\,]\d+)?%?|\w+(?:[\-/]\w+)*"

    )

    return ExpReg.tokenize(data)


# print(Termes)
def motsVide(termes):
    motsVide = nltk.corpus.stopwords.words("english")
    termesSansMotVide = [terme for terme in termes if terme.lower() not in motsVide]
    return termesSansMotVide


def splitting(data):
    # using split method
    return data.split(" ")


def porterStem(termes):
    porter = nltk.PorterStemmer()
    termesNormalise = [porter.stem(terme) for terme in termes]
    return termesNormalise


def lancaster(termes):
    lancaster = nltk.LancasterStemmer()
    termesNormalise = [lancaster.stem(terme) for terme in termes]
    return termesNormalise



# def fullIndexing(termesList, porter=False, split=False):
#     termesFrequent = {}
#     N = len(termesList)
#     for i in range(len(termesList)):
#         j = i + 1
#         if split:
#             txt = splitting(termesList[i])
#         else:
#             txt = token(termesList[i])
#         ntxt = motsVide(txt)
#         if porter:
#             ntxt = porterStem(ntxt)
#         else:
#             ntxt = lancaster(ntxt)
#         for terme in ntxt:
#             if (terme, j) in termesFrequent.keys():
#                 termesFrequent[(terme, j)] += 1
#             else:
#                 termesFrequent[(terme, j)] = 1
#     # calculating the weights
#     termes_frequent_avec_poids = {}
#     for item in termesFrequent:
#         freq = termesFrequent[item]
#         ni = 0
#         docs = []
#         # docs.append(item[1])
#         maxFreq = termesFrequent[item]
#         for jtem in termesFrequent:
#             if jtem[0] == item[0] and jtem[1] not in docs:
#                 docs.append(jtem[1])
#                 ni += 1
#             if maxFreq < termesFrequent[jtem] and jtem[1] == item[1]:
#                 maxFreq = termesFrequent[jtem]
#         poid = (freq / maxFreq) * math.log10((N / ni) + 1)
#         # poid = poid * np.log10((N / ni) + 1)
#         # poids = poid * math.log10((N / ni) + 1)
#         termes_frequent_avec_poids[item] = (freq, poid)
#     return termes_frequent_avec_poids
# def fullIndexing(termesList, porter=False, split=False):
#     allTerms = []
#     allFreqs = []
#     N = len(termesList)
#     uniqueTerms = []
#     allWeights = [0] * N
#     for i in range(len(termesList)):
#         if split:
#             txt = splitting(termesList[i])
#         else:
#             txt = token(termesList[i])
#         ntxt = motsVide(txt)
#         if porter:
#             ntxt = porterStem(ntxt)
#         else:
#             ntxt = lancaster(ntxt)

#         # append to the termesFrequent dict
#         terms = []
#         freqs = []
#         visited = []
#         for terme in ntxt:
#             if terme in visited:
#                 continue
#             if terme not in uniqueTerms:
#                 uniqueTerms.append(terme)
#             # append the term with the frequency and the doc number
#             terms.append(terme)
#             freqs.append(ntxt.count(terme))
#             visited.append(terme)
#         allTerms.append(np.array(terms))
#         allFreqs.append(np.array(freqs))
#         allWeights[i] = [0] * len(terms)
#         # calculating the weights
#     # print(len(uniqueTerms))
#     # print(N)
#     for term in uniqueTerms:
#         # get the number of docs that contains the term
#         # ni = np.count_nonzero(term in lists)
#         ni = np.sum(np.array([term in terms for terms in allTerms]))
#         for i, terms in enumerate(allTerms):
#             # get the index of the term in the terms list
#             freqs = allFreqs[i]
#             index = np.where(terms == term)
#             if len(index[0]) == 0:
#                 continue
#             for ind in index[0]:
#                 maxFreq = max(freqs)
#                 poid = (freqs[ind] / maxFreq) * math.log10((N / ni) + 1)
#                 allWeights[i][ind] = poid
#     print(allTerms, allWeights, allFreqs)
#     return allTerms, allWeights, allFreqs
from collections import Counter
# def fullIndexing(termesList, porter=False, split=False):
#     N = len(termesList)
#     allWeights = [0] * N
#     allTerms = []
#     allFreqs = []
#     for docid, sublist in enumerate(termesList):
#         if split:
#             txt = splitting(sublist)
#         else:
#             txt = token(sublist)
#         ntxt = motsVide(txt)
#         if porter:
#             ntxt = porterStem(ntxt)
#         else:
#             ntxt = lancaster(ntxt)
#         term_freqs = Counter(ntxt)
#         max_freq = max(term_freqs.values())
#         terms = list(term_freqs.keys())
#         freqs = list(term_freqs.values())
#         allTerms.append(np.array(terms))
#         allFreqs.append(np.array(freqs))
#         allWeights[docid] = [0] * len(terms)
#     allTerms_flat = np.concatenate(allTerms)
#     print(len(allTerms))
#     print(N)
#     for i, sublist in enumerate(allTerms):
#         ni = np.array([np.sum(allTerms_flat == term) for term in sublist])
#         freqs = allFreqs[i]
#         max_freq = np.max(freqs)
#         allWeights[i] = (freqs / max_freq) * np.log10((N / ni) + 1)

#     return allTerms, allWeights, allFreqs
from collections import defaultdict
import math
import numpy as np

def fullIndexing(termesList, porter=False, split=False):
    N = len(termesList)
    allWeights = [0] * N
    allTerms = []
    allFreqs = []
    term_to_doc_freqs = defaultdict(set)

    for docid, sublist in enumerate(termesList):
        if split:
            txt = splitting(sublist)
        else:
            txt = token(sublist)
        ntxt = motsVide(txt)
        if porter:
            ntxt = porterStem(ntxt)
        else:
            ntxt = lancaster(ntxt)
        term_freqs = Counter(ntxt)
        terms = list(term_freqs.keys())
        freqs = list(term_freqs.values())
        allTerms.append(np.array(terms))
        allFreqs.append(np.array(freqs))
        allWeights[docid] = np.zeros(len(terms))
        for term in terms:
            term_to_doc_freqs[term].add(docid)
    for i, terms in enumerate(allTerms):
        ni = np.array([len(term_to_doc_freqs[term]) for term in terms])
        freqs = allFreqs[i]
        max_freq = max(freqs)
        x = N/ni
        x.round(4)
        denom = np.log10(x + 1)
        y = freqs/max_freq
        y.round(4)
        allWeights[i] = y * denom

    return allTerms, allWeights, allFreqs
dataArray = []
for i in range(1,6005):
    with open("docs/D"+str(i)+".txt", "r") as file:
        d = file.read().replace("\n", "")
        dataArray.append(d)
print("done")
# print(len(dataArray))
fullIndexedData = fullIndexing(dataArray)


# print(fullIndexedData)
# write in file
def write_file(name, dataArray, inversed=False, porter=False, split=False):
    allTerms, allWeights, allFreqs = fullIndexing(dataArray, porter=porter, split=split)
    with open("inverse5/Inverse" + name + ".txt", "w") as file:
        # write each item in the dictionary in a line
        for c, terms in enumerate(allTerms):
            for i, term in enumerate(terms):
                file.write(
                    str(term)
                    + " "
                    + str(c+1)
                    + " "
                    + str(allFreqs[c][i])
                    + " "
                    + str(allWeights[c][i])
                    + "\n"
                )
    with open("descripteur5/Descripteur" + name + ".txt", "w") as file:
        # write each item in the dictionary in a line
        for c, terms in enumerate(allTerms):
            for i, term in enumerate(terms):
                file.write(
                    str(c+1)
                    + " "
                    + str(term)
                    + " "
                    + str(allFreqs[c][i])
                    + " "
                    + str(allWeights[c][i])
                    + "\n"
                )



write_file("TokenLancaster", dataArray, inversed=False, porter=False, split=False)
print("done")
write_file("TokenPorter", dataArray, inversed=False, porter=True, split=False)
print("done")
write_file("SplitPorter", dataArray, inversed=False, porter=True, split=True)
print("done")
write_file("SplitLancaster", dataArray, inversed=False, porter=False, split=True)
