import nltk
import math
import re
import numpy as np
import nltk

def motsVide(termes):
    motsVide = nltk.corpus.stopwords.words("english")

    termesSansMotVide = [terme for terme in termes if terme.lower() not in motsVide]
    return termesSansMotVide

def read_judgments():
    judgments = []
    with open('LISA.REL', 'r') as file:
        judgments = {}
        for line in file:
            if line.startswith('Query'):
                queryid = line.split()[-1]
                line = next(file)
                line = next(file)
                #skip the next line
                numbers = line.split()
                    #convert to int
                numbers = [int(i) for i in numbers]
                numbers_without_minus_one = numbers[:-1]    
                judgments[int(queryid)] = numbers_without_minus_one
                    # judgments.append(numbers_without_minus_one)
    return judgments 
def read_queries():
    queries = []
    with open('Queries_1.txt', 'r') as file:
        for line in file:
            queries.append(line[:-1])
    print(len(queries))
    return queries
def process(query, porcessing=0):
    if porcessing == 0:
        lancaster = nltk.LancasterStemmer()
        new_query = [lancaster.stem(word) for word in query]
    else:
        porter = nltk.PorterStemmer()
        new_query = [porter.stem(word) for word in query]
    return new_query 


# read the files
def read_file(name):
    with open(name, "r") as file:
        data = []
        # read line by line
        for line in file:
            line = line.rstrip("\n")
            data.append(line.split(" "))
        return data


def get_file_name(processing=0, split=0, doc=0):
    if doc == 0:
        doc_name = "descripteur5/Descripteur"
    else:
        doc_name = "inverse/Inverse"
    if split == 0:
        # append to the string the word Split
        doc_name = doc_name + "Token"
    else:
        doc_name = doc_name + "Split"
    if processing == 0:
        doc_name = doc_name + "Lancaster"
    else:
        doc_name = doc_name + "Porter"
    doc_name = doc_name + ".txt"
    return doc_name


def find_by_term(query, data):
    # search the data with the query
    result = []
    for item in data:
        # check if the query is equal or similar to the item
        if query == item[0]:
            result.append(item)
    return result


def find_by_doc(query, data):
    # search the data with the query
    result = []
    for item in data:
        if int(item[0]) == int(query):
            result.append(item)
    return result


def search(query, processing=0, split=0, doc=0):
    if doc == 0:
        file_name = get_file_name(processing, split, doc)
        data = read_file(file_name)
        result = find_by_doc(query, data)
    else:
        query = process(query, processing)
        file_name = get_file_name(processing, split, doc=1)
        data = read_file(file_name)
        result = find_by_term(query, data)
    return result


def token(data):
    ExpReg = nltk.RegexpTokenizer(
        "(?:[A-Za-z]\.)+|[A-Za-z]+[\-@]\d+(?:\.\d+)?|\d+[A-Za-z]+|\d+(?:[\.\,]\d+)?%?|\w+(?:[\-/]\w+)*"
        # r"(?:[A-Za-z]\.)+|[A-Za-z]+[\-@]\d+(?:\.\d+)?|\d+[A-Za-z]+|\d+(?:[\.\,]\d+)?%?|\w+(?:[\-/]\w+)*"

    )

    return ExpReg.tokenize(data)

def splitting(data):
    # using split method
    return data.split()
def scalar_product(query_list, processing=0, split=0):
    empty_words = nltk.corpus.stopwords.words("english")
    query_list = [x.lower() for x in query_list.split() if x.lower() not in empty_words]
    query_list = " ".join(query_list)
    if split == 0:
        query_list = token(query_list)
    else:  
        query_list = splitting(query_list)
    query_list = process(query_list, processing)
    #convert to an array
    file_name = get_file_name(processing, split, doc=0)
    voc = np.array(read_file(file_name))
    mask = np.isin(voc[:, 1], query_list)
    unique_values, sums = np.unique(voc[mask][:, 0].astype(int), return_inverse=True)
    weights = voc[mask][:, -1].astype(float)
    result = np.bincount(sums, weights)
    #sort the array descending
    RSV = dict(zip(unique_values, result))
    #sort the rsv
    RSV = dict(sorted(RSV.items(), key=lambda item: item[1], reverse=True))
    #eliminate the zeros
    RSV = {k: v for k, v in RSV.items() if v != 0}
    return RSV
def cosine(query_list, processing=0, split=0):
    if split == 0:
        query_list = token(query_list)
    else:  
        query_list = splitting(query_list)
    query_list = list(filter(None, query_list))
    query_list = [x.lower() for x in query_list]
    query_list = process(query_list, processing)
    # convert to an array
    file_name = get_file_name(processing, split, doc=0)
    voc = np.array(read_file(file_name))
    # print(voc)
    mask = np.isin(voc[:, 1], query_list)
    unique_values, sums = np.unique(voc[mask][:, 0].astype(int), return_inverse=True)
    weights = voc[mask][:, -1].astype(float)
    all_weights = voc[:, -1].astype(float)
    result = np.bincount(sums, weights)
    weights = np.sqrt(np.sum(all_weights**2))
    vi = np.sqrt(len(query_list))
    RSV = dict(zip(unique_values, result / (vi * weights)))
    # sort the RSV
    RSV = dict(sorted(RSV.items(), key=lambda item: item[1], reverse=True))
    RSV = {k: v for k, v in RSV.items() if v != 0}
    return RSV
def jaccard(query_list, processing=0, split=0):
    if split == 0:
        query_list = token(query_list)
    else:  
        query_list = splitting(query_list)
    query_list = list(filter(None, query_list))
    query_list = [x.lower() for x in query_list]
    query_list = process(query_list, processing)
    # convert to an array
    file_name = get_file_name(processing, split, doc=0)
    voc = np.array(read_file(file_name))
    # print(voc)
    mask = np.isin(voc[:, 1], query_list)
    voc = voc[mask]
    unique_values, sums = np.unique(voc[:, 0].astype(int), return_inverse=True)
    weights = voc[:, -1].astype(float)
    nominator = np.bincount(sums, weights)
    denominator = len(query_list) + np.bincount(sums, weights**2) - nominator
    result = nominator / denominator
    result[np.isnan(result)] = 0
    RSV = dict(zip(unique_values, result))
    # sort the rsv
    RSV = dict(sorted(RSV.items(), key=lambda item: item[1], reverse=True))
    RSV = {k: v for k, v in RSV.items() if v != 0}
    return RSV




def freq(q, d):
    count = 0
    for item in d:
        if item == q:
            count += 1
    return count


def BM25(query_list, k, b, processing=0, split=0):
    if split == 0:
        query_list = token(query_list)
    else:  
        query_list = splitting(query_list)
    query_list = list(filter(None, query_list))
    query_list = [x.lower() for x in query_list]
    query_list = process(query_list, processing)
    file_name = get_file_name(processing, split, doc=0)
    voc = read_file(file_name)
    docs = {}
    for item in voc:
        docs[(int(item[0]), item[1])] = item[2:]
    # qs = {}
    dl = {}
    ni = {}
    freqs = {}
    for term in query_list:
        ni[term] = 0
        for doc in docs.keys():
            dl[doc[0]] = 0
            if term == doc[1]:
                freqs[(doc[0], term)] = int(docs[(doc[0], term)][0])
                ni[term] += 1
    RSV = {}
    for doc in docs.keys():
        if doc[0] not in RSV.keys():
            RSV[doc[0]] = 0
        # (docid,term)
        dl[doc[0]] += int(docs[(doc[0], doc[1])][0])

    avdl = 0
    for doc in dl.keys():
        avdl += dl[doc]

    N = len(dl.keys())
    avdl = avdl / N
    for comb in freqs.keys():
        nom = freqs[comb]
        p1 = (1 - b) + (b * (dl[comb[0]] / avdl))
        deno = (k * p1) + nom
        p2 = math.log10((N - ni[comb[1]] + 0.5) / (ni[comb[1]] + 0.5))
        RSV[comb[0]] += ((nom) / (deno)) * p2
    # print(query_list)
    # print(voc)
    #sort the rsv
    RSV = dict(sorted(RSV.items(), key=lambda item: item[1], reverse=True))
    RSV = {k: v for k, v in RSV.items() if v != 0}
    return RSV



# BM25(["document", "ranking"], 1.5, 0.75, 0, 1)


def divide_expression(expresion):
    # seperate the expression by space
    expresion = expresion.split(" ")
    return expresion


def check_occurence(expression, doc):
    for x in doc:
        # print(x)
        if x == expression:
            print("here ",x)
            return True
    return False


def check_expression(expression):
    terms = expression.split()
    nterms = []
    for x in terms :
        if x != "AND" and x != "OR" and x != "NOT":
            print
            nterms.append(x.lower())
        else:
            nterms.append(x)
    expression = " ".join(nterms)
    pattern = (
        r"^((NOT\s)?[^ \t\n\r\f\vANDOR]+(\s(AND|OR)\s)?(NOT\s)?[^ \t\n\r\f\vANDOR]+)*$"
    )
    return bool(re.fullmatch(pattern, expression))


def evaluate_expression(expression, processing, split):
    file_name = get_file_name(processing, split, doc=0)
    voc = read_file(file_name)
    docs = {}
    res = {}
    for item in voc:
        res[int(item[0])] = 0
        if int(item[0]) not in docs.keys():
            docs[int(item[0])] = []
        docs[int(item[0])].append(item[1])
    # seperate the expression by space
    expression = divide_expression(expression)
    # make the expression in lower case but not the operators
    for i in range(len(expression)):
        if expression[i] != "AND" and expression[i] != "OR" and expression[i] != "NOT":
            expression[i] = process(expression[i], processing)
            expression[i] = expression[i].lower()
            # process the expression
    if not check_expression(" ".join(expression)):
        return "error"
    # replace the terms with the binary value
    for r in res.keys():
        exp = expression.copy()
        for i, x in enumerate(exp):
            if x == "AND":
                exp[i] = "and"
            elif x == "OR":
                exp[i] = "or"
            elif x == "NOT":
                exp[i] = "not"
            else:
                exp[i] = str(check_occurence(x, docs[r]))
        exp = " ".join(exp)
        res[r] = eval(exp)
    return res


def infix_to_postfix(expression):
    precedence = {"not": 3, "and": 2, "or": 1}
    stack = []
    postfix = []

    for token in expression:
        if token not in ["and", "or", "not"]:
            postfix.append(token)
        else:
            while stack and precedence.get(token, 0) <= precedence.get(stack[-1], 0):
                postfix.append(stack.pop())
            stack.append(token)

    while stack:
        postfix.append(stack.pop())

    return postfix


def evaluate_postfix(expression):
    stack = []
    for token in expression:
        if token in ["True", "False"]:
            stack.append(token == "True")
        else:
            if token == "not":
                operand = stack.pop()
                result = not operand
            else:
                operand2 = stack.pop()
                operand1 = stack.pop()
                if token == "and":
                    result = operand1 and operand2
                elif token == "or":
                    result = operand1 or operand2
            stack.append(result)

    return stack[0]


def evaluate_expression2(expression, processing, split):
    file_name = get_file_name(processing, split, doc=0)
    voc = read_file(file_name)
    docs = {}
    res = {}
    res2 = {}
    for item in voc:
        res[int(item[0])] = 0
        res2[int(item[0])] = 0
        if int(item[0]) not in docs.keys():
            docs[int(item[0])] = []
        docs[int(item[0])].append(item[1])
    # seperate the expression by space
    expression = divide_expression(expression)
    # make the expression in lower case but not the operators
    nexpression = []
    for exp in expression:
        if exp != "AND" and exp != "OR" and exp != "NOT":
            nexpression.append(process(exp.lower(), processing))
            # process the expression
        else:
            nexpression.append(exp)
    if not check_expression(" ".join(expression)):
        return "error"
    # replace the terms with the binary value
    for r in res.keys():
        exp = expression.copy()
        for i, x in enumerate(exp):
            if x == "AND":
                exp[i] = "and"
            elif x == "OR":
                exp[i] = "or"
            elif x == "NOT":
                exp[i] = "not"
            else:
                exp[i] = str(check_occurence(x.lower(), docs[r]))
        postfix = infix_to_postfix(exp)
        res2[r] = evaluate_postfix(postfix)
        exp = " ".join(exp)
        exp = eval(exp)
        if exp == True:
            res[r] = 1
        elif exp == False:
            res[r] = 0
    #keep only nonzero values
    res = {k: v for k, v in res.items() if v != 0}
            
    return res, res2

