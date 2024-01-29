import nltk
import math
import re
import numpy as np


def process(query, porcessing=0):
    if isinstance(query, list):
        for i in range(len(query)):
            query[i] = process(query[i], porcessing)
        return query
    else:
        if porcessing == 0:
            lancaster = nltk.LancasterStemmer()
            # termesNormalise = [lancaster.stem(terme) for terme in termes]
            query = lancaster.stem(query)
        else:
            porter = nltk.PorterStemmer()
            query = porter.stem(query)
        return query


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
        doc_name = "Descripteur"
    else:
        doc_name = "Inverse"
    if split == 0:
        # append to the string the word Split
        doc_name = doc_name + "Split"
    else:
        doc_name = doc_name + "Token"
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


def get_W_V(query_list, processing=0, split=0):
    query_list = process(query_list, processing)
    file_name = get_file_name(processing, split, doc=0)
    voc = read_file(file_name)

    V = []
    W = {}
    for item in voc:
        if item[1] in query_list:
            V.append(1)
        else:
            V.append(0)
        if item[0] not in W.keys():
            W[item[0]] = []
    for key in W.keys():
        for item in voc:
            if item[0] == key:
                W[item[0]].append(float(item[3]))
            else:
                W[key].append(0)
    return W, V


# Documents AND NOT ranking OR queries OR GPT-3.5


def scalar_product(query_list, processing=0, split=0):
    query_list = process(query_list, processing)
    W, V = get_W_V(query_list, processing, split)
    n = len(V)
    RSV = {}
    for key in W.keys():
        RSV[key] = 0
    for key in W.keys():
        for i in range(n):
            RSV[key] += V[i] * W[key][i]
    print("original", RSV)
    return RSV
    # print(voc)


# def scalar_product2(query_list, processing=0, split=0):
#     query_list = process(query_list, processing)
#     file_name = get_file_name(processing, split, doc=0)
#     voc = np.array(read_file(file_name))
#     unique_values = np.unique([sub_array[0] for sub_array in voc])
#     unique_values = unique_values.tolist()
#     for unique in unique_values:
#         matching_sub_arrays = voc[voc[:, 0] == unique]
#         print(matching_sub_arrays)
#     print(unique_values)
def scalar_product2(query_list, processing=0, split=0):
    query_list = process(query_list, processing)
    file_name = get_file_name(processing, split, doc=0)
    voc = np.array(read_file(file_name))
    mask = np.isin(voc[:, 1], query_list)
    unique_values, sums = np.unique(voc[mask][:, 0].astype(int), return_inverse=True)
    weights = voc[mask][:, -1].astype(float)
    result = np.bincount(sums, weights)
    RSV = dict(zip(unique_values, result))
    return RSV


# scalar_product2(["document", "ranking"], 0, 0)
# scalar_product(["document", "ranking"], 0, 0)


def cosine(query_list, processing=0, split=0):
    query_list = process(query_list, processing)
    W, V = get_W_V(query_list, processing, split)
    n = len(V)
    RSV = {}
    for key in W.keys():
        RSV[key] = 0
    for key in W.keys():
        V_squared_sum = 0
        W_squared_sum = 0
        for i in range(n):
            RSV[key] += V[i] * W[key][i]
            V_squared_sum += math.pow(V[i], 2)
            W_squared_sum += math.pow(W[key][i], 2)
        try:
            RSV[key] = RSV[key] / (math.sqrt(V_squared_sum) * math.sqrt(W_squared_sum))
        except ZeroDivisionError:
            RSV[key] = 0
    print("original", RSV)
    return RSV
    # print(voc


def cosine2(query_list, processing=0, split=0):
    query_list = query_list.split(" ")
    query_list = process(query_list, processing)
    # convert to an array
    file_name = get_file_name(processing, split, doc=0)
    voc = np.array(read_file(file_name))
    # print(voc)
    mask = np.isin(voc[:, 1], query_list)
    voc = voc[mask]
    mask = np.isin(voc[:, 1], query_list)
    unique_values, sums = np.unique(voc[mask][:, 0].astype(int), return_inverse=True)
    weights = voc[mask][:, -1].astype(float)
    result = np.bincount(sums, weights)
    weights = np.sqrt(np.sum(weights**2))
    vi = np.sqrt(len(query_list))
    RSV = dict(zip(unique_values, result / (vi * weights)))
    # sort the RSV
    RSV = dict(sorted(RSV.items(), key=lambda item: item[1], reverse=True))
    return RSV

    # return RSV


cosine2("document ranking", 0, 0)
cosine(["document", "ranking"], 0, 0)


def jaccard(query_list, processing=0, split=0):
    query_list = process(query_list, processing)
    W, V = get_W_V(query_list, processing, split)
    n = len(V)
    RSV = {}
    for key in W.keys():
        RSV[key] = 0
    for key in W.keys():
        V_squared_sum = 0
        W_squared_sum = 0
        nom = 0
        for i in range(n):
            nom += V[i] * W[key][i]
            V_squared_sum += math.pow(V[i], 2)
            W_squared_sum += math.pow(W[key][i], 2)
        RSV[key] = nom / (V_squared_sum + W_squared_sum - nom)
    return RSV


# jaccard(["document", "ranking"], 0, 0)
# cosine(["document", "ranking"], 0, 0)


def freq(q, d):
    count = 0
    for item in d:
        if item == q:
            count += 1
    return count


def BM252(query_list, k, b, processing=0, split=0):
    query_list = process(query_list, processing)
    file_name = get_file_name(processing, split, doc=0)
    voc = np.array(read_file(file_name))
    unique_values, counts = np.unique(voc[:, 0].astype(int), return_counts=True)
    mask = np.isin(voc[:, 1], query_list)
    voc = voc[mask]
    terms, ni = np.unique(voc[:, 1], return_counts=True)
    ni = dict(zip(terms, ni))
    # print(voc[:, 2])
    # freqs keep the first and third column
    freqs = voc[:, [0, 2]].astype(int)
    counts = dict(zip(unique_values.astype(int), counts))
    N = len(counts.keys())
    RSV = {}
    for i in range(len(freqs)):
        dl = counts[freqs[i][0]]
        avdl = np.mean(list(counts.values()))
        d = k * (freqs[i][1] * (1 - b + b * (dl / avdl)))
        n = ni[voc[i][1]]
        p2 = np.log10((N - n + 0.5) / (n + 0.5))
        if freqs[i][0] not in RSV.keys():
            RSV[freqs[i][0]] = 0
        RSV[freqs[i][0]] = (freqs[i][1] / d) * p2

    # sort the RSV
    RSV = dict(sorted(RSV.items(), key=lambda item: item[1], reverse=True))
    print(RSV)

    return RSV


# BM252(["document", "ranking"], 1.5, 0.75, 0, 0)


def BM25(query_list, k, b, processing=0, split=0):
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
    print(RSV)
    return RSV


# BM25(["document", "ranking"], 1.5, 0.75, 0, 1)


def divide_expression(expresion):
    # seperate the expression by space
    expresion = expresion.split(" ")
    print(expresion)
    return expresion


def check_occurence(expression, doc):
    for x in doc:
        if x == expression:
            return True
    return False


def check_expression(expression):
    pattern = (
        r"^((NOT\s)?[^ \t\n\r\f\vANDOR]+(\s(AND|OR)\s)?(NOT\s)?[^ \t\n\r\f\vANDOR]+)*$"
    )
    print(bool(re.fullmatch(pattern, expression)))
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
        postfix = infix_to_postfix(exp)
        res2[r] = evaluate_postfix(postfix)
        exp = " ".join(exp)
        exp = eval(exp)
        if exp == True:
            res[r] = 1
        elif exp == False:
            res[r] = 0
    print(res)
    print(res2)
    return res, res2


# evaluate_expression2("Document AND NOT query", 0, 0)
# check_expression("document AND NOT query")
# divide_expression("x tt AND term OR term")
check_expression("x AND term OR term ")
check_expression("moh AND 9dor AND NOT @xx")
check_expression("NOT jj AND rm OR NOT trm")
# check_expression("NOT term AND NOT term")
# check_expression("term AND term NOT OR NOT term")
# check_expression("NOT term AND term OR NOT term NOT")
# check_expression("term term AND term")
# check_expression("NOT NOT term")
# check_expression("AND term")
# check_expression("AND")
# check_expression("OR")
# check_expression("NOT")
# check_expression("AND OR term")
# check_expression("term AND OR term")
# check_expression("term AND NOT")
# check_expression("NOT query AND document OR NOT")
# check_expression("term term term")


# search(query=3, processing=0, split=1)

# scalar_product(["query", "expansion", "QE"], 0, 0)
# cosine(["query", "expansion", "QE"], 0, 0)
# jaccard(["query", "expansion", "QE"], 0, 0)
