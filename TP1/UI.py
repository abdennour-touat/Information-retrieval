import streamlit as st
import pandas as pd
import numpy as np
import search as sh
import tp8 as tp8
import plotly.graph_objects as go

# st.set_page_config(layout="wide")


def read_file(name):
    # read the judgment file and the queries file
    with open(name, "r") as file:
        # read line by line
        lines = file.readlines()
        # split each line by space
        lines = [line.split() for line in lines]
        # convert to numpy array
        lines = np.array(lines)
        # convert to int
        lines = lines.astype(int)
        judgements = lines
    return judgements


def main():
    pd.set_option("display.max_columns", None)
    with open("Queries", "r") as file:
        # read line by line
        lines = file.readlines()
        # split each line by space
        lines = [line.split() for line in lines]
        queries = lines
    column1, column2 = st.columns(2)
    query = column1.text_input("Query")
    datasetnum = column2.number_input("Dataset", value=1, min_value=1)
    dataset_checkbox = column2.checkbox("Query dataset")
    # query = list(filter(None, query))

    column1, column2, column3 = st.columns(3)
    tokenization = column1.checkbox("Tokenization")
    if tokenization:
        token = 0
    else:
        token = 1
    porter = column1.checkbox("Porter stemmer")
    if porter:
        port = 1
    else:
        port = 0

    matching = column2.radio(
        "Matching",
        (
            "",
            "Vectorial space model",
            "Probabilistic model",
            "Boolean model",
            "Data mining",
        ),
    )
    if query != "":
        if matching == "Vectorial space model":
            formula = column3.radio(
                "Formula",
                (
                    "Scalar product",
                    "Cosine similarity",
                    "Jaccard similarity",
                ),
            )
            if formula == "Scalar product":
                res = sh.scalar_product(query, port, token)
                res = pd.DataFrame.from_dict(res, orient="index")
                st.write(res)
            elif formula == "Cosine similarity":
                res = pd.DataFrame.from_dict(
                    sh.cosine(query, port, token), orient="index"
                )
                st.write(res)
            elif formula == "Jaccard similarity":
                res = pd.DataFrame.from_dict(
                    sh.jaccard(query, port, token), orient="index"
                )
                st.write(res)
        elif matching == "Probabilistic model":
            k = column3.number_input("k", value=1.5)
            b = column3.number_input("b", value=0.75)
            res = sh.BM25(query, k, b, port, token)
            res = pd.DataFrame.from_dict(res, orient="index")
            st.write(res)
        elif matching == "Boolean model":
            if sh.check_expression(query):
                res, res2 = sh.evaluate_expression2(query, port, token)
                res = pd.DataFrame.from_dict(res, orient="index")
                st.write(res)
            else:
                st.write("### Invalid expression")
        elif matching == "Data mining":
            pass
        elif matching == "":
            index = column2.radio("Index", ("Docs per Term", "Term per Doc"))
            if query != "":
                if index == "Docs per Term":
                    res = pd.DataFrame(sh.search(query, port, token, doc=1))
                    st.write(res)
                elif index == "Term per Doc":
                    res = pd.DataFrame(sh.search(query, port, token, doc=0))
                    st.write(res)
    if dataset_checkbox:
        judgements = read_file("Judgements")
        # get all the subarrays that have the same dataset number
        judgement = judgements[judgements[:, 0] == datasetnum][:, 1]
        query_dataset = queries[datasetnum - 1]
        # p = tp8.calc_precision()
        query_dataset = " ".join(query_dataset)
        st.write(query_dataset)
        st.write(judgement)
        if matching == "Vectorial space model":
            formula = column3.radio(
                "Formula",
                (
                    "Scalar product",
                    "Cosine similarity",
                    "Jaccard similarity",
                ),
            )
            if formula == "Scalar product":
                res = sh.scalar_product(query_dataset, port, token)
            elif formula == "Cosine similarity":
                res = pd.DataFrame.from_dict(
                    sh.cosine(query_dataset, port, token), orient="index"
                )
            elif formula == "Jaccard similarity":
                res = pd.DataFrame.from_dict(
                    sh.jaccard(query_dataset, port, token), orient="index"
                )
        elif matching == "Probabilistic model":
            k = column3.number_input("k", value=1.5)
            b = column3.number_input("b", value=0.75)
            res = sh.BM25(query_dataset, k, b, port, token)
        elif matching == "Boolean model":
            if sh.check_expression(query_dataset):
                res, res2 = sh.evaluate_expression2(query_dataset, port, token)
            else:
                st.write("### Invalid expression")
        if res:
            resdf = pd.DataFrame.from_dict(res, orient="index")
            st.write(resdf)
            p5 = tp8.calc_precision(res, 5)
            p10 = tp8.calc_precision(res, 10)
            p = tp8.calc_precision2(res)
            r = tp8.calc_recall(res, judgement)
            f = tp8.calc_fscore(res, judgement)
            # dataframe
            df = pd.DataFrame(
                {
                    "Precision@5": [p5],
                    "Precision@10": [p10],
                    "Precision": [p],
                    "Recall": [r],
                    "Fscore": [f],
                }
            )
            st.write(df)
            precision, rappel = tp8.rappel_precision(res, judgement)
            # plot the graph with the two arrays rappel as the x axis and precision as the y axis
            figure = go.Figure(
                data=go.Scatter(x=rappel, y=precision, mode="lines+markers")
            )
            st.plotly_chart(figure)


if __name__ == "__main__":
    main()
