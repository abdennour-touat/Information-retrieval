import streamlit as st
import pandas as pd
import numpy as np
import search as sh
import tp8 as tp8
import plotly.graph_objects as go
# st.set_page_config(layout="wide")




def main():
    pd.set_option("display.max_columns", None)
    column1, column2 = st.columns(2)
    query = column1.text_input("Query")
    datasetnum = column2.number_input("Dataset", value=1, min_value=1)
    dataset_checkbox = column2.checkbox("Query dataset")
    # query = list(filter(None, query))

    column1, column2, column3 = st.columns(3)
    tokenization = column1.checkbox("Tokenization")
    if tokenization:
        token = 1
    else:
        token = 0
    porter = column1.checkbox("Porter stemmer")
    if porter:
        port = 1
    else:
        port = 0

    matching = column2.selectbox(
        "Matching",
        ("d",
            "Vectorial space model",
            "Probabilistic model",
            "Boolean model",
        ),
    )
    if query != "":
        if matching == "Vectorial space model":
            formula = column3.selectbox(
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
            k = column3.number_input("k", value=2.0)
            b = column3.number_input("b", value=1.5)
            res = sh.BM25(query, k, b, port, token)
            res = pd.DataFrame.from_dict(res, orient="index")
            st.write(res)
        elif matching == "Boolean model":
            if sh.check_expression(query):
                res, res2 = sh.evaluate_expression2(query, port, token)
                print("hello")
                res2 = pd.DataFrame.from_dict(res, orient="index")
                st.write(res2)
            else:
                st.write("### Invalid expression")
        elif matching == "d":
            st.write("### Index")
            index = st.radio("Index", ("Docs per Term", "Term per Doc"))
            if query != "":
                if index == "Docs per Term":
                    res = pd.DataFrame(sh.search(query, port, token, doc=1))
                    st.write(res)
                elif index == "Term per Doc":
                    res = pd.DataFrame(sh.search(query, port, token, doc=0))
                    st.write(res)
    if dataset_checkbox:
        column1, column2 = st.columns(2)
        judgements = sh.read_judgments()
        queries = sh.read_queries()
        judgement = judgements[datasetnum ]
        query_dataset = queries[datasetnum - 1]
        #convert the string to lowercase
        # p = tp8.calc_precision()
        if matching == "Vectorial space model":
            formula = column3.selectbox(
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
                res = sh.cosine(query_dataset, port, token)
                
            elif formula == "Jaccard similarity":
                res = sh.jaccard(query_dataset, port, token)
        elif matching == "Probabilistic model":
            k = column3.number_input("k", value=2.0)
            b = column3.number_input("b", value=1.5)
            res = sh.BM25(query_dataset, k, b, port, token)
        if res:
            st.write(query_dataset)
            column1 , column2 = st.columns(2)
            #a dataframe of the docid and the score
            resdf = pd.DataFrame({"DocID": list(res.keys()), "Score": list(res.values())})
            file_name = sh.get_file_name(port, token, doc=0)
            voc = np.array(sh.read_file(file_name))
            p5 = tp8.calc_precision(res,judgement, 5)
            p10 = tp8.calc_precision(res,judgement, 10)
            p = tp8.calc_precision2(res,judgement,voc)
            r = tp8.calc_recall(res, judgement)
            f = tp8.calc_fscore(res, judgement,voc)
            # dataframe
            df = pd.DataFrame(
                {
                    "P@5": [p5],
                    "P@10": [p10],
                    "P": [p],
                    "R": [r],
                    "Fscore": [f],
                }
            )
            column2.write(judgement)
            column1.write(df)
            column1.write(resdf)
            precision, rappel = tp8.rappel_precision(res, judgement)
            # plot the graph with the two arrays rappel as the x axis and precision as the y axis
            figure = go.Figure(
                data=go.Scatter(x=rappel, y=precision, mode="lines+markers")
            )
            column2.plotly_chart(figure)


if __name__ == "__main__":
    main()
