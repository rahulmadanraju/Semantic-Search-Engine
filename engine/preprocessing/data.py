import pandas as pd
import numpy as np
from tqdm import tqdm
import re

def data_processing (path):
    df = pd.read_json(path)
    df = df[~df["Label"].isin(["Blacklisted"])]
    df = df.drop(["timestamp", "Label"], axis=1)
    print(df.shape)
    df = df.replace(r'', np.nan)
    df = df.drop_duplicates(subset=['title', 'snippet'], keep="first")
    print(df.shape)
    df.dropna(inplace=True)
    print(df.shape)
    df["snippet"] = df["snippet"].apply(lambda x: re.sub("[^a-zA-z0-9\s]",'',x))
    df["keywords"] = df["keywords"].apply(lambda x: re.sub("[^a-zA-z0-9\s]",'',x))
    df["title"] = df["title"].apply(lambda x: re.sub("[^a-zA-z0-9\s]",'',x))
    df.reset_index(inplace= True)
    df_dict = df.to_dict()
    len_text = len(df_dict["keywords"])

    keyword_list  = [] # keyword
    snippet_description_list = [] # snippet_description
    snippet_list = [] # snippet
    title_list = [] # title
    url_list = [] # url 
    domain_list = [] # domain
    for i in tqdm(range(0,len_text)):
        keywords = df_dict["keywords"][i]
        snippet = df_dict["snippet"][i].split("\n")
        snippet_description = df_dict["snippet"][i]
        title = df_dict["title"][i]
        url = df_dict["url"][i]
        domain = df_dict["domain"][i]
        for b in snippet:
            keyword_list.append(keywords)
            snippet_list.append(b)
            snippet_description_list.append(snippet)
            title_list.append(title)
            url_list.append(url)
            domain_list.append(domain)

    df_sentences = pd.DataFrame({"keywords":keyword_list},index=snippet_list)
    df_sentences.to_csv("keyword.csv")
    df_sentences = pd.read_csv("keyword.csv")
    df_sentences = df_sentences.set_index("Unnamed: 0")
    df_sentences = df_sentences["keywords"].to_dict()
    df_sentences_list = list(df_sentences.keys())
    len(df_sentences_list)
    df_sentences_list = [str(d) for d in tqdm(df_sentences_list)]

    df = pd.DataFrame({"keyword":keyword_list,"title":title_list,"url":url_list,"domain":domain_list, "snippet_description": snippet_description_list },index=snippet_list)
    df.to_csv("Processed_Full.csv")

    return df_sentences_list






#queries = ['molded interconnect device MID electronics Two-shot molding']
#query_embeddings = embedder.encode(queries,show_progress_bar=True)