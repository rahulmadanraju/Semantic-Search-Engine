
import pandas as pd
import re
from tqdm import tqdm

def processing_combined (query_data):
    query_data.drop_duplicates(keep = 'first')
    query_data['snippet'].fillna('no description', inplace=True)
    query_data.fillna('no title', inplace=True)
    query_data['snippet'] = query_data['snippet'].apply(lambda x: re.sub('[^a-zA-z0-9\s]','',x))
    df_dict = query_data.to_dict()
    len_text = len(df_dict["title"])

    #keyword_list  = [] # keyword
    snippet_description_list = [] # snippet_description
    snippet_list = [] # snippet
    title_list = [] # title
    url_list = [] # url 
    domain_list = [] # domain
    for i in tqdm(range(1,len_text)):
        snippet = df_dict["snippet"][i].split("\n")
        snippet_description = df_dict["snippet"][i]
        title = df_dict["title"][i]
        url = df_dict["url"][i]
        domain = df_dict["domain"][i]
        for b in snippet:
            #keyword_list.append(keyword)
            snippet_list.append(b)
            snippet_description_list.append(snippet)
            title_list.append(title)
            url_list.append(url)
            domain_list.append(domain)

    df_sentences = pd.DataFrame({"title":title_list},index=snippet_list)
    df_sentences.to_csv("keyword.csv")
    # df_sentences.head()

    df_sentences = pd.DataFrame({"title":title_list,"url":url_list,"domain":domain_list, "snippet_description": snippet_description_list },index=snippet_list)
    df_sentences.to_csv("Processed_Full.csv")
    # df_sentences.head()
    # "keyword":keyword_list,

    df_sentences = pd.read_csv("keyword.csv")
    df_sentences = df_sentences.set_index("Unnamed: 0")
    # df_sentences.head()

    df_sentences = df_sentences["title"].to_dict()
    df_sentences_list = list(df_sentences.keys())
    # len(df_sentences_list)

    df_sentences_list = [str(d) for d in tqdm(df_sentences_list)]
    # print(df_sentences_list)

    full_data = pd.read_csv("Processed_Full.csv", index_col=0)
    # df.head()

    return df_sentences_list, full_data

    