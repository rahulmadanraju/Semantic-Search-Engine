import pandas as pd
import glob
from model import model_transformer
import scipy.spatial
import pickle as pkl


""" importing data from article_data """

#path = r'JSON_Files/' # use your path
all_files = glob.glob("article_data/*.json")

# df = pd.read_csv('data_searchAPI/output.csv')
li = []

for filename in all_files:
    df = pd.read_json(filename)
    li.append(df)

frame = pd.concat(li, axis=0, ignore_index=True)

selectColumns = ['title','siteName','pageUrl','text', 'date']
frame = frame[selectColumns]

frame = frame.rename(columns={"pageUrl": "url", "text": "snippet", "siteName": "domain"})
frame = frame.drop(["date"], axis=1)
frame = frame.head(10)
frame = frame.drop([0])

""" importing the data from google_data """

path = r'google_data/GSR.json' # use your path
# all_files = glob.glob(path + "/*.csv")

# df = pd.read_csv('data_searchAPI/output.csv')
# li = []

# for filename in all_files:
#    df = pd.read_csv(filename, index_col=None, header=0)
#    li.append(df)

# frame = pd.concat(li, axis=0, ignore_index=True)

# frame.head()

df_sv = pd.read_json(path, dtype={
    'date': str, 
    'keyword': str
})

# df_sv.head()
df = df_sv[~df_sv['Label'].isin(['Blacklisted'])]
df = df.drop(["timestamp","keywords", "Label"], axis=1)
df = df.head(10)

# Combining the data from article and google data
frame_data= pd.concat([frame,df])
# frame_data = df
# load the processed and trained model
embedder, corpus, full_data = model_transformer(frame_data)
print (corpus)
#with open("path/.pkl" , "rb") as file_:
corpus_embeddings = pkl.load(open('finalized_model.sav','rb'))


queries = ['molded interconnect devices']
query_embeddings = embedder.encode(queries,show_progress_bar=True)

# Find the closest 5 sentences of the corpus for each query sentence based on cosine similarity
closest_n = 5
print("\nTop 10 most similar sentences in corpus:")
for query, query_embedding in zip(queries, query_embeddings):
    distances = scipy.spatial.distance.cdist([query_embedding], corpus_embeddings, "cosine")[0]

    results = zip(range(len(distances)), distances)
    results = sorted(results, key=lambda x: x[1])

    print("\n\n=========================================================")
    print("==========================Query==============================")
    print("====",query,"=====")
    print("=========================================================")


    for idx, distance in results[0:closest_n]:
        print("Score:   ", "(Score: %.4f)" % (1-distance) , "\n" )
        print("Paragraph:   ", corpus[idx].strip(), "\n" )
        row_dict = full_data.loc[full_data.index== corpus[idx]].to_dict()
        # print("Keyword:  " , row_dict["keyword"][corpus[idx]] , "\n")
        print("Title:  " , row_dict["title"][corpus[idx]] , "\n")
        print("Domain:  " , row_dict["domain"][corpus[idx]] , "\n")
        print("Url:  " , row_dict["url"][corpus[idx]] , "\n")
        print("Snippet:  " , row_dict["snippet_description"][corpus[idx]] , "\n")
        print("-------------------------------------------")