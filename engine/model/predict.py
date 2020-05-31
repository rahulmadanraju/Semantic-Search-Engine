
import pickle as pkl
import pandas as pd
import scipy
from tqdm import tqdm


def predict_results(query_input, embedder):
    corpus_embeddings = pkl.load(open('finalized_model.sav','rb'))

    df_sentences = pd.read_csv("keyword.csv")
    df_sentences = df_sentences.set_index("Unnamed: 0")
    df_sentences = df_sentences["keywords"].to_dict()
    df_sentences_list = list(df_sentences.keys())
    df_sentences_list = [str(d) for d in tqdm(df_sentences_list)]

    corpus = df_sentences_list
    queries = query_input
    query_embeddings = embedder.encode(queries,show_progress_bar=True)

    full_data = pd.read_csv("Processed_Full.csv", index_col=0)
    # Find the closest 5 sentences of the corpus for each query sentence based on cosine similarity
    closest_n = 1
    final_results = []
    # print("\nTop 10 most similar sentences in corpus:")
    for query, query_embedding in zip(queries, query_embeddings):
        distances = scipy.spatial.distance.cdist([query_embedding], corpus_embeddings, "cosine")[0]

        results = zip(range(len(distances)), distances)
        results = sorted(results, key=lambda x: x[1])


        for idx, distance in results[0:closest_n]:
            Score = (1-distance)
            Paragraph = corpus[idx].strip()
            row_dict = full_data.loc[full_data.index== corpus[idx]].to_dict()
            Title = row_dict["title"][corpus[idx]]
            Domain = row_dict["domain"][corpus[idx]]
            Url = row_dict["url"][corpus[idx]]
            Snippet = row_dict["snippet_description"][corpus[idx]]
            result = {'Score': Score, 'Paragraph': Paragraph , 'Title' : Title, 'Domain' : Domain, 'Url' : Url,'Snippet' : Snippet}
            final_results.append(result)

    return final_results