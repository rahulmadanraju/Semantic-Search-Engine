from sentence_transformers import SentenceTransformer
from process import processing_combined
import pickle as pkl

# Corpus with example sentences

def model_transformer(query_data):
    df_sentences_list, df = processing_combined(query_data)
    embedder = SentenceTransformer('bert-base-nli-mean-tokens')
    corpus = df_sentences_list
    corpus_embeddings = embedder.encode(corpus,show_progress_bar = True)

    filename = 'finalized_model.sav'
    pkl.dump(corpus_embeddings, open(filename, 'wb'))

    return embedder, corpus, df

