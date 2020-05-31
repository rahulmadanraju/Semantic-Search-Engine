import numpy as np
import pandas as pd
import glob
from Summarizer.Summarization import bart_summarizer
from KeywordExtraction.Keyword_Extractor import  Synonym_Keywords_Generation
from SpellCorrection.Spell_Correction import SpellCheck2
from sentence_transformers import SentenceTransformer
from flask import Flask, request, jsonify, render_template
from tqdm import tqdm
import pickle as pkl
import scipy.spatial
import sys

app = Flask(__name__)
# model = pickle.load(open('model.pkl', 'rb'))


def predict_results(query_input):
    corpus_embeddings = pkl.load(open('finalized_model.sav','rb'))
    embedder = SentenceTransformer('bert-base-nli-mean-tokens')
    df_sentences = pd.read_csv("keyword.csv")
    df_sentences = df_sentences.set_index("Unnamed: 0")
    # df_sentences.head()

    df_sentences = df_sentences["title"].to_dict()
    df_sentences_list = list(df_sentences.keys())
    # len(df_sentences_list)

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

        # print("\n\n=========================================================")
        # print("==========================Query==============================")
        # print("====",query,"=====")
        # print("=========================================================")


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


            # print("Score:   ", "(Score: %.4f)" % (1-distance) , "\n", file=sys.stdout )
            # print("Paragraph:   ", corpus[idx].strip(), "\n", file=sys.stdout )
            # row_dict = full_data.loc[full_data.index== corpus[idx]].to_dict()
            #print("Keyword:  " , row_dict["keyword"][corpus[idx]] , "\n")
            # print("Title:  " , row_dict["title"][corpus[idx]] , "\n", file=sys.stdout)
            # print("Domain:  " , row_dict["domain"][corpus[idx]] , "\n", file=sys.stdout)
            # print("Url:  " , row_dict["url"][corpus[idx]] , "\n", file=sys.stdout)
            # print("Snippet:  " , row_dict["snippet_description"][corpus[idx]] , "\n", file=sys.stdout)
            #print("-------------------------------------------")


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    #int_features = [int(x) for x in request.form.values()]
    #final_features = [np.array(int_features)]
    rawtext = request.form['rawtext']
    prediction_Spell = SpellCheck2(rawtext)
    condition = request.form.get('condition')
    if condition:
        return render_template('index.html',ctext=rawtext, prediction_Spell='Did you mean: {}'.format(prediction_Spell))
    else:
        rawtext = request.form['rawtext']
        prediction_Spell = SpellCheck2(rawtext)
        prediction_Summ, Summ_Scores = bart_summarizer(prediction_Spell)
        scores , prediction_KeyWord,prediction_Synonyms = Synonym_Keywords_Generation(prediction_Spell)
        lst = [prediction_Summ]
        input_query = lst + prediction_KeyWord
        return render_template('index.html',ctext=rawtext, prediction_Spell='Did you mean: {}'.format(prediction_Spell),
        # prediction_Summ = 'Summary: {}' .format(prediction_Summ),
        # prediction_KeyWord= 'Keywords: {}' .format(prediction_KeyWord),
        # prediction_Synonyms = 'Synonyms: {}' .format(prediction_Synonyms),
        prediction_Results = predict_results(input_query))
    
    # Move the action to crawl the data and train the model
    '''
    if condition == 'Y':
        prediction_Summ, Summ_Scores = bart_summarizer(prediction_Spell)
        scores , prediction_KeyWord,prediction_Synonyms = Synonym_Keywords_Generation(prediction_Spell)
        # bring the function of data crawl here
        # bring the function of training here
        print('The model is trained and ready to use')

    elif rawtext == 'N':
        input_query = rawtext
        predict_results(input_query)

    '''
    
    
    #input_query = prediction_Summ, prediction_KeyWord



    # output = round(prediction[0], 2)
    


@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction_Spell = SpellCheck2(data)
    prediction_Summ, Summ_Scores = bart_summarizer(prediction_Spell)
    scores , prediction_KeyWord, prediction_Synonyms = Synonym_Keywords_Generation(prediction_Spell)
    input_query = [prediction_Summ] + prediction_KeyWord

    #rawtext = request.form['rawtext']
	#prediction = SpellCheck2(rawtext)
    # output = prediction
    return jsonify(prediction_Spell, predict_results(input_query))

if __name__ == "__main__":

         app.run(debug=True)