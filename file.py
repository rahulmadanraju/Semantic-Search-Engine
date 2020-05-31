import os
import json
from sentence_transformers import SentenceTransformer

from engine.utils import test
from engine.preprocessing.keywords import extract_keywords
from engine.preprocessing.data import data_processing
from engine.model.transformers import sentence_embeddings, predict_results

#from rokin.serpstack import Serpstack

#apikey = os.environ.get('SERPSTACK_API_KEY', 'Serpstack API key not provided')
#serp = Serpstack(apikey)
serp = []
embedder = SentenceTransformer('bert-large-nli-mean-tokens')

def hello(request):
    return test()

def serpstack(request):
    search = request.form.get('search')
    description = request.form.get('description')
    # problem statement
    problem = search + ' ' + description
    keywords = extract_keywords(problem)
    queries = serp.create_queries(keywords)
    results = serp.execute_queries(queries)
    return json.dumps(results)

# read data from serp/database and train the model 
def model_training(path):
    sentences = data_processing(path)
    sentence_embeddings(sentences, embedder)

# path to database
path = "GSR.json"

train = input("Press Y to train or N to make use of the Search Engine: ")
if train == "Y":
    query = input("Enter your query or problem statement here: ")
    # serpstack(request)
    model_training(path)
elif train == "N":
    query = [input("Enter you query here: ")]
    results = predict_results(query, embedder)
    print(results)
else:
    print("Please enter the right key")








'''
# test code
body = """
Welcome to Wikipedia! Before starting a new article, please review Wikipedia's notability requirements. In short, the topic of an article must have already been the subject of publication in reliable sources, such as books, newspapers, magazines, peer-reviewed scholarly journals and websites that meet the same requirements as reputable print-based sources. Information on Wikipedia must be verifiable; if no reliable third-party sources can be found on a topic, then it should not have a separate article. Please search Wikipedia first to make sure that an article does not already exist on the subject.
"""

keywords = extract_keywords(body)

print(keywords)
'''