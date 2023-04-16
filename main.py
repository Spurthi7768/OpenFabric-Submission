import os
import warnings
from ontology_dc8f06af066e4a7880a5938933236037.simple_text import SimpleText

from openfabric_pysdk.context import OpenfabricExecutionRay
from openfabric_pysdk.loader import ConfigClass
from time import time
import pandas as pd
import numpy as np
import string
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import math
warnings.filterwarnings('ignore')

stopwords_list = stopwords.words('english')

lemmatizer = WordNetLemmatizer()

greet=["Hi", "Hey", "Is anyone there?", "Hello", "Hay"]
goodbye=["Bye", "See you later", "Goodbye"]
thanks= ["Thanks", "Thank you", "That's helpful", "Thanks for the help"]
about= ["Who are you?", "What are you?", "Who you are?" ]
name= ["what is your name", "what should I call you", "whats your name?"]

def my_tokenizer(doc):
    words = word_tokenize(doc)
    
    pos_tags = pos_tag(words)
    
    non_stopwords = [w for w in pos_tags if not w[0].lower() in stopwords_list]
    
    non_punctuation = [w for w in non_stopwords if not w[0] in string.punctuation]
    
    lemmas = []
    for w in non_punctuation:
        if w[1].startswith('J'):
            pos = wordnet.ADJ
        elif w[1].startswith('V'):
            pos = wordnet.VERB
        elif w[1].startswith('V'):
            pos = wordnet.VERB
        elif w[1].startswith('N'):
            pos = wordnet.NOUN
        elif w[1].startswith('R'):
            pos = wordnet.ADV
        else:
            pos = wordnet.NOUN
        
        lemmas.append(lemmatizer.lemmatize(w[0], pos))

    return lemmas

def ask_question(question, all_data,tfidf_vectorizer,tfidf_matrix):
    if(question in greet):
      return "Hello!"
    elif(question in goodbye):
      return "See you later"
    elif(question in thanks):
      return "Happy to help!"
    elif(question in about):
      return "My name is SciBot. You can ask questions about science"
    elif(question in name):
      return "My name is SciBot"
    else:
      query_vect = tfidf_vectorizer.transform([question])
      similarity = cosine_similarity(query_vect, tfidf_matrix)
      max_similarity = np.argmax(similarity, axis=None)     
      ans=all_data.iloc[max_similarity]['correct_answer']
      if(math.isnan(all_data.iloc[max_similarity]['support'])):
         extra_info=""
      else:
        extra_info=all_data.iloc[max_similarity]['support']
      return str(ans)+str(extra_info)
    

############################################################
# Callback function called on update config
############################################################
def config(configuration: ConfigClass):
    # TODO Add code here
    configuration.dependencies.append({
        'name': 'numpy',
        'repository': 'pypi'
    })
    configuration.dependencies.append({
        'name': 'pandas',
        'repository': 'pypi'
    })
    configuration.dependencies.append({
        'name': 'nltk',
        'repository': 'pypi'
    })
    configuration.dependencies.append({
        'name': 'scikit-learn',
        'repository': 'pypi'
    })
    import nltk
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')
    pass


############################################################
# Callback function called on each execution pass
############################################################
def execute(request: SimpleText, ray: OpenfabricExecutionRay) -> SimpleText:
    output = []
    data_path = os.path.join(ray.execution_path, 'data')
    df1 = pd.read_csv(os.path.join(data_path, 'train_cmu.csv'))
    df2=pd.read_csv(os.path.join(data_path, 'train_sciq.csv'))
    all_data = pd.concat([df1, df2], ignore_index=True, sort=False)
    all_data = all_data[['question', 'correct_answer','support']]
    tfidf_vectorizer = TfidfVectorizer(tokenizer=my_tokenizer)
    tfidf_matrix = tfidf_vectorizer.fit_transform(tuple(all_data['question']))
    
    for text in request.text:
        # TODO Add code here

        response = ask_question(text,all_data,tfidf_vectorizer,tfidf_matrix)
        output.append(response)

    return SimpleText(dict(text=output))
