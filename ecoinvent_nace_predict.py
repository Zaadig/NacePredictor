import pandas as pd
import numpy as np
import os
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity
import re
from tqdm import tqdm

word_vectors = KeyedVectors.load_word2vec_format('glove.6B.300d.w2vformat.txt', binary=False)

def preprocess_text(text):
    text = text.lower()

    text = re.sub(r'[^\w\s]', '', text)

    text = re.sub(r'\s+', ' ', text)

    text = text.strip()

    return text

def text_to_vector(text, tfidf_vectorizer):
    words = text.split()
    word_vecs = []
    weights = []
    
    text_tfidf = tfidf_vectorizer.transform([text])
    text_tfidf_dict = dict(zip(tfidf_vectorizer.get_feature_names_out(), text_tfidf.toarray()[0]))
    
    for word in words:
        if word in word_vectors:
            word_vecs.append(word_vectors[word])
            weights.append(text_tfidf_dict.get(word, 0))
    if word_vecs:
        if np.sum(weights) == 0:
            return np.average(word_vecs, axis=0)
        else:
            return np.average(word_vecs, axis=0, weights=weights)
        
    else:
        return np.zeros(word_vectors.vector_size)

def find_closest_nace(input_text, tfidf_vectorizer, nace_vectors):

    input_vector = text_to_vector(input_text, tfidf_vectorizer).reshape(1, -1)

    similarities = {}

    for code, vec in nace_vectors.items():
        similarities[code] = cosine_similarity(input_vector, vec.reshape(1, -1))[0][0]
    closest_code = max(similarities.keys(), key=lambda code: similarities[code])
    print(f"Predicted NACE for '{input_text}': {closest_code}")
    
    return closest_code

with open('nace_data.json', 'r') as f:
    data = json.load(f)
nace_includes = data['nace_includes']
nace_includes_clean = {code: preprocess_text(description) for code, description in nace_includes.items()}

vectorizer = TfidfVectorizer()
descriptions = list(nace_includes_clean.values())
vectorizer.fit(descriptions)
nace_vectors = {code: text_to_vector(description, vectorizer) for code, description in nace_includes_clean.items()}

df_new = pd.read_excel('ecoinvent_dst/ecoinvent_preprocessed.xlsx')

unique_activity_names = df_new['Activity Name'].unique()
activity_name_to_input_text = {activity_name: activity_name + ' ' + df_new[df_new['Activity Name'] == activity_name]['Special Activity Type'].iloc[0] + ' ' + df_new[df_new['Activity Name'] == activity_name]['Sector'].iloc[0] + ' ' + df_new[df_new['Activity Name'] == activity_name]['ISIC Classification'].iloc[0] for activity_name in unique_activity_names}

print(f"Total number of unique texts: {len(activity_name_to_input_text)}")

predictions = {activity_name: find_closest_nace(input_text, vectorizer, nace_vectors) for activity_name, input_text in tqdm(activity_name_to_input_text.items())}

df_new['predicted_nace'] = df_new['Activity Name'].map(predictions)

df_new.to_excel('ecoinvent_dst/ecoinvent_with_nace.xlsx', index=False)
