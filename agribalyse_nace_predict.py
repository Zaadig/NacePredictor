import requests
import json
from bs4 import BeautifulSoup
from nltk.util import ngrams
from rapidfuzz import process, fuzz


main_urls = ["https://nacev2.com/en/activity/agriculture-forestry-and-fishing",
            "https://nacev2.com/en/activity/mining-and-quarrying",
            "https://nacev2.com/en/activity/manufacturing",
            "https://nacev2.com/en/activity/electricity-gas-steam-and-air-conditioning-supply",
            "https://nacev2.com/en/activity/water-supply-sewerage-waste-management-and-remediation-activities",
            "https://nacev2.com/en/activity/construction",
            "https://nacev2.com/en/activity/wholesale-and-retail-trade-repair-of-motor-vehicles-and-motorcycles",
            "https://nacev2.com/en/activity/transportation-and-storage",
            "https://nacev2.com/en/activity/accommodation-and-food-service-activities",
            "https://nacev2.com/en/activity/information-and-communication",
            "https://nacev2.com/en/activity/financial-and-insurance-activities",
            "https://nacev2.com/en/activity/real-estate-activities",
            "https://nacev2.com/en/activity/professional-scientific-and-technical-activities",
            "https://nacev2.com/en/activity/administrative-and-support-service-activities",
            "https://nacev2.com/en/activity/public-administration-and-defence-compulsory-social-security",
            "https://nacev2.com/en/activity/education",
            "https://nacev2.com/en/activity/human-health-and-social-work-activities",
            "https://nacev2.com/en/activity/arts-entertainment-and-recreation",
            "https://nacev2.com/en/activity/other-service-activities",
            "https://nacev2.com/en/activity/activities-of-households-as-employers-undifferentiated-goods-and-services-producing-activities-of-households-for-own-use",
            "https://nacev2.com/en/activity/activities-of-extraterritorial-organisations-and-bodies"
            ]

data_file = 'nace_data.json'

def generate_ngrams(words):
    one_grams = words
    two_grams = [' '.join(gram) for gram in ngrams(words, 2)]
    three_grams = [' '.join(gram) for gram in ngrams(words, 3)]

    return one_grams + two_grams + three_grams

def scrape_data(urls):

    nace_codes = {}
    nace_keywords = {}
    nace_includes = {}

    for main_url in urls:
        response = requests.get(main_url)
        soup = BeautifulSoup(response.text, 'html.parser')

        items = soup.find_all('a', {'class': 'list__grid--item'})

        for item in items:

            code_name = item.get_text()

            url = "https://nacev2.com" + item['href']

            code, name = code_name.split(' - ', 1)
            nace_codes[code] = name

            response = requests.get(url)
            soup = BeautifulSoup(response.content, 'html.parser')

            includes = soup.find(class_='item--description').text
            nace_includes[code] = includes

            keywords = generate_ngrams(includes.split())

            for keyword in keywords:
                nace_keywords.setdefault(keyword, []).append(code)

    with open(data_file, 'w') as f:
        json.dump({'nace_keywords': nace_keywords, 'nace_codes': nace_codes, 'nace_includes': nace_includes}, f)

def load_data():

    with open(data_file, 'r') as f:
        data = json.load(f)

    return data['nace_keywords'], data['nace_codes'], data['nace_includes']


def find_best_matches(word, possibilities):
    return process.extract(word, possibilities, scorer=fuzz.WRatio, score_cutoff=90, limit=7)

def search_nace(user_input, nace_keywords, nace_codes):
    if user_input in nace_codes:
        return [{"NACE code": user_input, "title": nace_codes[user_input], "score": 100}]
    else:
        best_matches = find_best_matches(user_input, nace_keywords.keys())
        matching_codes_with_titles = []

        for match, score, _ in best_matches:
            if len(match) < 3:
                continue

            for code in nace_keywords.get(match, []):
                matching_codes_with_titles.append({"NACE code": code, "title": nace_codes[code], "score": score})

        if matching_codes_with_titles:
            matching_codes_with_titles = list({v['NACE code']:v for v in matching_codes_with_titles}.values())
            matching_codes_with_titles.sort(key=lambda x: x['score'],reverse=True)
            return matching_codes_with_titles
        else:
            return [{"message": "No matching NACE code found."}]


try:
    nace_keywords, nace_codes, nace_includes = load_data()
except FileNotFoundError:
    scrape_data(main_urls)
    nace_keywords, nace_codes, nace_includes = load_data()

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()

descriptions = list(nace_includes.values())

tfidf_matrix = vectorizer.fit_transform(descriptions)

feature_names = vectorizer.get_feature_names_out()

tfidf_vector = tfidf_matrix[0]

tfidf_dict = dict(zip(feature_names, tfidf_vector.toarray()[0]))

sorted_tfidf_dict = sorted(tfidf_dict.items(),key=lambda item: item[1],reverse=True)

for word, score in sorted_tfidf_dict[:10]:
    print(f"{word}: {score}")



import re

def preprocess_text(text):
    text = text.lower()

    text = re.sub(r'[^\w\s]', '', text)

    text = re.sub(r'\s+', ' ', text)

    text = text.strip()

    return text


nace_includes_clean = { code: preprocess_text(description) for code, description in nace_includes.items() }



import numpy as np
from gensim.models import KeyedVectors

word_vectors = KeyedVectors.load_word2vec_format('glove.6B.300d.w2vformat.txt', binary=False)

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


from sklearn.metrics.pairwise import cosine_similarity

def find_closest_nace(input_text, tfidf_vectorizer):

    input_vector = text_to_vector(input_text, tfidf_vectorizer).reshape(1, -1)

    similarities = {}

    for code, vec in nace_vectors.items():
        similarities[code] = cosine_similarity(input_vector, vec.reshape(1, -1))[0][0]
    closest_code = max(similarities.keys(), key=lambda code: similarities[code])

    return closest_code




import os
import pandas as pd


src_folder = "agribalyse_src"
dst_folder = "agribalyse_dst"

nace_vectors = {code: text_to_vector(description, vectorizer) for code, description in nace_includes_clean.items()}

if not os.path.exists(dst_folder):
    os.makedirs(dst_folder)

excel_files = [f for f in os.listdir(src_folder) if f.endswith('.xlsx') or f.endswith('.xls')]

for file in excel_files:

    df = pd.read_excel(os.path.join(src_folder, file))

    category_index = df['Category'].first_valid_index()

    if category_index is not None:

        category = df.loc[category_index, 'Category']
        if isinstance(category, str):
            category = category.replace('/', ' ')
            predicted_nace = find_closest_nace(category, vectorizer)
            df.loc[category_index, 'predicted_nace'] = predicted_nace

        else:
            df['predicted_nace'] = np.nan
    else:
        df['predicted_nace'] = np.nan

    df = df[['Name', 'Category', 'predicted_nace', 'Description', 'input_item', 'input_category', 'input_amount', 'input_unit', 'input_description', 'input_additional_description', 'output_item', 'output_category', 'output_amount', 'output_unit', 'output_description', 'output_additional_description']]

    df.to_excel(os.path.join(dst_folder, file), index=False)