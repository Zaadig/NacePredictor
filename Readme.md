# Description

This Python script is designed to scrape NACE codes, titles, and descriptions from [nacev2.com] and then match user inputs or categories from excel files to their closest NACE codes for further analysis.

## Requirements

- `requests`
- `json`
- `bs4`
- `nltk`
- `rapidfuzz`
- `pandas`
- `numpy`
- `gensim`
- `sklearn`
- `tqdm`
- `re`

The `glove.6B.300d.w2vformat.txt` word vectors file is also required, it is a large dictionary of vectors representations for words, accessible from [https://nlp.stanford.edu/projects/glove/].

## Scripts

### 1. agribalyse_nace_predict.py
     - Extracts the nace codes, titles, and descriptions from the specified URLs.
     - Processes and vectorizes the textual descriptions using TF-IDF and glove word vectors.
     - Matches user-inputted data with the best nace codes using fuzzy string matching.
     - Uses cosine similarity to find the closest nace codes for product descriptions.
     - Processes and appends predictions to the generated excel files.

### 2. ecoinvent_preprocess.py
     - Loads Ecoinvent data excel file and performs preliminary processing.
     - Merges data from different sheets of the excel file.
     - Extracts desired columns from the merged data and savec the data to a new output excel file


### 3. ecoinvent_nace_predict.py
     - Uses the preprocessed data from Ecoinvent to predict nace codes.
     - Vectorizes activity names and other features using TF-IDF and glove word vectors.
     - Predicts nace codes for unique activity names using cosine similarity.
     - Maps the predictions back to the original dataframe.