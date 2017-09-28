import json
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import os
from pathlib import Path
import numpy as np
import pandas as pd
from time import time

from utils.get_store_data import get_store_listings
from utils.get_store_data import get_all_stores
from utils.tfidf import tfidf_diy
from utils.tfidf import tfidf_scikit


class Etsy(object):
    """ Etsy Represents the a group of n stores's listings text (title and description)

    """

    def __init__(self):
        self._stores_listings = None
        self._stores_to_analyze = None

    @property
    def stores_listings(self):
        if not self._stores_listings:
            self._stores_listings = self.get_listings()
        return self._stores_listings

    def get_listings(self):
        """ Read stores' listings from file, or download from top stores with most listings

        Returns:
             list: listings of each store
        """
        if self._logger:
            self._logger.info("""Getting Stores' listings""")
        data_path = os.path.join(os.getcwd(), "data", "listings_text.json")
        if Path(data_path).is_file():
            stores_listings = json.load(open(data_path, "r"))
        else:
            # if stores data not present, it gets the n top lists with most items listed
            all_stores = get_all_stores()
            stores_listings = get_store_listings(20, all_stores)
            try:
                json.dump(stores_listings, open(data_path, "w"))
            except Exception:
                os.remove(data_path)

        return stores_listings


class EtsyTermAnalyzes(Etsy):
    def __init__(self):
        super(EtsyTermAnalyzes, self).__init__()

        self._get_nltk_data()
        self._bloblist = None
        self._logger = None
        self._config = None
        self._tfidf_method = "scikit"

    @property
    def stores_to_analyze(self):
        return self._stores_to_analyze

    @stores_to_analyze.setter
    def stores_to_analyze(self, stores):
        self._stores_to_analyze = stores

    @property
    def tfidf_method(self):
        return self._tfidf_method

    @tfidf_method.setter
    def tfidf_method(self, method):
        self._tfidf_method = method

    @property
    def bloblist(self):
        if not self._bloblist:
            self._merge_listings_and_preprocess_tokens()
        return self._bloblist

    @property
    def top_meaningful_words(self):
        bloblist = self.bloblist
        start = time()
        if self.tfidf_method == "scikit":
            self.calculate_top_words_with_scikit_tfidf(bloblist)
        elif self.tfidf_method == "diy":
            self.calculate_top_words_with_own_tfidf(bloblist)
        else:
            raise Exception("Invalid tfidf method chosen")

        if self._logger:
            self._logger.info("""\nIt tools {}s to calculate the top words""".format(str(round(time() - start, 3))))

    @staticmethod
    def _get_nltk_data():
        nltk.download('punkt')
        nltk.download('wordnet')

    def _merge_listings_and_preprocess_tokens(self):
        """ Merge all store listings' text (title and description) and preprocess tokens by:
            - lowercase
            - remove newline character
            - remove english stop words
            - remove non-words (punctuation, digits)
            - lemmatize to remove plurals
            - remove words with one or two characters, to avoic abbreviations such as cm (centimeter)
            - filter by word type (http://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html)
                - remove adverbs, pronouns, interjections, etc.
                - keep only Nouns, verbs, and adjectives, since these provide more meaningful descriptions of a store

        Returns:
            list(string): a list of strings. Each string is a filtered concatenation of all of the store's listings text
        """
        if self._logger:
            self._logger.info("""Merging store's listings and tokenize words""")
        _ = self.stores_listings
        wnl = WordNetLemmatizer()
        stopwords_dict = {k: None for v, k in enumerate(stopwords.words('english'))}
        self._bloblist = []  # list of text blob of all listings for each store
        for store in self.stores_to_analyze:

            store_text = ""
            for listing in self.stores_listings[store]:
                store_text = store_text + " " + listing["title"].replace("\n", " ") + " " + \
                             listing["description"].replace("\n", " ")
            store_text = store_text.split()  # tokenize, splitting on white space
            filtered_words = [wnl.lemmatize(word.lower()) for word in store_text if
                              len(word) > 2 and word.isalpha() and word not in stopwords_dict]
            tagged = nltk.pos_tag(filtered_words)  # classify the type of the word
            # word_tags = [x[1] for x in tagged]
            # type_counter = Counter(word_tags)

            filtered_words = [x[0] for x in tagged if
                              x[1] in ('NN', 'NNS', 'JJ', 'JJS', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ')]

            self._bloblist += [" ".join(filtered_words)]

    def calculate_top_words_with_own_tfidf(self, bloblist):
        """ calculate tfidf scores for each word of each document, using my crude, non-optimized implementation of tfidf
        Args:
            bloblist: list of document blobs (TextBlob)
        """
        scores_list = tfidf_diy(bloblist, self._logger)

        # calculate tfidf scores of each word, using a simple raw count for the tf schema
        top_words = []
        for i, scores in enumerate(scores_list):
            sorted_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            # x = ["""{}({})""".format(word, round(score, 2)) for word, score in sorted_words[:5]]
            x = [word for word, score in sorted_words[:5]]
            top_words.append((self.stores_to_analyze[i], ", ".join(x)))

            # for word, score in sorted_words[:5]:
            #     _logger.info("\tStore: {}, Word: {}, TF-IDF: {}".format(stores_to_analyze[i], word, round(score, 5)))

        df = pd.DataFrame(top_words)
        df.columns = ['Store Name', 'Top Five Meaningful Words']
        print()
        print("Top Meaningful words for each store:")
        with pd.option_context('display.max_rows', None, 'display.max_columns', 10):
            print(df)

    def calculate_top_words_with_scikit_tfidf(self, docslist):
        """calculate tfidf scores for each word of each document, using scikit-learn library

        Args:
            docslist (list): list of text documents

        """
        scores_matrix, tfidf_vectorizer = tfidf_scikit(docslist)

        for i in range(len(docslist)):
            top_n_words = int(self._config.get("etsy", "TOP_N_WORDS"))
            row = np.squeeze(scores_matrix[i].toarray())
            features = tfidf_vectorizer.get_feature_names()
            topn_ids = np.argsort(row)[::-1][:top_n_words]
            top_feats = [(features[i], row[i]) for i in topn_ids]

            df = pd.DataFrame(top_feats)
            df.columns = ['Word', 'tfidf score']
            print("""Top Words for Store {}:""".format(self.stores_to_analyze[i]))
            print(df)
            print()
