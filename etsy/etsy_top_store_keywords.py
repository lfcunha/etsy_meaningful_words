from collections import Counter
import json
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import os
from pathlib import Path

from utils.get_store_data import get_store_listings
from utils.get_store_data import get_all_stores


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

    @property
    def stores_to_analyze(self):
        return self._stores_to_analyze

    @stores_to_analyze.setter
    def stores_to_analyze(self, stores):
        self._stores_to_analyze = stores

    @property
    def bloblist(self):
        if not self._bloblist:
            self._merge_listings_and_preprocess_tokens()
        return self._bloblist

    @staticmethod
    def _get_nltk_data():
        nltk.download('punkt')
        nltk.download('wordnet')

    def _merge_listings_and_preprocess_tokens(self):
        """ Merge all store listings' text (title and description) and preprocess tokens by:
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

