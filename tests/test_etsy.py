from etsy import config
import json
import numpy as np
import os
import pytest
import unittest
import scipy
from sklearn.feature_extraction.text import TfidfVectorizer

from utils.tfidf import tfidf_scikit
from utils.tfidf import tfidf_diy
from etsy.etsy_top_store_keywords import EtsyTermAnalyzes



class TestEtsy(unittest.TestCase):

    @pytest.fixture
    def get_etsy(self):
        return EtsyTermAnalyzes()

    @pytest.fixture
    def get_listings(self):
        e = self.get_etsy()
        return e.get_listings()

    def test_read_file1(self):
        data_path = os.path.join(os.getcwd(), "data", "listings_text.json")
        stores_listings = json.load(open(data_path, "r"))
        self.assertIsInstance(stores_listings, dict)
        self.assertIn("Plumailes", stores_listings)

    def test_instantiate_class(self):
        e = self.get_etsy()
        self.assertIsInstance(e, EtsyTermAnalyzes)

    def test_read_file2(self):
        sl = self.get_listings()
        self.assertIsInstance(sl, dict)
        self.assertIn("Plumailes", sl)

    def test_tokenize(self):
        e = self.get_etsy()
        #sl = self.get_listings()
        stores_to_analyze = config.get('etsy', 'STORES_TO_ANALYZE').split(",")
        e.stores_to_analyze = stores_to_analyze
        e._config = config
        e.tfidf_method = "scikit"

        self.assertIsInstance(e.bloblist, list)
        self.assertIsInstance(e.bloblist[0], str)
        self.assertGreater(len(e.bloblist), 1)

    def test_tfidf(self):
        e = self.get_etsy()
        #sl = self.get_listings()
        stores_to_analyze = config.get('etsy', 'STORES_TO_ANALYZE').split(",")
        e.stores_to_analyze = stores_to_analyze
        e._config = config
        e.tfidf_method = "scikit"

        bloblist = e.bloblist
        tfidf = tfidf_scikit(bloblist)

        self.assertIsInstance(tfidf, tuple)
        self.assertIsInstance(tfidf[1], TfidfVectorizer)
        self.assertIsInstance(tfidf[0], scipy.sparse.csr.csr_matrix)

    def test_tfidf_diy(self):
        e = self.get_etsy()
        # sl = self.get_listings()
        stores_to_analyze = config.get('etsy', 'STORES_TO_ANALYZE').split(",")
        e.stores_to_analyze = stores_to_analyze
        e._config = config
        e.tfidf_method = "scikit"

        bloblist = e.bloblist
        tfidf = tfidf_diy(bloblist)

        self.assertIsInstance(tfidf, list)
