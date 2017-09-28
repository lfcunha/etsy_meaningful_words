from etsy import config
from etsy.etsy_top_store_keywords import EtsyTermAnalyzes
import logging
import sys


# set up the logger
_logger = logging.getLogger("logger")
_logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
_logger.addHandler(ch)

# use a predefined list of store names to analyze
stores_to_analyze = config.get('etsy', 'STORES_TO_ANALYZE').split(",")

if __name__ == "__main__":
    # instantiate the Etsy meaningful word analyzer class
    e = EtsyTermAnalyzes()
    e.stores_to_analyze = stores_to_analyze
    e._logger = _logger
    e._config = config
    # e.tfidf_method = "diy"  #scikit

    # get the top n (defined in config) meaningful words for each store
    e.top_meaningful_words