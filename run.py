import argparse
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
    parser = argparse.ArgumentParser(description='Run Home test questions 3 or 8)')
    parser.add_argument('--tfidf', metavar='tfidf-method', type=str,
                        help='which tf-idf implementation to use: diy / scikit')

    parser.add_argument('--level', metavar='LOG_LEVEL', type=int, default=logging.DEBUG,
                        help='Log Level (10=debug, 20=info, 30=warning')

    args = parser.parse_args()

    _logger.setLevel(args.level)
    if args.tfidf and args.tfidf not in ("tfidf", "diy") or args.level and args.level not in (10,20,30,40):
        print("Usage:\n> python3 run.py [--tfidf scikit|diy [--level 10|20|30|40]]")
        sys.exit(0)
    tfidf_impl = args.tfidf if args.tfidf else "scikit"

    # instantiate the Etsy meaningful word analyzer class
    e = EtsyTermAnalyzes()
    e.stores_to_analyze = stores_to_analyze
    e._logger = _logger
    e._config = config
    e.tfidf_method = tfidf_impl  #scikit

    # get the top n (defined in config) meaningful words for each store
    e.top_meaningful_words
