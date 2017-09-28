from etsy import config
from etsy.etsy_top_store_keywords import EtsyTermAnalyzes
import logging
import sys
from utils.tfidf import tfidf_diy


_logger = logging.getLogger()
_logger.setLevel(logging.DEBUG)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
_logger.addHandler(ch)

stores_to_analyze = config.get('etsy', 'STORES_TO_ANALYZE').split(",")

e = EtsyTermAnalyzes()
#e._stores_to_analyze = stores_to_analyze



#print(e.listings_text)

e.stores_to_analyze = stores_to_analyze
#print("BL", e.bloblist)


scores_list = tfidf_diy(e.bloblist, _logger)


# calculate tfidf scores of each word, using a simple raw count for the tf schema
for i, scores in enumerate(scores_list):

    sorted_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    for word, score in sorted_words[:5]:
        _logger.info("\tWord: {}, TF-IDF: {}".format(word, round(score, 5)))

