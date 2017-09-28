import math
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob as tb


def tf(word, blob):
    """ calculate term frequency with simple raw count. More advanced schemas could be used, such as binary, term frequency,
        log normalization, double normalization 0.5, double normalization K.
        See https://en.wikipedia.org/wiki/Tf%E2%80%93idf for more information

    Args:
        word (string):
        blob (TextBlob):

    Returns:
        float
    """
    return blob.words.count(word) / len(blob.words)


def df(word, bloblist):
    """Calculate document frequency

    Args:
        word (string):
        blob (TextBlob):

    Returns:
        float
    """
    return sum(1 for blob in bloblist if word in blob.words)


def idf(word, bloblist):
    """Calculate inverse document frequency

    Args:
        word (string):
        blob (TextBlob):

    Returns:
        float
    """
    return math.log(len(bloblist) / (1 + df(word, bloblist)))


def tfidf(word, blob, bloblist):
    """

    Args:
        word (string): a single term
        blob (TextBlob): a blob representation of the document text
        bloblist (list): all documents
    Returns:
        float
    """
    return tf(word, blob) * idf(word, bloblist)


def tfidf_diy(bloblist, _logger=None):
    """ Calculate TF-IDF scores for each word of each document

    Args:
        bloblist (list(TextBlob)): list with text of each document
        n (int): top n words (higher tfidf scores)

    Returns:
       list(dict): list of top n words (dict with word and score) for each document
    """
    bloblist = [tb(x) for x in bloblist]  # convert each document to a TextBlob
    scores_list = []
    for i, blob in enumerate(bloblist):
        if _logger:
            _logger.info("Calculating tfidf scores of document {} of {}".format(i + 1, len(bloblist)))
        scores = {word: tfidf(word, blob, bloblist) for word in blob.words}
        scores_list.append(scores)
    return scores_list


def tfidf_scikit(docslist, _logger=None):
    """Create a matrix of tfidf scores using scikit-learn's implementation

    Args:
     docslist:
     _logger (optional): instance of the logger

    Returns:
         tfidf(np.array): np array of tfidf scores
         tfidf_vectorizer: TfidfVectorizer (to extracts words)
    """
    tokenize = lambda doc: doc.split(" ")  # text was already pre-processed, otherwise it could be done here
    n_features = 1000  # n of words

    tfidf_vectorizer = TfidfVectorizer(norm='l2', min_df=2, max_df=0.95, use_idf=True, smooth_idf=False,
                                       sublinear_tf=True, tokenizer=tokenize, max_features=n_features,
                                       stop_words='english')
    tfidf = tfidf_vectorizer.fit_transform(docslist)

    return tfidf, tfidf_vectorizer
