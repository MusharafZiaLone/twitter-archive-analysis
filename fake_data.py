
import os
import csv
import random
import datetime
import pdb  #   Debugger
import analyze


HEADER = analyze.HEADER
HEADER_DICT = analyze.HEADER_DICT


THIS_DIR = os.path.dirname(os.path.realpath(__file__))


def main(n=100):
    data = create_fake_data(n)
    save_fake_data(data)

    
def create_fake_data(n=100):
    ft = FakeTweet()
    for idx in xrange(n):
        #   Create a fake tweet with new data.
        yield ft()


def save_fake_data(data):
    path = os.path.join(THIS_DIR, 'tweets.csv')
    with open(path, mode='wb') as f:
        writer = csv.DictWriter(f, fieldnames=HEADER)
        #   Write the header
        writer.writerow(dict(zip(HEADER, HEADER)))
        for d in data:
            writer.writerow(d)
    
    
class FakeTweet(object):
    """
    HEADER
    ------
    'tweet_id'
    'in_reply_to_status_id'
    'in_reply_to_user_id'
    'retweeted_status_id'
    'retweeted_status_user_id'
    'timestamp'
    'source'
    'text'
    'expanded_urls'
    """
    
    __slots__ = HEADER + ['categorized_text', 'data']
        
    def __init__(self):
        self.categorized_text = CategorizedText()
        #   @todo   Define these attributes.
        self.source = None
        self.expanded_urls = None
    
    @property
    def text(self):
        #   Get the next text string from the CategorizedText instance.
        return self.categorized_text()

    @property
    def tweet_id(self):
        return random.randrange(1000, 2000)
    
    @property
    def in_reply_to_status_id(self):
        return random.randrange(1000, 2000)

    @property
    def in_reply_to_user_id(self):
        return random.randrange(100, 200)

    @property
    def retweeted_status_id(self):
        return random.randrange(1000, 2000)
        
    @property
    def retweeted_status_user_id(self):
        return random.randrange(100, 200)
    
    @property
    def timestamp(self):
        #   Format: '%Y-%m-%d %H:%M:%S +0000'
        timestamp = datetime.datetime(
            random.randrange(2007, 2014), 
            random.randrange(1, 13),
            random.randrange(1, 29),
            random.randrange(0, 24),
            random.randrange(0, 60),
            random.randrange(0, 60)
        )
        timestamp = timestamp.strftime('%Y-%m-%d %H:%M:%S +0000')
        return timestamp
        
    @property
    def data(self):
        d = {}
        for k in HEADER:
            d[k] = getattr(self, k)
        return d

    def __call__(self):
        return self.data



class CategorizedText(object):
    def __init__(self):
        from nltk.corpus import reuters
        self.corpus = reuters
        self.categories = self.corpus.categories()
        self.cached_paras = {}
        self.text = self.get_text()

    def get_text(self):
        category_paras = None
        random_paras = []
        category = None
        last = None
        while True:
            cat_count = len(random_paras)
            if not cat_count:
                while category == last: category = random.choice(self.categories)
                last = category
                if category in self.cached_paras:
                    category_paras = self.cached_paras[category]
                else:
                    category_paras = self.corpus.paras(categories=category)
                    self.cached_paras[category] = category_paras
                #   Only select two tweets in a row from the same category
                random_paras = random.sample(category_paras, 2)
            sentences = random_paras.pop()
            words = list(flatten(sentences))
            if len(words) > 25: words = random.sample(words, 25)
            text = ' '.join(words)
            yield text
    
    def __call__(self):
        return self.text.next()
            


def flatten(iterable):
    """
    Flattens a nested list or tuple until at least one subsequence is at a
    single level of nesting. If all subsequence have the same level of
    nesting this will return a flat list. (Note: All subsequences must be of the 
    same type.)
    
    :param iterable: :type list or tuple:   Iterable to flatten. 
    """
    while True:
        try:
            if isinstance(iterable, list):      result = list()
            elif isinstance(iterable, tuple):   result = tuple()
            else:                               break
            iterable = sum(iterable, result)
            if not iterable:
                #   Obtained an empty list or tuple after flattening.
                break
        #   If the sub-types cannot be concatenated to a list or tuple, then break.
        except TypeError as type_error:
            break
    return iter(iterable)
    
    