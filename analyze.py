import os, csv, re

from optparse import OptionParser

import datetime, pytz
from dateutil.tz import tzlocal

import itertools

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib.patches as patches
import matplotlib.path as path
import matplotlib.cm as cm

import webbrowser

from collections import Counter

from nltk.cluster import KMeansClusterer, GAAClusterer, euclidean_distance
import nltk.corpus
from nltk import decorators
import nltk.stem

from pytagcloud import create_tag_image, make_tags
from pytagcloud.lang.counter import get_tag_counts
from pytagcloud.lang.stopwords import StopWords
from pytagcloud.colors import COLOR_SCHEMES

stopwords = set(nltk.corpus.stopwords.words('english'))

HEADER = [  'tweet_id', 'in_reply_to_status_id', 'in_reply_to_user_id', 'retweeted_status_id', \
            'retweeted_status_user_id', 'timestamp', 'source', 'text', 'expanded_urls']

HEADER_DICT = dict( (name,i) for i, name in enumerate(HEADER) )

def load_tweets():
    tweets = []
    file_path = "tweets.csv"
    with open(file_path,'r') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',', quotechar='"')
        csvreader.next() # Skip header
        for row in csvreader:
            tweets.append(row)

    print 'Loaded %d tweets' % len(tweets)

    print tweets[:10]

    return tweets

def by_hour(tweets):    
    hours = []
    for tweet in tweets:
        timestamp_str = tweet[ HEADER_DICT['timestamp'] ]
        timestamp = datetime.datetime.strptime(timestamp_str,'%Y-%m-%d %H:%M:%S +0000')
        timestamp = timestamp.replace(tzinfo=pytz.utc)
        timestamp = timestamp.astimezone( tzlocal() )
        hours.append(timestamp.hour)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    n, bins = np.histogram(hours, range(25))

    print n,bins

    # get the corners of the rectangles for the histogram
    left = np.array(bins[:-1])
    right = np.array(bins[1:])
    bottom = np.zeros(len(left))
    top = bottom + n

    # we need a (numrects x numsides x 2) numpy array for the path helper
    # function to build a compound path
    XY = np.array([[left,left,right,right], [bottom,top,top,bottom]]).T

    # get the Path object
    barpath = path.Path.make_compound_path_from_polys(XY)

    # make a patch out of it
    patch = patches.PathPatch(barpath, facecolor='blue', edgecolor='gray', alpha=0.8)
    ax.add_patch(patch)

    # update the view limits
    ax.set_xlim(left[0], right[-1])
    ax.set_ylim(bottom.min(), top.max())

    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.xticks( range(0,24), ha='center' )

    plt.xlabel('Hour')
    plt.ylabel('# Tweets')
    plt.title('# of Tweets by Hour')

    plt.savefig('by-hour.png', bbox_inches=0)
    plt.show()

def by_dow(tweets):
    dow = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
    c = Counter()
    for tweet in tweets:
        timestamp_str = tweet[ HEADER_DICT['timestamp'] ]
        timestamp = datetime.datetime.strptime(timestamp_str,'%Y-%m-%d %H:%M:%S +0000')
        timestamp = timestamp.replace(tzinfo=pytz.utc)
        timestamp = timestamp.astimezone( tzlocal() )
        c[timestamp.strftime('%A')] += 1
    print c.most_common(10)
    
    N = len(dow)

    ind = np.arange(N)
    width = 0.9

    fig = plt.figure()
    ax = fig.add_subplot(111)

    rects1 = ax.bar(0.05+ind, [c[d] for d in dow], width, color='b')

    ax.set_ylabel('# Tweets')
    ax.set_title('Tweets by Day of Week')
    ax.set_xticks(ind + 0.5 * width)
    ax.set_xticklabels( [d[:3] for d in dow] )

    plt.savefig('by-dow.png', bbox_inches=0)
    plt.show()

def by_month(tweets):
    c = Counter()
    for tweet in tweets:
        timestamp_str = tweet[ HEADER_DICT['timestamp'] ]
        timestamp = datetime.datetime.strptime(timestamp_str,'%Y-%m-%d %H:%M:%S +0000')
        timestamp = timestamp.replace(tzinfo=pytz.utc)
        timestamp = timestamp.astimezone( tzlocal() )
        c[timestamp.strftime('%Y-%m')] += 1
    print c.most_common(10)
    
    N = len(c)

    ind = np.arange(N)  # the x locations for the groups
    width = 0.8         # the width of the bars

    fig = plt.figure()
    ax = fig.add_subplot(111)

    rects1 = ax.bar(ind, [ c[x] for x in sorted(c.keys()) ], width, color='b')

    ax.set_ylabel('# Tweets')
    ax.set_title('Tweets by Month')

    ax.set_xticks([ i for i,x in enumerate(sorted(c.keys())) if i % 6 == 0])
    ax.set_xticklabels( [ x for i,x in enumerate(sorted(c.keys())) if i % 6 == 0], rotation=30 )

    plt.savefig('by-month.png', bbox_inches=0)
    plt.show()

def by_month_dow(tweets):
    dow = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
    # Get the # of week and weekday for each tweet
    data = {}
    for tweet in tweets:
        timestamp_str = tweet[ HEADER_DICT['timestamp'] ]
        timestamp = datetime.datetime.strptime(timestamp_str,'%Y-%m-%d %H:%M:%S +0000')
        timestamp = timestamp.replace(tzinfo=pytz.utc)
        timestamp = timestamp.astimezone( tzlocal() )
        weekday = timestamp.strftime('%A')
        iso_yr, iso_wk, iso_wkday = timestamp.isocalendar()
        key = str(iso_yr) + '-' + str(iso_wk)
        key = timestamp.strftime('%Y-%m')
        if key  not in data:
            data[key] = Counter()
        data[key][weekday] += 1
    print data
    # Convert to numpy
    xs = []
    ys = []
    a = np.zeros( (7, len(data)) )
    for i,key in enumerate(sorted(data.iterkeys())):
        for j,d in enumerate(dow):
            a[j,i] = data[key][d]
            for k in range(data[key][d]):
                xs.append(j)
                ys.append(i)
    #Convert to x,y pairs
    heatmap, xedges, yedges = np.histogram2d(np.array(xs), np.array(ys), bins=(7,len(data)))
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    plt.clf()
    plt.imshow(heatmap, extent=extent)
    plt.show()

    x = np.array(xs)
    y = np.array(ys)

    gridsize=30
    plt.hexbin(x, y, C=None, gridsize=gridsize, cmap=cm.jet, bins=None)
    plt.axis([x.min(), x.max(), y.min(), y.max()])

    plt.title('Tweets by Day of Week and Month')
    plt.xlabel('Day of Week')
    plt.ylabel('Month')
    plt.gca().set_xticklabels( [d[:3] for d in dow] )
    plt.gca().set_yticklabels( [key for i,key in enumerate(sorted(data.iterkeys())) if i % 6 == 0] )
    plt.gca().set_yticks([i for i,key in enumerate(sorted(data.iterkeys())) if i % 6 == 0])

    print [key for key in sorted(data.iterkeys())]

    cb = plt.colorbar()
    cb.set_label('# Tweets')

    plt.savefig('by-month-dow.png', bbox_inches=0)
    plt.show()

def by_month_length(tweets):
    c = Counter()
    s = Counter()
    for tweet in tweets:
        timestamp_str = tweet[ HEADER_DICT['timestamp'] ]
        timestamp = datetime.datetime.strptime(timestamp_str,'%Y-%m-%d %H:%M:%S +0000')
        timestamp = timestamp.replace(tzinfo=pytz.utc)
        timestamp = timestamp.astimezone( tzlocal() )
        c[timestamp.strftime('%Y-%m')] += 1
        s[timestamp.strftime('%Y-%m')] += len(tweet[ HEADER_DICT['text'] ])
    print c.most_common(10)
    
    N = len(c)
    ind = np.arange(N)
    width = 0.8

    fig = plt.figure()
    ax = fig.add_subplot(111)

    rects1 = ax.bar(ind, [ s[x]/c[x] for x in sorted(c.keys()) ], width, color='b')

    ax.set_ylabel('Avg Tweet Length')
    ax.set_title('Avg Tweet Length by Month')

    ax.set_xticks([ i for i,x in enumerate(sorted(c.keys())) if i % 6 == 0])
    ax.set_xticklabels( [ x for i,x in enumerate(sorted(c.keys())) if i % 6 == 0], rotation=30 )

    plt.savefig('by-month-length.png', bbox_inches=0)
    plt.show()

def by_month_type(tweets):
    c_total   = Counter()
    c_tweets  = Counter()
    c_rts     = Counter()
    c_replies = Counter()
    months  = set()
    for tweet in tweets:
        timestamp_str = tweet[ HEADER_DICT['timestamp'] ]
        timestamp = datetime.datetime.strptime(timestamp_str,'%Y-%m-%d %H:%M:%S +0000')
        timestamp = timestamp.replace(tzinfo=pytz.utc)
        timestamp = timestamp.astimezone( tzlocal() )
        key = timestamp.strftime('%Y-%m')
        months.add(key)
        c_total[key] += 1
        if tweet[ HEADER_DICT['in_reply_to_status_id'] ]:
            c_replies[key] += 1
        elif tweet[ HEADER_DICT['retweeted_status_id'] ]:
            c_rts[key] += 1
        else:
            c_tweets[key] += 1

    months = [x for x in sorted(months)]
    N = len(months)
    ind = np.arange(N)

    # Create the non stacked version
    width = 0.3

    fig = plt.figure()
    ax = fig.add_subplot(111)

    rects1 = ax.bar(ind, [ c_tweets[m] for m in months ], width, color='r')
    rects2 = ax.bar(ind + width, [ c_rts[m] for m in months ], width, color='b')
    rects3 = ax.bar(ind + width * 2, [ c_replies[m] for m in months ], width, color='g')

    ax.set_ylabel('# Tweets')
    ax.set_title('Type of Tweet by Month')

    ax.set_xticks([ i + width for i,x in enumerate(months) if i % 6 == 0])
    ax.set_xticklabels( [ x for i,x in enumerate(months) if i % 6 == 0], rotation=30 )

    ax.legend( (rects1[0], rects2[0], rects3[0]), ('Tweet', 'RT', 'Reply') )

    fig.set_size_inches(12,6) 
    plt.savefig('by-month-type.png', bbox_inches=0)
    plt.show()

    # Create the stacked version
    width = 0.9

    fig = plt.figure()
    ax = fig.add_subplot(111)

    d_tweets  = np.array([ float(c_tweets[m])/c_total[m] for m in months ])
    d_rts     = np.array([ float(c_rts[m])/c_total[m] for m in months ])
    d_replies = np.array([ float(c_replies[m])/c_total[m] for m in months ])

    rects1 = ax.bar(ind + width/2, d_tweets, width, color='r')
    rects2 = ax.bar(ind + width/2, d_rts, width, bottom=d_tweets, color='b')
    rects3 = ax.bar(ind + width/2, d_replies, width, bottom=d_tweets + d_rts, color='g')

    ax.set_ylabel('Tweet Type %')
    ax.set_title('Type of Tweet by Month')

    ax.set_xticks([ i for i,x in enumerate(months) if i % 6 == 0])
    ax.set_xticklabels( [ x for i,x in enumerate(months) if i % 6 == 0], rotation=30 )

    ax.legend( (rects1[0], rects2[0], rects3[0]), ('Tweet', 'RT', 'Reply'), loc=4 )

    plt.savefig('by-month-type-stacked.png', bbox_inches=0)
    plt.show()
    

@decorators.memoize
def get_words(tweet_text):
    return [word.lower() for word in re.findall('\w+', tweet_text) if len(word) > 3]


def word_frequency(tweets):
    c = Counter()
    hash_c = Counter()
    at_c = Counter()
    s = StopWords()     
    s.load_language("english")
    
    for tweet in tweets:
        for word in get_words( tweet[ HEADER_DICT['text'] ] ):
            if not s.is_stop_word(word):
                if c.has_key(word):
                    c[ word ] += 1
                else:
                    c[ word ] = 1
                    
        for word in re.findall('@\w+', tweet[ HEADER_DICT['text'] ]):
            at_c[ word.lower() ] += 1
        for word in re.findall('\#[\d\w]+', tweet[ HEADER_DICT['text'] ]):
            hash_c[ word.lower() ] += 1
            
    print c.most_common(50)
    print hash_c.most_common(50)
    print at_c.most_common(50)

    #Making word clouds for your most common words, most common @replies and most common #hashtags.

    ctags = make_tags(c.most_common(100), maxsize=90, 
                         colors=COLOR_SCHEMES['audacity'])
    create_tag_image(ctags, 'c_most_common.png', size=(900, 600), fontname='Lobster')
    webbrowser.open('c_most_common.png')

    hash_ctags = make_tags(hash_c.most_common(100), maxsize=100, 
                         colors=COLOR_SCHEMES['citrus'])
    create_tag_image(hash_ctags, 'hash_c_most_common.png', size=(900, 600), fontname='Cuprum')
    webbrowser.open('hash_c_most_common.png')

    at_ctags = make_tags(at_c.most_common(100), maxsize=90)
    create_tag_image(at_ctags, 'at_c_most_common.png', size=(900, 600), fontname='Yanone Kaffeesatz')
    webbrowser.open('at_c_most_common.png')

#Word clusters are still not working. I'm going to get help on this.
    #If you have experience with nltk, feedback is appreciated!
    
def get_word_clusters(tweets):
    ListTweets = get_all_text(tweets)
    cluster = KMeansClusterer(10, euclidean_distance, avoid_empty_clusters = True)
    cluster.cluster([vectorspaced(tweet) for tweet in ListTweets])

    classified_examples = [
        cluster.classify(vectorspaced(tweet)) for tweet in ListTweets
    ]

    for cluster_id, tweet in sorted(zip(classified_examples, ListTweets)):
        print cluster_id, tweet


def get_all_words(tweets):
    for tweet in tweets:
        words = get_words( tweet[ HEADER_DICT['text'] ] )
        for word in words:
            yield word


def get_all_text(tweets):    
    ListTweets = []
    for tweet in tweets:
        ListTweets.append(tweet[ HEADER_DICT['text']])
    return ListTweets


@decorators.memoize
def vectorspaced(tweet_text):
    components = [word.lower() for word in ListWords]
    return np.array([
        word in components and not word in stopwords for word in ListWords])
        
        
if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("-d", "--dir", dest="directory",
                      help="Twitter archive directory - FILE", metavar="FILE")

    (options, args) = parser.parse_args()

    tweets = load_tweets()
    
    by_month(tweets)
    by_month_type(tweets)
    by_month_length(tweets)
    by_month_dow(tweets)
    by_dow(tweets)
    by_hour(tweets)
    word_frequency(tweets)

# Word Clusters are broken still. Better than before, but will only find one cluster.

##    get_word_clusters(tweets)
    
