
from optparse import OptionParser
from twitter-archive-analysis import analyze


def main():
    tweets = analyze.load_tweets()    
    analyze.by_month(tweets)
    analyze.by_month_type(tweets)
    analyze.by_month_length(tweets)
    analyze.by_month_dow(tweets)
    analyze.by_dow(tweets)
    analyze.by_hour(tweets)
    analyze.word_frequency(tweets)
    

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("-d", "--dir", dest="directory",
                      help="Twitter archive directory - FILE", metavar="FILE")
    options, args = parser.parse_args()

    