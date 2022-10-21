import pandas as pd
import snscrape.modules.twitter as sntwitter
from tqdm import tqdm


def get_tweets(keyword, start_date, end_date, lang='en', max_tweets=1000):
    attributes = ['id', 'date', 'rawContent', 'sourceLabel', 'user']
    tweet_list = []

    # can use from: since: lang: until:
    p_bar = tqdm(total=max_tweets)

    scrape_str = "{} lang:{} since:{} until:{}".format(keyword, lang, start_date, end_date)
    # 'Milk lang:en since:2020-01-01 until:now'
    for i, tweet in enumerate(sntwitter.TwitterSearchScraper(scrape_str).get_items()):

        if i > max_tweets:
            break

        tweet_list.append([tweet.__dict__[a] if a != "user" else tweet.__dict__[a].username \
                           for a in attributes])

        p_bar.update(1)

    tweet_df = pd.DataFrame(tweet_list, columns=attributes)
    return tweet_df
