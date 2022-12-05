import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm.notebook import tqdm
from wordcloud import WordCloud, STOPWORDS

from sentiment_model.data_utils.tweet_dataset import tokenize, process_token_list
from inference.run_inference import scrape_and_predict
from utils import get_project_root, pickle_save, pickle_load
from wordcloud import WordCloud, STOPWORDS
from pathlib import Path
from sentiment_model.data_utils.tweet_dataset import tokenize, process_token_list
import os


def create_test_df(preds, targets, softmax_output=False):
    if softmax_output:
        preds = np.argmax(preds, axis=1)
    labels = ['negative', 'neutral', 'positive']
    text_labels = [labels[int(i)] for i in preds]
    target_labels = [labels[int(i)] for i in targets]

    return pd.DataFrame({"target": target_labels, "predictions": text_labels})


def plot_bar(df, x_col, y_col):
    fig, ax = plt.subplots(1, 1)
    ax.plot(df[x_col], df[y_col])

    return


def plot_label_count(df):
    counts = [np.sum(np.array(df['target']) == i) for i in np.unique(df['target'])]

    fig, ax = plt.subplots(1, 1)

    ax.bar(np.unique(df['target']), counts)


def plot_confusion_matrix(cm):
    fig, ax = plt.subplots(1, 1)
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', ax=ax)
    labels = ['negative', 'neutral', 'positive']
    ax.set_title('Seaborn Confusion Matrix with labels\n\n')
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values ')
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)


def plot_sentiment(tweet_df):
    sent_cl = np.unique(tweet_df["sentiment_label"])

    counts = [np.sum(np.array(tweet_df['sentiment_label']) == cl) for cl in sent_cl]

    fig, ax = plt.subplots(1, 1)

    ax.bar(sent_cl, counts)
    return


def plot_text_lengths(tweet_df):
    fig, ax = plt.subplots(1, 1)
    sns.kdeplot(tweet_df['text_length'][tweet_df['sentiment_label'] == 'negative'], fill=True, color="r", ax=ax,
                label="negative")
    sns.kdeplot(tweet_df['text_length'][tweet_df['sentiment_label'] == 'neutral'], fill=True, color="b", ax=ax,
                label="neutral")
    sns.kdeplot(tweet_df['text_length'][tweet_df['sentiment_label'] == 'positive'], fill=True, color="g", ax=ax,
                label="positive")
    ax.set_xlim(-10, 500)
    ax.legend()
    ax.set_title("Text length per sentiment label")


def plot_date_dist(tweet_df):
    fig, ax = plt.subplots(1, 1)
    sns.histplot(tweet_df['date'], ax=ax, color="#4472c4")

    ax.tick_params(axis='x', rotation=60)
    ax.set_title("Tweet frequency over time")
    return fig


def plot_stacked_bar(tweet_df, by="month", title=None):
    def get_percentage(x):
        return x[0] / totals[x['date']]

    if by == "week":
        w = 5
        tweet_df_week = tweet_df.copy()
        tweet_df_week['date'] = tweet_df_week['date'] - pd.to_timedelta(tweet_df_week['date'].dt.dayofweek, unit='d')
        sentiment_week = pd.DataFrame(
            tweet_df_week.groupby([tweet_df_week.date.dt.year, tweet_df_week.date.dt.month, tweet_df_week.date.dt.day,
                                   'sentiment_label']).size())

        datetime_col = pd.to_datetime({'year': [i[0] for i in list(sentiment_week.index)],
                                       'month': [i[1] for i in list(sentiment_week.index)],
                                       'day': [i[2] for i in list(sentiment_week.index)]})

        sentiments = [i[3] for i in list(sentiment_week.index)]
        sentiment_week = sentiment_week.reset_index(drop=True)
        sentiment_week["date"] = datetime_col
        sentiment_week["sentiment"] = sentiments

        totals = sentiment_week.groupby("date")[[0]].sum(numeric_only=True)[0]

        sentiment_week['ratio'] = sentiment_week.apply(get_percentage, axis=1)

        week_array = np.repeat(sentiment_week['date'].unique(), 3)
        sent_array = ['negative', 'neutral', 'positive'] * len(sentiment_week['date'].unique())

        df_weekly_sentiment = pd.DataFrame(
            {"date": week_array, "sentiment": sent_array, "ratio": np.zeros(len(week_array))})

        def get_ratio(x):
            ratio = \
                sentiment_week[(sentiment_week['sentiment'] == x['sentiment']) & (sentiment_week['date'] == x['date'])][
                    'ratio']
            if not len(ratio) == 0:
                return ratio.iloc[0]

            else:
                return 0

        df_weekly_sentiment['ratio'] = df_weekly_sentiment.apply(get_ratio, axis=1)
        df_final_sentiment = df_weekly_sentiment.sort_values('date', ignore_index=True)

    elif by == "month":
        w = 10
        sentiment_month = pd.DataFrame(
            tweet_df.groupby([tweet_df.date.dt.year, tweet_df.date.dt.month, 'sentiment_label']).size())

        datetime_col = pd.to_datetime({'year': [i[0] for i in list(sentiment_month.index)],
                                       'month': [i[1] for i in list(sentiment_month.index)],
                                       'day': [1 for _ in range(len(sentiment_month.index))]})

        sentiments = [i[2] for i in list(sentiment_month.index)]
        sentiment_month = sentiment_month.reset_index(drop=True)
        sentiment_month["date"] = datetime_col
        sentiment_month["sentiment"] = sentiments

        totals = sentiment_month.groupby("date")[[0]].sum(numeric_only=True)[0]

        sentiment_month['ratio'] = sentiment_month.apply(get_percentage, axis=1)

        month_array = np.repeat(sentiment_month['date'].unique(), 3)
        sent_array = ['negative', 'neutral', 'positive'] * len(sentiment_month['date'].unique())

        df_montly_sentiment = pd.DataFrame(
            {"date": month_array, "sentiment": sent_array, "ratio": np.zeros(len(month_array))})

        def get_ratio(x):
            ratio = \
            sentiment_month[(sentiment_month['sentiment'] == x['sentiment']) & (sentiment_month['date'] == x['date'])][
                'ratio']
            if not len(ratio) == 0:
                return ratio.iloc[0]

            else:
                return 0

        df_montly_sentiment['ratio'] = df_montly_sentiment.apply(get_ratio, axis=1)
        df_final_sentiment = df_montly_sentiment.sort_values('date', ignore_index=True)

    elif by == "year":
        w = 120
        sentiment_year = pd.DataFrame(
            tweet_df.groupby([tweet_df.date.dt.year, 'sentiment_label']).size())

        # datetime_col = [i[0] for i in list(sentiment_year.index)]
        datetime_col = pd.to_datetime({'year': [i[0] for i in list(sentiment_year.index)],
                                       'month': [1 for _ in list(sentiment_year.index)],
                                       'day': [1 for _ in range(len(sentiment_year.index))]})

        sentiments = [i[1] for i in list(sentiment_year.index)]
        sentiment_year = sentiment_year.reset_index(drop=True)
        sentiment_year["date"] = datetime_col
        sentiment_year["sentiment"] = sentiments

        totals = sentiment_year.groupby("date")[[0]].sum(numeric_only=True)[0]

        sentiment_year['ratio'] = sentiment_year.apply(get_percentage, axis=1)

        year_array = np.repeat(sentiment_year['date'].unique(), 3)
        sent_array = ['negative', 'neutral', 'positive'] * len(sentiment_year['date'].unique())

        df_sentiment_year = pd.DataFrame(
            {"date": year_array, "sentiment": sent_array, "ratio": np.zeros(len(year_array))})

        def get_ratio(x):
            ratio = \
            sentiment_year[(sentiment_year['sentiment'] == x['sentiment']) & (sentiment_year['date'] == x['date'])][
                'ratio']
            if not len(ratio) == 0:
                return ratio.iloc[0]

            else:
                return 0

        df_sentiment_year['ratio'] = df_sentiment_year.apply(get_ratio, axis=1)
        df_final_sentiment = df_sentiment_year.sort_values('date', ignore_index=True)

    negative = df_final_sentiment[df_final_sentiment['sentiment'] == 'negative']['ratio'].to_numpy()
    neutral = df_final_sentiment[df_final_sentiment['sentiment'] == 'neutral']['ratio'].to_numpy()
    positive = df_final_sentiment[df_final_sentiment['sentiment'] == 'positive']['ratio'].to_numpy()
    dates = df_final_sentiment[df_final_sentiment['sentiment'] == 'negative']['date'].to_list()

    fig, ax = plt.subplots(1, 1, figsize=(15, 5))
    ax.bar(dates, negative, width=w, color='r')
    ax.bar(dates, neutral, width=w, bottom=negative, color='#4472c4')
    ax.bar(dates, positive, width=w, bottom=negative + neutral, color='g')
    ax.set_ylabel("Ratio")

    if title is not None:
        ax.set_title(title)

    else:
        ax.set_title(f"Ratio sentiment Tweets by {by}")

    plt.show()
    return df_final_sentiment, fig


# def get_word_cloud_text(tweet_df):
#     tweet_tokens = []
#     for tweet in tqdm(tweet_df['content']):
#         tweet_tokens.extend(tokenize(tweet))
#
#     return " ".join(token for token in tweet_tokens)


def get_word_cloud_text(tweet_df):
    all_tweets = " ".join(tweet for tweet in tweet_df['content'])
    all_tweets_tokens = process_token_list(tokenize(all_tweets))
    return " ".join(token for token in all_tweets_tokens)


def plot_word_cloud(tweet_df, keyword, sentiment_class=None):
    stopwords = set(STOPWORDS)
    stopwords.update({"<url>", "<user>", "user", "s", "url", "hashtag", "number", "n't", "now"})
    if type(keyword) == list:
        for word in keyword:
            if len(word.split()) > 1:
                stopwords.update({"".join(word.split())})

            for kw in word.split():
                stopwords.update({kw})

    else:
        stopwords.update(keyword)
    word_cloud = WordCloud(stopwords=stopwords, min_word_length=3, width=800, height=400)

    if not sentiment_class:
        text = get_word_cloud_text(tweet_df)
    else:
        text = get_word_cloud_text(tweet_df[tweet_df['sentiment_label'] == sentiment_class])

    word_cloud.generate(text)

    fig, ax = plt.subplots(1, 1, figsize=(20, 40))
    ax.imshow(word_cloud, interpolation='bilinear')
    ax.axis("off")

    plt.show()
    return fig


if __name__ == "__main__":
    keyword = ['vegan', 'veganism', 'vegetarian', 'plant based', 'veggie']
    df_name = "_".join(keyword) if type(keyword) == list else keyword
    df_save_path = os.path.join(get_project_root(), Path(f"sentiment_model/checkpoints/scraped_dataset/df_{df_name}.p"))
    tweet_df = pickle_load(df_save_path)
    # tweet_df = tweet_df.sort_values(by="date").reset_index(drop=True)[1000:]
    plot_stacked_bar(tweet_df, by="year")
