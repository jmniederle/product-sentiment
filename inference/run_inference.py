import numpy as np
import pandas as pd
import torch
from torch.nn import Softmax
from torch.utils.data import DataLoader
from torchtext.vocab import GloVe
from tqdm.notebook import tqdm
import os
from pathlib import Path

from inference.model_inference import load_model_for_inference
from sentiment_model.data_utils.tweet_dataset import TweetDataset, pad_batch_inference, pad_batch
from sentiment_model.data_utils.tweet_dataset import TweetDatasetInference

from data_collection.tweet_scraping import get_tweets

from utils import get_project_root, pickle_save
from sentiment_model.model_calibration import predict


def run_inference(model, data_loader, device):

    results = []

    soft_max = Softmax(dim=1)

    with torch.no_grad():
        p_bar = tqdm(total=len(data_loader.dataset))
        for batch_idx, (data, text_lengths) in enumerate(data_loader):
            # For assignment 3.2, we need to know the lengths of the targets

            data, text_lengths = data.to(device), text_lengths.to(device)

            # Forward pass
            output = model(data, text_lengths)
            prob_out = soft_max(output, )
            results.extend(prob_out.tolist())
            p_bar.update(data.shape[0])

    return np.array(results)


def scrape_and_predict(keyword="Milk", start_date="2020-01-30", end_date="now", max_tweets=1000, use_calib_model=True,
                       model_args=None, decision_bound=(0.5208333333333334, 0.625)):
    # Load training dataset to build vocab
    cache_path = os.path.join(get_project_root(), Path("sentiment_model/.vector_cache/"))
    glove_twitter = GloVe(name="twitter.27B", dim=50, cache=cache_path)
    tweet_dataset = TweetDataset(split='train', pretrained_vecs=glove_twitter)

    if type(keyword) == list:
        df_list = []
        for kw in keyword:
            # Scrape tweets and create dataset/dataloader
            df_list.append(get_tweets(kw, start_date, end_date, max_tweets=max_tweets))

        tweet_df = pd.concat(df_list, ignore_index=True)
        tweet_df = tweet_df.drop_duplicates(subset="id").reset_index(drop=True)

    else:
        # Scrape tweets and create dataset/dataloader
        tweet_df = get_tweets(keyword, start_date, end_date, max_tweets=max_tweets)

    #pickle_save(tweet_df, "DF_FIND_ERROR.p")

    inf_data = TweetDatasetInference(tweet_df["content"], train_pipeline=tweet_dataset.text_pipeline)
    inf_loader = DataLoader(inf_data, batch_size=1024, shuffle=False, collate_fn=pad_batch)

    # Set device and load pretrained model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if model_args is None:
        model_args = {"vocab_size": 1193516, "embedding_dim": 50, "rnn_hidden_dim": 256, "rnn_n_layers": 2,
                      "rnn_bidirectional": True, "dropout_rate": 0.5, "num_classes": 2, "pretrained_embeddings": None,
                      "freeze_embed": True}

    if use_calib_model:
        calibrated_net = load_model_for_inference(model_args=model_args, device=device, use_calibrated=use_calib_model)
        inf_out = torch.tensor(predict(calibrated_net, inf_data, decision_bound)).to(device)

        class_labels = inf_out.cpu().numpy()
        labels = ['negative', 'neutral', 'positive']
        text_labels = [labels[l] for l in class_labels]

        tweet_df['sentiment_label'] = text_labels
        tweet_df['text_length'] = [len(text) for text in tweet_df['content']]

        return class_labels, text_labels, tweet_df

    else:
        sentiment_net = load_model_for_inference(model_file="vivid-thunder-47/vivid-thunder-47-epoch-7.pth", device=device)
        # Run inference on scraped tweets
        results = run_inference(sentiment_net, inf_loader, device)

        class_labels = np.argmax(results, axis=1)
        labels = ['negative', 'neutral', 'positive']
        text_labels = [labels[l] for l in class_labels]
        # for i, (tweet, class_label) in enumerate(zip(tweet_df['rawContent'], class_labels)):
        #     print(tweet)
        #     print(f"sentiment: {labels[class_label]}\n")

        return class_labels, text_labels, tweet_df


if __name__ == "__main__":
    keyword = ['qatar', 'world cup']
    sentiment, text_sentiment, tweet_df = scrape_and_predict(keyword=keyword, start_date="2021-01-1", end_date="now", max_tweets=1000)
