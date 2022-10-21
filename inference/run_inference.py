import numpy as np
import torch
from datasets import load_dataset
from torch.nn import Softmax
from torch.utils.data import DataLoader
from torchtext.vocab import GloVe
from tqdm.notebook import tqdm

from model_inference import load_model_for_inference
from sentiment_model.data_utils.tweet_dataset import TweetDataset, pad_batch_inference
from sentiment_model.data_utils.tweet_dataset import TweetDatasetInference

from data_collection.tweet_scraping import get_tweets


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


def scrape_and_predict():
    # Set device and load pretrained model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    sentiment_net = load_model_for_inference(model_file="vivid-thunder-47/vivid-thunder-47-epoch-7.pth", device=device)

    # Load training dataset to build vocab
    glove_twitter = GloVe(name="twitter.27B", dim=50,)
    tweet_dataset = TweetDataset(split='train', pretrained_vecs=glove_twitter)

    # Scrape tweets and create dataset/dataloader
    tweet_df = get_tweets("Milk", "2020-01-30", "now", max_tweets=1000)
    inf_data = TweetDatasetInference(tweet_df["rawContent"], train_pipeline=tweet_dataset.text_pipeline)
    test_loader = DataLoader(inf_data, batch_size=1024, shuffle=False, collate_fn=pad_batch_inference)

    # Run inference on scarped tweets
    results = run_inference(sentiment_net, test_loader, device)

    class_labels = np.argmax(results, axis=1)
    labels = ['negative', 'neutral', 'positive']
    for i, (tweet, class_label) in enumerate(zip(tweet_df['rawContent'], class_labels)):
        print(tweet)
        print(f"sentiment: {labels[class_label]}\n")

        if i > 100:
            break


if __name__ == "__main__":
    scrape_and_predict()