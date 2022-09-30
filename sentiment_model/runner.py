from data_utils.tweet_dataset import TweetDataset

train_dataset = TweetDataset(split="train")
valid_dataset = TweetDataset(split="valid")
test_dataset = TweetDataset(split="test")

print(f"train data {len(train_dataset)}")
print(f"valid data {len(valid_dataset)}")
print(f"test data {len(test_dataset)}")
print(train_dataset[0])


