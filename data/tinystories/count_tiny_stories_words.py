from collections import Counter
from datasets import load_dataset  # huggingface datasets
from tqdm import tqdm
import re


def get_unique_words(dataset):
    unique_words = set()

    for label in dataset:
        for example in tqdm(dataset[label]):
            text = example["text"]
            words = [re.sub(r"[^a-zA-Z0-9]", "", word) for word in text.split()]
            unique_words.update(words)

    return list(unique_words)


def main():

    if True:  # True to use debug subset
        data_files = {
            "train": r"..\..\datasets\TinyStories\TinyStoriesV2-GPT4-train-1k-lines.txt",
            "val": r"..\..\datasets\TinyStories\TinyStoriesV2-GPT4-valid-1k-lines.txt",  # Just for testing
        }
    else:
        data_files = {
            "train": r"..\..\datasets\TinyStories\TinyStoriesV2-GPT4-train.txt",  # Full original
            "val": r"..\..\datasets\TinyStories\TinyStoriesV2-GPT4-valid.txt",
        }

    dataset = load_dataset("text", data_files=data_files)

    # Get unique words from the dataset
    unique_words = get_unique_words(dataset)

    # Print the list of unique words
    print("Unique words in the dataset:")
    for word in sorted(unique_words):
        print(word)

    print("Count of unique words", len(unique_words))


if __name__ == "__main__":
    main()
