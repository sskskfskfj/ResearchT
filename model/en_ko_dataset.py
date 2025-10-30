from datasets import load_dataset

import pandas as pd
import random
import os
import random

def get_dataset():
    dataset = load_dataset("lemon-mint/korean_parallel_sentences_v1.1")
    sample_size = len(dataset["train"]) * 0.3
    processed_dataset = dataset["train"].shuffle(seed = 42).select(range(int(sample_size)))

    return processed_dataset


if __name__ == "__main__":
    dataset = get_dataset()
    print(dataset)
    # print(len(dataset["train"]))
    # print(len(dataset["test"]))