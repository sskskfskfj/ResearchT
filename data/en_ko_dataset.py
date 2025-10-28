from datasets import load_dataset

import pandas as pd
import random
import os


def get_dataset():
    dataset = load_dataset("lemon-mint/korean_parallel_sentences_v1.1")
    dataset = dataset["train"].train_test_split(test_size=0.2, seed=42)
    
    return dataset


if __name__ == "__main__":
    dataset = get_dataset()
    print(len(dataset["train"]))
    print(len(dataset["test"]))