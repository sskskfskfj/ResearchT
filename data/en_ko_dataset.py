from datasets import load_dataset

import random



def load_arxiv_dataset():
    dataset = load_dataset("nick007x/arxiv-papers")["train"]["abstract"]
    return dataset


def random_sampling(dataset, stratum = 100, seed=42):
    random.seed(seed)
    dataset_size = len(dataset)
    
    


if __name__ == "__main__":
    dataset = load_arxiv_dataset()
    