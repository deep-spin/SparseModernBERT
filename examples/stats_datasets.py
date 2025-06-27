import argparse
import numpy as np
from mteb import MTEB
from transformers import AutoTokenizer
from tqdm import tqdm


def get_token_length_stats(texts, tokenizer):
    """Compute statistics on token lengths given a list of texts."""
    lengths = []
    for text in tqdm(texts, desc="Tokenizing texts", total=len(texts)):
        tokens = tokenizer.tokenize(text)
        lengths.append(len(tokens))
    if not lengths:
        return {"num_samples": 0, "mean_tokens": 0, "std_tokens": 0, "min_tokens": None, "max_tokens": None}

    lengths = np.array(lengths)
    return {
        "num_samples": len(lengths),
        "mean_tokens": float(np.mean(lengths)),
        "std_tokens": float(np.std(lengths)),
        "min_tokens": int(np.min(lengths)),
        "max_tokens": int(np.max(lengths)),
        "longer_than_512": np.sum(lengths > 512),
        "longer_than_1024": np.sum(lengths > 1024),
        "longer_than_2048": np.sum(lengths > 2048),
        "longer_than_4096": np.sum(lengths > 4096),
        "longer_than_8192": np.sum(lengths > 8192),
    }


def process_task(task, tokenizer):
    """Process and print statistics for a single task."""
    print(f"Processing task: {task.metadata.name}")

    try:
        # Ensure the dataset is loaded
        task.load_data()
    except Exception as e:
        print(f"Error loading task {task.metadata.name}: {e}")
        return

    # Ensure the dataset is available
    if not hasattr(task, "corpus") or not hasattr(task, "queries") or not hasattr(task, "relevant_docs"):
        print(f"Task {task.metadata.name} did not load corpus/queries/qrels. Skipping...")
        return

    # Collect corpus and queries

    for split in ["train", "dev", "test"]:
        if split not in task.corpus.keys():
            continue

        # Get the texts for the corpus and queries
        queries_texts = list(task.queries[split].values())
        corpus_texts = list(task.corpus[split].values())

        # Print split
        print(f"Split: {split}")

        # Filter out empty strings
        print('Before filter:', len(queries_texts), len(corpus_texts))
        queries_texts = [text for text in queries_texts if text.strip()]
        corpus_texts = [text for text in corpus_texts if text.strip()]
        print('After filter:', len(queries_texts), len(corpus_texts))

        # Compute statistics for corpus and queries
        query_stats = get_token_length_stats(queries_texts, tokenizer)
        corpus_stats = get_token_length_stats(corpus_texts, tokenizer)

        # Print statistics
        print("  -> Queries:")
        print(f"     #Queries: {query_stats['num_samples']}")
        print(f"     Mean tokens: {query_stats['mean_tokens']:.2f}")
        print(f"     Std tokens:  {query_stats['std_tokens']:.2f}")
        print(f"     Min tokens:  {query_stats['min_tokens']}")
        print(f"     Max tokens:  {query_stats['max_tokens']}")
        print(f"     Longer than 512:  {query_stats['longer_than_512']}")
        print(f"     Longer than 1024: {query_stats['longer_than_1024']}")
        print(f"     Longer than 2048: {query_stats['longer_than_2048']}")
        print(f"     Longer than 4096: {query_stats['longer_than_4096']}")
        print(f"     Longer than 8192: {query_stats['longer_than_8192']}")
        print()

        print("  -> Documents:")
        print(f"     #Docs: {corpus_stats['num_samples']}")
        print(f"     Mean tokens: {corpus_stats['mean_tokens']:.2f}")
        print(f"     Std tokens:  {corpus_stats['std_tokens']:.2f}")
        print(f"     Min tokens:  {corpus_stats['min_tokens']}")
        print(f"     Max tokens:  {corpus_stats['max_tokens']}")
        print(f"     Longer than 512:  {corpus_stats['longer_than_512']}")
        print(f"     Longer than 1024: {corpus_stats['longer_than_1024']}")
        print(f"     Longer than 2048: {corpus_stats['longer_than_2048']}")
        print(f"     Longer than 4096: {corpus_stats['longer_than_4096']}")
        print(f"     Longer than 8192: {corpus_stats['longer_than_8192']}")
        print()


if __name__ == "__main__":
    # Argument parser for selecting the tokenizer
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=str, default="bert-base-uncased")
    args = parser.parse_args()

    # Load MTEB tasks
    task_names = ["SciFact", "NFCorpus", "FiQA2018", "TRECCOVID"]
    tasks = MTEB(tasks=task_names).tasks

    # Initialize the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    # Process each task
    for task in tasks:
        print("=" * 70)
        process_task(task, tokenizer)
        print("=" * 70)
