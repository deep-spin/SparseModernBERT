import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
from sentence_transformers.models import Transformer, Pooling
from mteb import MTEB

import sys
sys.path.append("..")

from sparse_modern_bert import CustomModernBertModel


class SparseTransformer(Transformer):
    def _load_model(self, model_name_or_path, config, cache_dir, *args, **model_args) -> None:
        config.attn_implementation="eager"
        self.auto_model = CustomModernBertModel.from_pretrained(
            model_name_or_path, config=config, cache_dir=cache_dir, attn_implementation="eager", **model_args
        )


def plot_attention_map(attn_weights, layer, head, tokens, output_path):
    """Plot and save the attention map for a specific layer and head."""
    fig, ax = plt.subplots(figsize=(10, 8))
    cax = ax.matshow(attn_weights, cmap="viridis")
    fig.colorbar(cax)

    ax.set_xticks(range(len(tokens)))
    ax.set_yticks(range(len(tokens)))

    ax.set_xticklabels(tokens, rotation=90, fontsize=10)
    ax.set_yticklabels(tokens, fontsize=10)

    ax.set_title(f"Attention Map - Layer {layer}, Head {head}", pad=20)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)
    print(f"Saved attention map for Layer {layer}, Head {head} to {output_path}")


def process_sample(model, tokenizer, sample, output_dir, layers, heads):
    """Process a single sample and generate attention maps for selected layers and heads."""
    # Ensure the input sample is a valid string
    if not isinstance(sample, str):
        print(f"Invalid input for tokenization: {sample}")
        return

    # Tokenize the sample
    encoded = tokenizer(sample, return_tensors="pt", truncation=True, max_length=128)
    input_ids = encoded["input_ids"]
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    # Extract attention weights from the transformer model
    with torch.no_grad():
        transformer_model = model[0].auto_model  # Access the CustomModernBertModel
        outputs = transformer_model(input_ids, output_attentions=True)
        attentions = outputs.attentions  # List of attention weights for all layers

    for layer in layers:
        for head in heads:
            attn_weights = attentions[layer][0, head].cpu().numpy()  # Extract attention weights
            output_path = f"{output_dir}/attention_layer{layer}_head{head}.png"
            plot_attention_map(attn_weights, layer, head, tokens, output_path)

def load_trec_covid_samples(task_name="TRECCOVID", num_samples=3):
    """Load TREC-COVID queries for visualization."""
    tasks = MTEB(tasks=[task_name]).tasks
    task = tasks[0]
    task.load_data()

    # Take a few sample queries
    queries = list(task.queries['test'].values())[:num_samples]

    # Debug: Print the queries to ensure they are valid strings
    for i, query in enumerate(queries):
        print(f"Query {i + 1}: {query} (type: {type(query)})")

    return queries


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, help="Path to the pretrained model")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save attention maps")
    parser.add_argument("--layers", type=int, nargs="+", default=[0, 6, 11], help="Layers to visualize")
    parser.add_argument("--heads", type=int, nargs="+", default=[0, 1, 2], help="Heads to visualize")
    parser.add_argument("--alpha", type=float, default=2.0, help="Entmax alpha parameter")
    parser.add_argument("--pre_iter", type=int, default=5, help="Number of pre iterations for Entmax")
    parser.add_argument("--post_iter", type=int, default=5, help="Number of post iterations for Entmax")
    args = parser.parse_args()

    # Model arguments
    model_args = {
        "alpha": args.alpha,
        "use_triton_entmax": False,
        "pre_iter": args.pre_iter,
        "post_iter": args.post_iter,
    }

    # Load SparseTransformer and SentenceTransformer
    transformer = SparseTransformer(args.model_name, model_args=model_args)
    pooling = Pooling(transformer.get_word_embedding_dimension(), pooling_mode="mean")
    model = SentenceTransformer(modules=[transformer, pooling])
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Load TREC-COVID samples
    queries = load_trec_covid_samples()

    # Process each query
    for i, query in enumerate(queries):
        print(f"Processing sample {i + 1}:")
        process_sample(model, tokenizer, query, args.output_dir, args.layers, args.heads)
        if i >= 0:
            print("=" * 70)
            break
