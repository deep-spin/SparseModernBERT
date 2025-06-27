# Copyright 2024 onwards Answer.AI, LightOn, and contributors
# License: Apache-2.0

import argparse
import sys
from collections import defaultdict

from datasets import load_dataset
from pylate import losses, models, utils, evaluation, indexes, retrieve
from sentence_transformers import (
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.models import Transformer, Pooling

# add previous dir to path
sys.path.append("..")
from sparse_modern_bert import CustomModernBertModel


class SparseTransformer(Transformer):
    def _load_model(self, model_name_or_path, config, cache_dir, *args, **model_args) -> None:
        self.auto_model = CustomModernBertModel.from_pretrained(
            model_name_or_path, config=config, cache_dir=cache_dir, **model_args
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=8e-5)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--accum_steps", type=int, default=1)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--model_name", type=str, default="answerdotai/ModernBERT-base")
    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument("--use_triton_entmax", action="store_true")
    parser.add_argument("--pre_iter", type=int, default=5)
    parser.add_argument("--post_iter", type=int, default=5)
    parser.add_argument("--output_dir", type=str, default="/mnt/scratch-artemis/mtreviso/sparse_pretraining/modernbert")
    args = parser.parse_args()

    # Load the datasets required for knowledge distillation (train, queries, documents)
    train = load_dataset(
        path="lightonai/ms-marco-en-bge",
        name="train",
    )

    queries = load_dataset(
        path="lightonai/ms-marco-en-bge",
        name="queries",
    )

    documents = load_dataset(
        path="lightonai/ms-marco-en-bge",
        name="documents",
    )

    # Set the transformation to load the documents/queries texts using the corresponding ids on the fly
    train.set_transform(
        utils.KDProcessing(queries=queries, documents=documents).transform,
    )

    # Define the base model, training parameters, and output directory
    num_train_epochs = args.num_train_epochs
    lr = args.lr
    batch_size = args.batch_size
    accum_steps = args.accum_steps

    model_name = args.model_name
    model_shortname = f'SparseModernBERT-alpha{args.alpha}'
    run_name = model_shortname
    output_dir = args.output_dir

    # 1. Load a model to finetune
    print('1. Loading model...')
    # model = SentenceTransformer(model_name)
    transformer = SparseTransformer(
        model_name,
        model_args={
            "alpha": args.alpha,
            "use_triton_entmax": args.use_triton_entmax,
            "pre_iter": args.pre_iter,
            "post_iter": args.post_iter,
        }
    )
    # pooling = Pooling(transformer.get_word_embedding_dimension(), pooling_mode="mean")

    # Initialize the ColBERT model from the base model
    print('2. Initializing ColBERT model...')
    model = models.ColBERT(
        modules=[transformer],
    )
    print('Document length:', model.document_length)

    # Configure the training arguments (e.g., epochs, batch size, learning rate)
    print('3. Configuring training arguments...')
    training_args = SentenceTransformerTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=batch_size,
        fp16=args.fp16,  # Set to False if you get an error that your GPU can't run on FP16
        bf16=False,  # Set to True if you have a GPU that supports BF16
        run_name=run_name,
        logging_steps=10,
        learning_rate=lr,
        gradient_accumulation_steps=accum_steps,
        warmup_ratio=0.05,
        save_strategy="steps",
        save_steps=1000,
        save_total_limit=1,
    )
    print('Training arguments:', training_args)

    # Use the Distillation loss function for training
    train_loss = losses.Distillation(model=model)

    # Initialize the trainer
    print('4. Initializing trainer...')
    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train,
        loss=train_loss,
        data_collator=utils.ColBERTCollator(tokenize_fn=model.tokenize),
    )

    # Start the training process
    print('5. Starting training...')
    trainer.train()

    # 6. Save the final model
    print('6. Saving final model...')
    model.save_pretrained(f"{output_dir}/final")

    # 7. Evaluate the model
    print('7. Evaluating model...')
    run_eval(model, model_name, output_dir)


def run_eval(model, model_name, output_dir):
    # Run evaluation as in the evaluation_pylate.py script
    eval_datasets = ["scifact", "nfcorpus", "fiqa", "trec-covid"]
    model_results = defaultdict(dict)

    # Set document_length=510 to match the ColBERT model at evaluation time
    model.document_length = 510

    for eval_dataset in eval_datasets:
        index = indexes.Voyager(index_name=eval_dataset, override=True, M=200, ef_construction=500, ef_search=500)

        retriever = retrieve.ColBERT(index=index)

        documents, queries, qrels = evaluation.load_beir(
            dataset_name=eval_dataset,
            split="test",
        )

        batch_size = 500

        documents_embeddings = model.encode(
            sentences=[document["text"] for document in documents],
            batch_size=batch_size,
            is_query=False,
            show_progress_bar=True,
        )

        index.add_documents(
            documents_ids=[document["id"] for document in documents],
            documents_embeddings=documents_embeddings,
        )

        queries_embeddings = model.encode(
            sentences=queries,
            is_query=True,
            show_progress_bar=True,
            batch_size=16,
        )

        scores = retriever.retrieve(queries_embeddings=queries_embeddings, k=100)

        evaluation_scores = evaluation.evaluate(
            scores=scores,
            qrels=qrels,
            queries=queries,
            metrics=["ndcg@10"],
        )
        print(f"{model_name} - {eval_dataset}")
        print(evaluation_scores)
        print("-----------")
        model_results[eval_dataset] = evaluation_scores

    # save model_results as json using pandas
    import pandas as pd
    df = pd.DataFrame(model_results)
    df.to_json(f"{output_dir}/{model_name}_results.json")


if __name__ == "__main__":
    main()
