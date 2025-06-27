# Copyright 2024 onwards Answer.AI, LightOn, and contributors
# License: Apache-2.0

import argparse
import sys
import numpy as np

from datasets import load_dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.models import Transformer, Pooling

from sentence_transformers.evaluation import TripletEvaluator
from sentence_transformers.losses import CachedMultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers

import mteb

# add previous dir to path
sys.path.append("..")
from sparse_modern_bert import CustomModernBertModel
from sparse_roberta import CustomRobertaModel


class SparseTransformer(Transformer):
    def _load_model(self, model_name_or_path, config, cache_dir, *args, **model_args) -> None:
        if 'roberta' in model_name_or_path.lower():
            self.auto_model = CustomRobertaModel.from_pretrained(
                model_name_or_path, config=config, cache_dir=cache_dir, **model_args
            )
        else:
            self.auto_model = CustomModernBertModel.from_pretrained(
                model_name_or_path, config=config, cache_dir=cache_dir, **model_args
            )

def main():
    # parse the lr & model name
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=8e-5)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--model_name", type=str, default="answerdotai/ModernBERT-base")
    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument("--use_triton_entmax", action="store_true")
    parser.add_argument("--pre_iter", type=int, default=5)
    parser.add_argument("--post_iter", type=int, default=5)
    parser.add_argument("--output_dir", type=str, default="/mnt/scratch-artemis/mtreviso/sparse_pretraining/modernbert")
    args = parser.parse_args()

    # /mnt/scratch-artemis/mtreviso/sparse_pretraining/mlm-modernbert-output-100k-finetuned
    # /mnt/scratch-artemis/mtreviso/sparse_pretraining/mlm-modernbert-output-100k-finetuned-alpha15

    model_name = args.model_name
    # model_shortname = model_name.split("/")[-1]

    if 'roberta' in model_name.lower():
        model_shortname = f'SparseRoBERTa-alpha{args.alpha}'
        tokenizer_name_or_path = 'roberta-base'
    else:
        model_shortname = f'SparseModernBERT-alpha{args.alpha}'
        tokenizer_name_or_path = 'answerdotai/ModernBERT-base'
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
        },
        tokenizer_name_or_path=tokenizer_name_or_path,
    )
    pooling = Pooling(transformer.get_word_embedding_dimension(), pooling_mode="mean")
    model = SentenceTransformer(modules=[transformer, pooling])

    # 2. Load a dataset to finetune on
    print('2. Loading dataset...')
    dataset = load_dataset(
        "sentence-transformers/msmarco-co-condenser-margin-mse-sym-mnrl-mean-v1",
        "triplet-hard",
        split="train",
    )
    dataset_dict = dataset.train_test_split(test_size=1_000, seed=12)
    train_dataset = dataset_dict["train"].select(range(1_250_000))
    eval_dataset = dataset_dict["test"]

    # 3. Define a loss function
    print('3. Defining loss function...')
    loss = CachedMultipleNegativesRankingLoss(model, mini_batch_size=16)  # Increase mini_batch_size if you have enough VRAM

    # 4. (Optional) Specify training arguments
    print('4. Specifying training arguments...')
    training_args = SentenceTransformerTrainingArguments(
        # Required parameter:
        output_dir=args.output_dir,
        # Optional training parameters:
        num_train_epochs=1,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        warmup_ratio=0.05,
        fp16=args.fp16,  # Set to False if GPU can't handle FP16
        bf16=False,  # Set to True if GPU supports BF16
        batch_sampler=BatchSamplers.NO_DUPLICATES,  # (Cached)MultipleNegativesRankingLoss benefits from no duplicates
        learning_rate=args.lr,
        # Optional tracking/debugging parameters:
        save_strategy="steps",
        save_steps=1000,
        save_total_limit=1,
        logging_steps=500,
        run_name=model_shortname,  # Used in `wandb`, `tensorboard`, `neptune`, etc. if installed
    )
    print('Training arguments:', training_args)

    # 5. (Optional) Create an evaluator & evaluate the base model
    print('5. Creating evaluator...')
    dev_evaluator = TripletEvaluator(
        anchors=eval_dataset["query"],
        positives=eval_dataset["positive"],
        negatives=eval_dataset["negative"],
        name="msmarco-co-condenser-dev",
    )
    dev_evaluator(model)

    # 6. Create a trainer & train
    print('6. Training...')
    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=loss,
        evaluator=dev_evaluator,
    )
    trainer.train()

    # 7. Save the model
    print('7. Saving model...')
    model.save_pretrained(f"{output_dir}/final")

    # 8. (Optional) Evaluate the trained model on the evaluator after training
    print('8. Evaluating...')
    dev_evaluator(model)

    # 9. (Optional) Push it to the Hugging Face Hub
    # model.push_to_hub(model_shortname, private=False)

    # 10. Run evaluation
    print('10. Running evaluation...')
    run_eval(model, output_dir=output_dir)


def run_eval(model, output_dir='tmp'):
    # Run evaluation as in the evaluation_st.py script

    # output_dir = '/mnt/scratch-artemis/mtreviso/sparse_pretraining/st-modernbert-100k-finetuned-alpha15'
    # model_name = output_dir+'/checkpoint-9766/'
    # transformer = SparseTransformer(
    #     model_name,
    #     model_args={
    #         "alpha": 1.5,
    #         "use_triton_entmax": True,
    #         "pre_iter": 5,
    #         "post_iter": 5,
    #         "reinit_layers": True,
    #     }
    # )
    # pooling = Pooling(transformer.get_word_embedding_dimension(), pooling_mode="mean")
    # model = SentenceTransformer(modules=[transformer, pooling])

    task_names = ["SciFact", "NFCorpus", "FiQA2018", "TRECCOVID"]
    tasks = mteb.get_tasks(tasks=task_names)
    evaluation = mteb.MTEB(tasks=tasks)
    results = evaluation.run(
        model,
        output_folder=f"{output_dir}/results/",
        encode_kwargs={"batch_size": 8},
    )

    # print the results
    print('Results:')
    print(results)
    print('-' * 50)
    pd_results = []

    try:
        for task_results in results:
            task_results_dict = task_results.to_dict()
            task_name = task_results_dict['task_name']
            task_scores_list = task_results_dict['scores']['test']
            print(f"{task_name}:")
            print('-' * 50)
            for i, task_scores in enumerate(task_scores_list):
                print(f"Fold {i}:")
                for metric, score in task_scores.items():
                    print(f"{metric}: {score}")
            print('-' * 50)
            print('')
            pd_results.append({
                'task_name': task_name,
                'task_scores': task_scores_list,
            })

        # save results as json using pandas
        import pandas as pd
        df = pd.DataFrame(pd_results)
        # add indentation to the json file
        df.to_json(f"{output_dir}/results.json", indent=2)

    except Exception as e:
        print(e)
        import ipdb;
        ipdb.set_trace()


if __name__ == "__main__":
    main()