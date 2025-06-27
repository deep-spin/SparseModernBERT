import sys

from sentence_transformers import SentenceTransformer
from sentence_transformers.models import Transformer, Pooling

import mteb

# add previous dir to path
sys.path.append("..")
from sparse_modern_bert import CustomModernBertModel


class SparseTransformer(Transformer):
    def _load_model(self, model_name_or_path, config, cache_dir, *args, **model_args) -> None:
        self.auto_model = CustomModernBertModel.from_pretrained(
            model_name_or_path, config=config, cache_dir=cache_dir, **model_args
        )


model_name = "sardinelab/SparseModernBERT-alpha1.5"
model_args = {
    "alpha": 1.5,
    "use_triton_entmax": True,
    "pre_iter": 5,
    "post_iter": 5,
    "reinit_layers": True,
}

transformer = SparseTransformer(model_name, model_args=model_args)
pooling = Pooling(transformer.get_word_embedding_dimension(), pooling_mode="mean")
model = SentenceTransformer(modules=[transformer, pooling])

task_names = ["SciFact", "NFCorpus", "FiQA2018", "TRECCOVID"]
tasks = mteb.get_tasks(tasks=task_names)
evaluation = mteb.MTEB(tasks=tasks)
results = evaluation.run(
    model,
    output_folder=f"./results/",
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
    import ipdb; ipdb.set_trace()