from data import IndexingTrainDataset, GenerateDataset, IndexingCollator, QueryEvalCollator
from transformers import (
    T5Tokenizer,
    T5TokenizerFast,
    T5ForConditionalGeneration,
    TrainingArguments,
    TrainerCallback,
    MT5Tokenizer,
    MT5TokenizerFast,
    MT5ForConditionalGeneration,
    HfArgumentParser,
    set_seed,
)
from trainer import DSITrainer, DocTqueryTrainer
import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader
from dataclasses import dataclass, field
from typing import Optional
import json
from tqdm import tqdm
set_seed(313)


@dataclass
class RunArguments:
    model_name: str = field(default=None)
    model_path: Optional[str] = field(default=None)
    max_length: Optional[int] = field(default=32)
    id_max_length: Optional[int] = field(default=20)
    remove_prompt: Optional[bool] = field(default=False)
    train_file: str = field(default=None)
    valid_file: str = field(default=None)
    task: str = field(default=None,  metadata={"help": "DSI, docTquery, generation"})
    top_k: Optional[int] = field(default=10)
    num_return_sequences: Optional[int] = field(default=10)
    q_max_length: Optional[int] = field(default=32)


# use to evaluate the eval set
def make_compute_metrics(tokenizer, valid_ids):

    # def compute_metrics(eval_preds):
    #     hit_at_1 = 0
    #     hit_at_10 = 0
    #     for beams, label in zip(eval_preds.predictions, eval_preds.label_ids):
    #         # ex: label = np.array([1617, 4305, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    #         rank_list = tokenizer.batch_decode(beams,
    #                                            skip_special_tokens=True)
    #         # ex: rank_list = ['', '3', '6039', '6', '23', ' ', '10', '17', '14', '29', '7', '7', '07', '9', '26', '13', '1', '00', '146', '16']
    #         label_id = tokenizer.decode(label, skip_special_tokens=True)
    #         # ex: label_id = 6039
    #         # filter out duplicates and invalid docids
    #         filtered_rank_list = []
    #         for docid in rank_list:
    #             if docid not in filtered_rank_list and docid in valid_ids:
    #                 filtered_rank_list.append(docid)
    #         # ex: filtered_rank_list = ['6039', '3', '6', '23', '10', '6039', '14', '29', '7', '9', '26', '13', '1', '146', '16']

    #         hits = np.where(np.array(filtered_rank_list)[:10] == label_id)[0]
    #         # ex: = np.array([0, 5])
    #         if len(hits) != 0:
    #             hit_at_10 += 1
    #             if hits[0] == 0:
    #                 hit_at_1 += 1
    #     return {"Hits@1": hit_at_1 / len(eval_preds.predictions), "Hits@10": hit_at_10 / len(eval_preds.predictions)}
    # return compute_metrics

    def compute_metrics(eval_preds):
        
        rank_list_arr = []
        label_ids_arr = []

        for beams, label in zip(eval_preds.predictions, eval_preds.label_ids):
            rank_list = tokenizer.batch_decode(beams,
                                               skip_special_tokens=True)
            if len(label.shape) == 2:
                label_ids = []
                for single_label in label:
                    # Lọc bỏ các giá trị -100 (thường là giá trị pad)
                    filtered_label = [token_id for token_id in single_label if token_id != -100]
                    sub_label_id = tokenizer.decode(filtered_label, skip_special_tokens=True)
                    if sub_label_id != "":
                        label_ids.append(sub_label_id)
            else:
                label_ids = [tokenizer.decode(label, skip_special_tokens=True)]
            print("rank_list", rank_list)
            print("label_ids", label_ids)
            
            rank_list_arr.append(rank_list)
            label_ids_arr.append(label_ids)

        recall_ks = [3, 5, 10, 20, 50, 100, 200]
        mrr_us = [10]
        map_us = [10]
        hits_ts = [1, 10]
        ndcg_us = [10]

        # Initialize metric results
        recall_results = {f"Recall@{k}": 0 for k in recall_ks}
        mrr_results = {f"MRR@{u}": 0 for u in mrr_us}
        map_results = {f"MAP@{u}": 0 for u in map_us}
        hits_results = {f"Hits@{t}": 0 for t in hits_ts}
        ndcg_results = {f"NDCG@{u}": 0 for u in ndcg_us}

        # Total number of queries
        total_queries = len(rank_list_arr)

        # Iterate through each rank list and corresponding label ids
        for rank_list, label_ids in zip(rank_list_arr, label_ids_arr):
            # Convert rank_list and label_ids to sets for quick lookup
            label_ids_set = set(label_ids)

            # Calculate Recall@k
            for k in recall_ks:
                relevant_in_top_k = len(label_ids_set.intersection(rank_list[:k]))
                recall_results[f"Recall@{k}"] += relevant_in_top_k / min(k, len(label_ids))

            # Calculate MRR@u
            for u in mrr_us:
                mrr = 0
                for rank, doc_id in enumerate(rank_list[:u], start=1):
                    if doc_id in label_ids_set:
                        mrr = 1 / rank
                        break
                mrr_results[f"MRR@{u}"] += mrr

            # Calculate MAP@u
            for u in map_us:
                relevant_docs = [1 if doc_id in label_ids_set else 0 for doc_id in rank_list[:u]]
                if sum(relevant_docs) > 0:
                    precision_at_ranks = [
                        sum(relevant_docs[:i + 1]) / (i + 1) for i in range(len(relevant_docs)) if relevant_docs[i] == 1
                    ]
                    map_results[f"MAP@{u}"] += np.mean(precision_at_ranks)

            # Calculate Hits@t
            for t in hits_ts:
                hits_results[f"Hits@{t}"] += int(any(doc_id in label_ids_set for doc_id in rank_list[:t]))

            # Calculate NDCG@u
            for u in ndcg_us:
                # Calculate DCG
                dcg = 0
                for rank, doc_id in enumerate(rank_list[:u], start=1):
                    if doc_id in label_ids_set:
                        dcg += 1 / np.log2(rank + 1)

                # Calculate IDCG
                idcg = 0
                for rank in range(1, min(u, len(label_ids_set)) + 1):
                    idcg += 1 / np.log2(rank + 1)

                # Calculate NDCG
                ndcg_results[f"NDCG@{u}"] += dcg / idcg if idcg > 0 else 0

        # Average metrics over total queries
        recall_results = {k: v / total_queries for k, v in recall_results.items()}
        mrr_results = {k: v / total_queries for k, v in mrr_results.items()}
        map_results = {k: v / total_queries for k, v in map_results.items()}
        hits_results = {k: v / total_queries for k, v in hits_results.items()}
        ndcg_results = {k: v / total_queries for k, v in ndcg_results.items()}

        # Combine all results
        return {**recall_results, **mrr_results, **map_results, **hits_results, **ndcg_results}
    return compute_metrics


def main():

    parser = HfArgumentParser((TrainingArguments, RunArguments))
    training_args, run_args = parser.parse_args_into_dataclasses()

    # We use wandb logger: https://wandb.ai/site.
    if training_args.local_rank == 0:  # only on main process
        # Initialize wandb run
        wandb.login()
        wandb.init(project="DSI", name=training_args.run_name)

    if 'mt5' in run_args.model_name:
        tokenizer = MT5Tokenizer.from_pretrained(run_args.model_name, cache_dir='cache')
        fast_tokenizer = MT5TokenizerFast.from_pretrained(run_args.model_name, cache_dir='cache')
        if run_args.model_path:
            model = MT5ForConditionalGeneration.from_pretrained(run_args.model_path, cache_dir='cache')
        else:
            model = MT5ForConditionalGeneration.from_pretrained(run_args.model_name, cache_dir='cache')
    else:
        tokenizer = T5Tokenizer.from_pretrained(run_args.model_name, cache_dir='cache')
        fast_tokenizer = T5TokenizerFast.from_pretrained(run_args.model_name, cache_dir='cache')
        if run_args.model_path:
            model = T5ForConditionalGeneration.from_pretrained(run_args.model_path, cache_dir='cache')
        else:
            model = T5ForConditionalGeneration.from_pretrained(run_args.model_name, cache_dir='cache')
            
    # Chú trọng vào task docTquery ta có:
    # - lấy max_length, train_dataset và valid_dataset từ tham số truyền vào và xử lý thành định dạng thích hợp để huấn luyện
    # sử dụng tokenizer đã có (trùng với tokenizer của model T5) riêng đối với valid thì có thêm remove_prompt = false
    # * remove_prompt chỉ là việc loại bỏ Question hoặc Passage phía trước của câu đầu vào thôi.
    # - Truyền vào DocTqueryTrainer tất cả dữ liệu + thông số + model + data_collator.
    # * data_collator là longest tức là không thực hiện thêm các padding vào sau.
    if run_args.task == "docTquery":
        train_dataset = IndexingTrainDataset(path_to_data=run_args.train_file,
                                             max_length=run_args.max_length,
                                             cache_dir='cache',
                                             tokenizer=tokenizer)

        valid_dataset = IndexingTrainDataset(path_to_data=run_args.valid_file,
                                             max_length=run_args.max_length,
                                             cache_dir='cache',
                                             remove_prompt=run_args.remove_prompt,
                                             tokenizer=tokenizer)
        trainer = DocTqueryTrainer(
            do_generation=False,
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            data_collator=IndexingCollator(
                tokenizer,
                padding='longest',
            ),
        )
        trainer.train()

    elif run_args.task == "DSI":
        train_dataset = IndexingTrainDataset(path_to_data=run_args.train_file,
                                             max_length=run_args.max_length,
                                             cache_dir='cache',
                                             tokenizer=tokenizer)

        valid_dataset = IndexingTrainDataset(path_to_data=run_args.valid_file,
                                             max_length=run_args.max_length,
                                             cache_dir='cache',
                                             remove_prompt=run_args.remove_prompt,
                                             tokenizer=tokenizer)
        ################################################################
        # docid generation constrain, we only generate integer docids.
        SPIECE_UNDERLINE = "▁"
        INT_TOKEN_IDS = []
        for token, id in tokenizer.get_vocab().items():
            if token[0] == SPIECE_UNDERLINE:
                if token[1:].isdigit():
                    INT_TOKEN_IDS.append(id)
            if token == SPIECE_UNDERLINE:
                INT_TOKEN_IDS.append(id)
            elif token.isdigit():
                INT_TOKEN_IDS.append(id)
        INT_TOKEN_IDS.append(tokenizer.eos_token_id)

        def restrict_decode_vocab(batch_idx, prefix_beam):
            return INT_TOKEN_IDS
        ################################################################

        trainer = DSITrainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            data_collator=IndexingCollator(
                tokenizer,
                padding='longest',
            ),
            compute_metrics=make_compute_metrics(fast_tokenizer, train_dataset.valid_ids),
            restrict_decode_vocab=restrict_decode_vocab,
            id_max_length=run_args.id_max_length
        )
        trainer.train()

    elif run_args.task == 'generation':
        generate_dataset = GenerateDataset(path_to_data=run_args.valid_file,
                                           max_length=run_args.max_length,
                                           cache_dir='cache',
                                           tokenizer=tokenizer)

        trainer = DocTqueryTrainer(
            do_generation=True,
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            data_collator=QueryEvalCollator(
                tokenizer,
                padding='longest',
            ),
        )
        predict_results = trainer.predict(generate_dataset,
                                          top_k=run_args.top_k,
                                          num_return_sequences=run_args.num_return_sequences,
                                          max_length=run_args.q_max_length)
        with open(f"{run_args.valid_file}.q{run_args.num_return_sequences}.docTquery", 'w', encoding='utf-8') as f:
            for batch_tokens, batch_ids in tqdm(zip(predict_results.predictions, predict_results.label_ids),
                                                desc="Writing file"):
                for tokens, docid in zip(batch_tokens, batch_ids):
                    query = fast_tokenizer.decode(tokens, skip_special_tokens=True)
                    jitem = json.dumps({'text_id': docid.item(), 'text': query}, ensure_ascii=False)
                    f.write(jitem + '\n')

    else:
        raise NotImplementedError("--task should be in 'DSI' or 'docTquery' or 'generation'")


if __name__ == "__main__":
    main()