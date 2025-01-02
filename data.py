from dataclasses import dataclass
from tqdm import tqdm
import datasets
from torch.utils.data import Dataset
import torch
from transformers import PreTrainedTokenizer, DataCollatorWithPadding


class IndexingTrainDataset(Dataset):
    def __init__(
            self,
            path_to_data,
            max_length: int,
            cache_dir: str,
            tokenizer: PreTrainedTokenizer,
            remove_prompt=False,
    ):
        self.train_data = datasets.load_dataset(
            'json',
            data_files=path_to_data,
            ignore_verifications=False,
            cache_dir=cache_dir
        )['train']

        self.max_length = max_length
        self.tokenizer = tokenizer
        self.remove_prompt = remove_prompt
        self.total_len = len(self.train_data)
        self.valid_ids = set()
        for data in tqdm(self.train_data):
            if "list_text_id" in data:
                self.valid_ids.update(tuple([str(text_id) for text_id in data['list_text_id']]))
            else:
                self.valid_ids.add(str(data['text_id']))

    def __len__(self):
        return self.total_len

    def __getitem__(self, item):
        data = self.train_data[item]
        if self.remove_prompt:
            data['text'] = data['text'][9:] if data['text'].startswith('Passage: ') else data['text']
            data['text'] = data['text'][10:] if data['text'].startswith('Question: ') else data['text']
        input_ids = self.tokenizer(data['text'],
                                   return_tensors="pt",
                                   truncation='only_first',
                                   max_length=self.max_length).input_ids[0]
        if "list_text_id" in data:
            return input_ids, [str(text_id) for text_id in data['list_text_id']]
        else:
            return input_ids, str(data['text_id'])


class GenerateDataset(Dataset):
    lang2mT5 = dict(
        ar='Arabic',
        bn='Bengali',
        fi='Finnish',
        ja='Japanese',
        ko='Korean',
        ru='Russian',
        te='Telugu'
    )

    def __init__(
            self,
            path_to_data,
            max_length: int,
            cache_dir: str,
            tokenizer: PreTrainedTokenizer,
    ):
        self.data = []
        with open(path_to_data, 'r') as f:
            for data in f:
                if 'xorqa' in path_to_data:
                    docid, passage, title = data.split('\t')
                    for lang in self.lang2mT5.values():
                        self.data.append((docid, f'Generate a {lang} question for this passage: {title} {passage}'))
                elif 'msmarco' in path_to_data or 'legal' in path_to_data:
                    docid, passage = data.split('\t')
                    self.data.append((docid, f'{passage}'))
                else:
                    raise NotImplementedError(f"dataset {path_to_data} for docTquery generation is not defined.")

        self.max_length = max_length
        self.tokenizer = tokenizer
        self.total_len = len(self.data)


    def __len__(self):
        return self.total_len

    def __getitem__(self, item):
        docid, text = self.data[item]
        input_ids = self.tokenizer(text,
                                   return_tensors="pt",
                                   truncation='only_first',
                                   max_length=self.max_length).input_ids[0]
        return input_ids, int(docid)


@dataclass
class IndexingCollator(DataCollatorWithPadding):
    def __call__(self, features):
        input_ids = [{'input_ids': x[0]} for x in features]
        docids = [x[1] for x in features] # ex: ['6111', '3525'] (len = batch_size)
        inputs = super().__call__(input_ids)

        # Check if the input is a flat list or a nested list
        if all(isinstance(d, str) for d in docids):
            # Case 1: Flat list
            labels = self.tokenizer(docids, padding="longest", return_tensors="pt").input_ids
        elif all(isinstance(d, list) for d in docids):
            # Case 2: Nested list
            labels_list  = [self.tokenizer(d, padding="longest", return_tensors="pt").input_ids for d in docids]
            # labels = torch.stack(labels_list)
            max_cols = max(tensor.size(1) for tensor in labels_list)
            padded_labels_list = [
                torch.nn.functional.pad(tensor, (0, max_cols - tensor.size(1)), value=-100)
                for tensor in labels_list
            ]
            labels = torch.nn.utils.rnn.pad_sequence(padded_labels_list, batch_first=True, padding_value=-100)
        else:
            raise ValueError("Invalid format of docids. Must be a flat list or a nested list.")
        # print("labels", labels)

        # replace padding token id's of the labels by -100 according to https://huggingface.co/docs/transformers/model_doc/t5#training
        labels[labels == self.tokenizer.pad_token_id] = -100
        inputs['labels'] = labels
        return inputs


@dataclass
class QueryEvalCollator(DataCollatorWithPadding):
    def __call__(self, features):
        input_ids = [{'input_ids': x[0]} for x in features]
        labels = [x[1] for x in features]
        inputs = super().__call__(input_ids)

        return inputs, labels
