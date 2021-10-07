# coding=utf-8
import json
import os

from torch.utils.data import Dataset


class CLeQADatasetReader(Dataset):
    def __init__(self, hparams: str = None, type: str = 'train') -> None:
        self.hparams = hparams
        self._batch_size = hparams.batch_size
        self._num_epochs = hparams.num_epochs
        self._buffer_size = hparams.buffer_size
        self._bucket_width = hparams.bucket_width
        self._max_length = hparams.max_length
        self._max_episode_length = hparams.max_episode_length
        self._max_knowledge = hparams.max_knowledge
        self._knowledge_truncate = hparams.knowledge_truncate
        self._dataset_dir = hparams.dataset_dir
        self._pad_to_max = hparams.pad_to_max
        self._bert_dir = hparams.bert_dir
        self._vocab_fname = os.path.join(self._bert_dir, 'vocab.txt')

        self.input_examples = []
        with open(self._dataset_dir + type + '.json', 'r') as fread:
            for line in fread.readlines():
                self.input_examples.append(json.loads(line))
        print('[%s] %d examples is loaded' % (type, len(self.input_examples)))

    def __len__(self):
        return len(self.input_examples)

    def __getitem__(self, index):
        # 数据预处理
        episodes, dictionary = self._load_and_preprocess_all()
        # num_episodes = len(episodes)
        # num_examples = sum([len(episode) for episode in episodes])
        # num_iters = int(num_episodes / self._batch_size)
        # self._dictionary = dictionary

        examples = {'question': [],
                    'response': [],
                    'document': [],
                    'p_knowledge_sentences': [],
                    'q_knowledge_sentences': [],
                    }
        print(episodes[index].keys())
        for idx, example in enumerate(episodes[index]):
            if idx == self._max_episode_length:
                break
            # 抽取q r d字段
            examples['question'].append(example['question'])
            examples['response'].append(example['response'])
            examples['document'].append(example['document'])

        examples['episode_length'] = len(examples['question'])
        return examples

    def _load_and_preprocess_all(self):
        episodes = self.input_examples
        # dictionary = tokenization.FullTokenizer(self._vocab_fname)

        # UNCASED = self._bert_dir
        # VOCAB = 'vocab.txt'
        # dictionary = BertTokenizer.from_pretrained(os.path.join(UNCASED, VOCAB))
        # bert = BertModel.from_pretrained(UNCASED)
        dictionary = ''
        return episodes, dictionary
