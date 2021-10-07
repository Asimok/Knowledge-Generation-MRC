import collections

from torch.utils.data import DataLoader
from tqdm import tqdm

from config.hparams import PARAMS
from data.CLeQA import CLeQADatasetReader

hparams = PARAMS
hparams = collections.namedtuple("HParams", sorted(hparams.keys()))(**hparams)


def build_dataloader():
    train_dataset = CLeQADatasetReader(hparams, type='train')
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=hparams.batch_size,
        num_workers=hparams.workers,
        shuffle=True,
        drop_last=True
    )


# build_dataloader()

train_dataset = CLeQADatasetReader(hparams, type='train')
train_dataloader = DataLoader(
        train_dataset,
        batch_size=hparams.batch_size,
        num_workers=hparams.workers,
        shuffle=True,
        drop_last=True
    )

tqdm_batch_iterator = tqdm(train_dataloader)
for batch_idx, batch in enumerate(tqdm_batch_iterator):
    print(batch)
    break