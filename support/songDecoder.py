# -*- coding: utf-8 -*-
import torch
import torchtext
from torchtext.data import Field, BucketIterator
import random
import numpy as np

from support.model import (
    Seq2Seq, Encoder, Decoder,
    tokenize_notes, count_parameters, initialize_weights
)

SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE = 128

SRC = Field(tokenize=tokenize_notes,
            init_token='<sos>',
            eos_token='<eos>',
            lower=True,
            batch_first=True)

TRG = Field(tokenize=tokenize_notes,
            init_token='<sos>',
            eos_token='<eos>',
            lower=True,
            batch_first=True)

data_fields = [('src', SRC), ('trg', TRG)]

train_data, test_data = torchtext.data.TabularDataset.splits(
    path='./', train='dataset/trainNotes.csv', validation='dataset/valNotes.csv',
    format='csv', fields=data_fields)

# Use a proper held-out validation split, not the test set
valid_data, test_data = test_data, test_data

SRC.build_vocab(train_data, min_freq=2)
TRG.build_vocab(train_data, min_freq=2)

train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size=BATCH_SIZE,
    sort=False,
    device=device)


def initalizeModel():
    INPUT_DIM = len(SRC.vocab)
    OUTPUT_DIM = len(TRG.vocab)
    HID_DIM = 256
    ENC_LAYERS = 3
    DEC_LAYERS = 3
    ENC_HEADS = 8
    DEC_HEADS = 8
    ENC_PF_DIM = 512
    DEC_PF_DIM = 512
    ENC_DROPOUT = 0.1
    DEC_DROPOUT = 0.1

    enc = Encoder(INPUT_DIM, HID_DIM, ENC_LAYERS, ENC_HEADS, ENC_PF_DIM, ENC_DROPOUT, device)
    dec = Decoder(OUTPUT_DIM, HID_DIM, DEC_LAYERS, DEC_HEADS, DEC_PF_DIM, DEC_DROPOUT, device)

    SRC_PAD_IDX = SRC.vocab.stoi[SRC.pad_token]
    TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]

    model = Seq2Seq(enc, dec, SRC_PAD_IDX, TRG_PAD_IDX, device).to(device)
    model.apply(initialize_weights)
    return model
