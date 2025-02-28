import os
import sys
import argparse
import importlib
import torch
import random

from tqdm import tqdm
from multiprocessing import Pool, Value
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from utils.ConfigManager import *
from utils.seedSet import *
from utils.fastIO import *
from utils.Parser import *
from utils.Scaler import *

from transformers import *
from dataset import *
from trainer import *
from model import *


INSTANCE_INDEX       = ['main_sentence', 'subsequence_stack', 'subsequence_length', 'shiptype', 'departure', 'destination', 'main_coordinates', 'meta_info']
STACK_INDEX          = ['timestamp', 'eda_label', 'eta_label', 'absolute_lon', 'absolute_lat', 'time_progress', 'eta_remain', 'relative_lon', 'relative_lat', 'sog', 'rot', 'cog', 'heading', 'ais_max_draught', 'nvg_status']
SUBSEQUENCE_FEATURES = ['eta_remain', 'relative_lon', 'relative_lat', 'sog', 'rot', 'cog', 'heading', 'ais_max_draught', 'nvg_status']
SCALER_TARGETS       = [STACK_INDEX.index(feature) for feature in SUBSEQUENCE_FEATURES]

def main():
    def _generator(corpus):
        for instance in tqdm(corpus):
            yield instance[INSTANCE_INDEX.index('subsequence_stack')]
    
    args = load_config()
    seed_set(args.seed)
    
    vocab, word2idx, idx2word = fastRead_pkl('../../data/vocab_fromcorpus.pkl')
    train_corpus     = fastRead_pkl('../../data/train_corpus.pkl')
    valid_corpus     = fastRead_pkl('../../data/valid_corpus.pkl')
    test_corpus      = fastRead_pkl('../../data/test_corpus.pkl')

    if args.abs_coord:
        print('   ::overwrite relative-coordinates to real-coordinates...')
        for corpus in [train_corpus, valid_corpus, test_corpus]:
            for subsequence_stack in _generator(corpus):
                subsequence_stack[:, [STACK_INDEX.index('relative_lon'), STACK_INDEX.index('relative_lat')]] = subsequence_stack[:, [STACK_INDEX.index('absolute_lon'), STACK_INDEX.index('absolute_lat')]]
    
    scaler = fit_scaler(train_corpus, feature_index=(INSTANCE_INDEX, SCALER_TARGETS), scaler_type='standard')

    train_dataset = Dataset(train_corpus, vocab, word2idx, idx2word, args, scaler=scaler, feature_index=(INSTANCE_INDEX, STACK_INDEX, SUBSEQUENCE_FEATURES, SCALER_TARGETS))
    valid_dataset = Dataset(valid_corpus, vocab, word2idx, idx2word, args, scaler=scaler, valid=True, feature_index=(INSTANCE_INDEX, STACK_INDEX, SUBSEQUENCE_FEATURES, SCALER_TARGETS)) 
    test_dataset  = Dataset(test_corpus,  vocab, word2idx, idx2word, args, scaler=scaler, valid=True, feature_index=(INSTANCE_INDEX, STACK_INDEX, SUBSEQUENCE_FEATURES, SCALER_TARGETS))

    train_loader  = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    valid_loader  = DataLoader(valid_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
    test_loader   = DataLoader(test_dataset,  batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
    mixed_loader = None

    trainer = Trainer(train_loader, mixed_loader, valid_loader, test_loader, args)
    trainer.train()

if __name__ == "__main__":
    main()



