import os
import sys
import copy
from numpy.ma import masked_singleton
import torch
import random
import pygeodesy
import numpy as np
from torch._C import dtype

from torch.utils.data import Dataset
from sklearn.utils.random import sample_without_replacement
from scipy.spatial import distance
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from tqdm import tqdm
from multiprocessing import Pool, Manager
from collections import defaultdict
from collections import deque

class Tokenizer():
    def __init__(self, vocab: list, word2idx: list, idx2word: list, max_length: int,
                 pad_token  :str="[PAD]", 
                 cls_token  :str="[CLS]",
                 sep_token  :str="[SEP]", 
                 unk_token  :str="[UNK]",
                 eos_token  :str="[EOS]",
                 mask_token :str="[MASK]"):

        self.vocab = vocab
        self.word2idx = word2idx
        self.idx2word = idx2word
        self.max_length = max_length

        self.pad_token,  self.pad_id  = pad_token,  self.word2idx[pad_token]
        self.cls_token,  self.cls_id  = cls_token,  self.word2idx[cls_token]
        self.sep_token,  self.sep_id  = sep_token,  self.word2idx[sep_token]
        self.unk_token,  self.unk_id  = unk_token,  self.word2idx[unk_token]
        self.eos_token,  self.eos_id  = eos_token,  self.word2idx[eos_token]
        self.mask_token, self.mask_id = mask_token, self.word2idx[mask_token]

    def get_vocab_size(self):
        return len(self.vocab)

    def get_index(self, token):
        index = self.word2idx.get(token)
        return index if index is not None else self.unk_id

    def get_token(self, index):
        return self.idx2word.get(index)

    def _generator(self, li):
        for i, item in enumerate(li):
            yield i, item
            
    def encode(self, sentence, n_paddings=0):
        # CLS
        encoded = []
        """
        encoded = [self.cls_id]
        """
        # encode
        for _, token in self._generator(sentence):
            encoded.append(self.get_index(token))
        # SEP & PAD
        encoded += [self.pad_id] * n_paddings
        """
        encoded += [self.sep_id] + [self.pad_id] * n_paddings
        """
        return encoded

    def decode(self, encoded_token_indice):
        encoded_token_indice = encoded_token_indice.numpy() if type(encoded_token_indice) == torch.Tensor else encoded_token_indice
        tokens = []
        for _, id in self._generator(encoded_token_indice):
            if id == self.pad_id:
                break
            tokens.append(self.get_token(id))
        return tokens


class Augmentor():
    def __init__(self, omitting_rate, tweaking_rate, aug_lambda, SUBSEQUENCE_FEATURES):
        self.SUBSEQUENCE_FEATURES = SUBSEQUENCE_FEATURES
        self.aug_lambda    = aug_lambda
        self.omitting_rate = omitting_rate # 0.15
        self.tweaking_rate = tweaking_rate # 0.15
        """
        SUBSEQUENCE_FEATURES = ['eta_remain', 'relative_lon', 'relative_lat', 'sog', 'rot', 'cog', 'heading', 'ais_max_draught', 'nvg_status']
           
        instance = list(
            [0] main_sentence      : list(str=(L,))
            [1] spacetime_x        : list(L, np.array(float64=(L, 3)) [main_coord(lon, lat), time_progress]
            [2] main_length        : int L
            [3] subsequence_x      : list(L, np.array(float64=(sub_length, n_features=10)))
            [4] subsequence_length : list(int=(L,))
            [5] shiptype           : int 0, 1, 2
            [6] departure          : int
            [7] destination        : int
            [8] subsequence_y      : list(L, np.array(float64=(n_sublabels=4)))
            [9] is_label           : bool
        )
        """     
        
    def m2o_augment(self, instance):
        augment_list = ['cutoff']
        for aug_func in augment_list:
            instance = self.__getattribute__(aug_func)(instance)
        return instance
        
    def m2m_augment(self, instance):
        augment_list = random.sample(['inversed_cutoff', 'omit', 'tweak'], min(np.random.poisson(self.aug_lambda, 1).item(), 3)) # 0 ~ 3 Augmentation Applied
        for aug_func in augment_list:
            instance = self.__getattribute__(aug_func)(instance)
        return instance

    def cutoff(self, instance):
        cutoff = random.choice(range(len(instance[0])+1)) + 1 # index [1 ~ L]
        instance[0] = instance[0][:cutoff] # main_sentence
        instance[1] = instance[1][:cutoff] # spacetime_x
        instance[3] = instance[3][:cutoff] # subsequence_x
        instance[4] = instance[4][:cutoff] # subsequence_length
        instance[8] = instance[8][:cutoff] # subsequence_y
        instance[2] = len(instance[0])     # main_length
        return instance

    def inversed_cutoff(self, instance):
        instance[6] = 0 # departure to unknown
        if len(instance[0]) > 3:
            cutoff = random.choice(range(len(instance[0])-3)) # index [0 ~ (L-4)]
            instance[0] = instance[0][cutoff:] # main_sentence
            instance[1] = instance[1][cutoff:] # spacetime_x
            instance[3] = instance[3][cutoff:] # subsequence_x
            instance[4] = instance[4][cutoff:] # subsequence_length
            instance[8] = instance[8][cutoff:] # subsequence_y
            instance[2] = len(instance[0])     # main_length
        return instance

    def omit(self, instance):
        if len(instance[0]) > 3:
            omit_idx = (np.random.rand(len(instance[0])) < self.omitting_rate).nonzero()[0]
            if len(instance[0]) - len(omit_idx) < 3:
                omit_idx = random.sample(omit_idx.tolist(), len(instance[0])-3)
            for i in sorted(omit_idx, reverse=True):
                del instance[0][i] # main_sentence
                del instance[1][i] # spacetime_x
                del instance[3][i] # subsequence_x
                del instance[4][i] # subsequence_length
                del instance[8][i] # subsequence_y
            instance[2] = len(instance[0]) # main_length
        return instance
    
    def tweak(self, instance):
        tweak_idx = (np.random.rand(len(instance[0])) < self.tweaking_rate).nonzero()[0] # main_length
        for i in tweak_idx:
            aug_lon, aug_lat = (instance[3][i][instance[4][i]-1, [self.SUBSEQUENCE_FEATURES.index('relative_lon'), self.SUBSEQUENCE_FEATURES.index('relative_lat')]] # naive direction pattern from subsequence_x
                                - instance[3][i][0, [self.SUBSEQUENCE_FEATURES.index('relative_lon'), self.SUBSEQUENCE_FEATURES.index('relative_lat')]]) 
            aug_norm = np.random.uniform(-2, 2)
            cen_lat, cen_lon, precision = pygeodesy.wgrs.decode3(instance[0][i])
            cen_lat += aug_lat * aug_norm
            cen_lon += aug_lon * aug_norm
            cen_lat = cen_lat+180 if cen_lat < -90 else cen_lat
            cen_lat = cen_lat-180 if cen_lat > 90 else cen_lat
            cen_lon = cen_lon+360 if cen_lon < -180 else cen_lon
            cen_lon = cen_lon-360 if cen_lon > 180 else cen_lon
            instance[0][i]     = pygeodesy.wgrs.encode(lat=cen_lat, lon=cen_lon, precision=precision) # main_sentence
            instance[1][i][:2] = cen_lon, cen_lat                                                     # main_coordinates
        return instance


class Dataset(Dataset):
    def __init__(self, labeled_corpus, vocab, word2idx, idx2word, args, scaler=None, unlabeled_corpus=None, valid=False, feature_index=None):
        self.INSTANCE_INDEX, self.STACK_INDEX, self.SUBSEQUENCE_FEATURES, self.SCALER_TARGETS = feature_index
        """
        feature_index : 
            INSTANCE_INDEX       = ['main_sentence', 'subsequence_stack', 'subsequence_length', 'shiptype', 'departure', 'destination', 'main_coordinates', 'meta_info']
            STACK_INDEX          = ['timestamp', 'eda_label', 'eta_label', 'absolute_lon', 'absolute_lat', 'time_progress', 'eta_remain', 'relative_lon', 'relative_lat', 'sog', 'rot', 'cog', 'heading', 'ais_max_draught', 'nvg_status']
            SUBSEQUENCE_FEATURES = ['eta_remain', 'relative_lon', 'relative_lat', 'sog', 'rot', 'cog', 'heading', 'ais_max_draught', 'nvg_status']
            SCALER_TARGETS       = [STACK_INDEX.index(feature) for feature in SUBSEQUENCE_FEATURES]
                
            corpus : list of instances (
                tokenized sentence : list(str=(L,))
                subsequence stack  : np.array(float64=(sum(subsequence_length)+1, n_features=15))
                                    --> actual vessel trajectory sequence
                subsequence length : np.array(int64=(L,))
                shiptype : int 0,1,2
                departure : int
                destination : int
                main_coordinates : np.array(float64=(L, 2))
                meta_info : [[shiptype, imo_no, start_idx, ended_idx], read_dir]
            )

            vocab    : list of Grid Cell Tokens
            word2idx : dictionary maps Token(str) to Index(int)
            idx2word : dictionary maps Index(int) to Token(str)
        """
        self.labeled_corpus   = labeled_corpus
        self.unlabeled_corpus = unlabeled_corpus
        self.mix_portion    = args.mix_portion
        self.mix_corpus()

        self.max_length = args.max_length
        self.sub_length = args.sub_length

        self.tokenizer = Tokenizer(vocab, word2idx, idx2word, max_length=self.max_length)

        self.aug_ratio = 0 if valid else args.aug_ratio
        self.augmentor = Augmentor(args.omitting_rate, args.tweaking_rate, args.aug_lambda, self.SUBSEQUENCE_FEATURES)

        self.encode  = self.tokenizer.encode
        self.decode  = self.tokenizer.decode
        self.augment = self.augmentor.m2m_augment # self.augment = self.augmentor.m2o_augment

        self.scaler  = scaler
        
    def mix_corpus(self):
        if self.unlabeled_corpus is not None:
            if self.mix_portion < 0:
                self.corpus = self.labeled_corpus + self.unlabeled_corpus
            else:
                self.corpus = self.labeled_corpus + random.sample(self.unlabeled_corpus, int(len(self.labeled_corpus)*self.mix_portion))
        else:
            self.corpus = self.labeled_corpus

    def downsample_subsequence(self, subsequence_stack, subsequence_length):
        """
        STACK_INDEX          = ['timestamp', 'eda_label', 'eta_label', 'absolute_lon', 'absolute_lat', 'time_progress', 'eta_remain', 'relative_lon', 'relative_lat', 'sog', 'rot', 'cog', 'heading', 'ais_max_draught', 'nvg_status']
        SUBSEQUENCE_FEATURES = ['eta_remain', 'relative_lon', 'relative_lat', 'sog', 'rot', 'cog', 'heading', 'ais_max_draught', 'nvg_status']
        SCALER_TARGETS       = [STACK_INDEX.index(feature) for feature in SUBSEQUENCE_FEATURES]
        
        subsequence_stack  : np.array(float64=(sum(subsequence_length)+1, n_features=15)) - V-Stacked
        subsequence_length : np.array(int=(L,)) 
        """
        def _generator(subsequence_length):
            for i, j in zip(np.cumsum([0] + list(subsequence_length))[:-1], np.cumsum([0] + list(subsequence_length))[1:]):
                yield i, j
                
        # sample 1~sub_len dots from each grid
        sampled_indice, sampled_length = [], []
        for i, j in _generator(subsequence_length):
            n_samples = min(j-i, self.sub_length, max(1, np.random.poisson(5, 1).item()))
            sampled_indice.append([i] + sorted(random.sample(range(i+1, j), n_samples-1)) + [-1]*(self.sub_length-n_samples))
            sampled_length.append(n_samples)            
        sampled_indice, sampled_length = np.array(sampled_indice), np.array(sampled_length)
        last_sample_indice = sampled_indice[range(len(sampled_length)), sampled_length-1]
        
        subsequence_stack = np.concatenate([subsequence_stack, np.zeros((1, subsequence_stack.shape[-1]))], axis=0) # index -1 for subsequenctial 0-padding
        subsequence_x = subsequence_stack[sampled_indice][:, :, self.SCALER_TARGETS]                       # SUBSEQUENCE_FEATURES      
        subsequence_y = subsequence_stack[last_sample_indice][:, [self.STACK_INDEX.index('eda_label'),     # ['eda_label', 'eta_label', 'absolute_lon', 'absolute_lat']
                                                                  self.STACK_INDEX.index('eta_label'), 
                                                                  self.STACK_INDEX.index('absolute_lon'), 
                                                                  self.STACK_INDEX.index('absolute_lat')]]         
        subsequence_y[:, -2:] = subsequence_stack[last_sample_indice+1][:, [self.STACK_INDEX.index('absolute_lon'), 
                                                                            self.STACK_INDEX.index('absolute_lat')]] # next (absolute_lon, absolute_lat) coordinates

        timeprogress_x = subsequence_stack[last_sample_indice][:, [self.STACK_INDEX.index('time_progress')]]         # time_progress from departure
        timeprogress_x[timeprogress_x<0] = 0.0

        return timeprogress_x, subsequence_x, subsequence_y, sampled_length
        # timeprogress_x : np.array(L, 1)
        # subsequence_x  : np.array(L, sub_length, n_features) --> x_features = [relative_lon, relative_lat, eta_remain, sog, rot, cog, heading, ais_max_draught, nvg_status]
        # subsequence_y  : np.array(L, 4)                      --> y_features = [eda_label, eta_label, next ('absolute_lon', 'absolute_lat')] 
        # sampled_length : np.array(L,)                        --> (1 ~ sub_length)
        
    def pad_preprocess(self, instance):
        """
        instance = list(
            [0] main_sentence      : list(str=(L,))
            [1] spacetime_x        : list(L, np.array(float64=(L, 3)) [main_coord(lon, lat), time_progress]
            [2] main_length        : int L
            [3] subsequence_x      : list(L, np.array(float64=(sub_length, n_features=10)))
            [4] subsequence_length : list(int=(L,))
            [5] shiptype           : int 0, 1, 2
            [6] departure          : int
            [7] destination        : int
            [8] subsequence_y      : list(L, np.array(float64=(n_sublabels=4)))
            [9] is_label           : bool
        )
        """        
        n_paddings = self.max_length - instance[2] # main_length
        if n_paddings < 0: # L over max_length
            s_pos = random.choice(range(-n_paddings+1))
            instance[0] = instance[0][s_pos:s_pos+self.max_length] # main_sentence
            instance[1] = instance[1][s_pos:s_pos+self.max_length] # spacetime_x
            instance[3] = instance[3][s_pos:s_pos+self.max_length] # subsequence_x
            instance[4] = instance[4][s_pos:s_pos+self.max_length] # subsequence_length
            instance[8] = instance[8][s_pos:s_pos+self.max_length] # subsequence_y
            instance[1], n_paddings = len(instance[0]), 0          # main_length & n_paddings
            
        return instance, n_paddings

    def __len__(self):
        return len(self.corpus)

    def __getitem__(self, index):
        """
            INSTANCE_INDEX       = ['main_sentence', 'subsequence_stack', 'subsequence_length', 'shiptype', 'departure', 'destination', 'main_coordinates', 'meta_info']
            STACK_INDEX          = ['timestamp', 'eda_label', 'eta_label', 'absolute_lon', 'absolute_lat', 'time_progress', 'eta_remain', 'relative_lon', 'relative_lat', 'sog', 'rot', 'cog', 'heading', 'ais_max_draught', 'nvg_status']
            SUBSEQUENCE_FEATURES = ['eta_remain', 'relative_lon', 'relative_lat', 'sog', 'rot', 'cog', 'heading', 'ais_max_draught', 'nvg_status']
                    
        instance = list(
            [0] main_sentence      : list(str=(L,)) 
            [1] subsequence_stack  : np.array(float64=(sum(subseq_length), n_features=12))
            [2] subsequence_length : np.array(int=(L,)) 
            [3] shiptype           : int 0, 1, 2
            [4] departure          : int
            [5] destination        : int
            [6] main_coordinates   : list(float=(L, 2))
            [7] meta_info          : [[shiptype, imo_no, start_idx, ended_idx], read_dir]
        )
        """
        
        is_label = True if index < len(self.labeled_corpus) else False
        instance = copy.deepcopy(self.corpus[index])

        # downsample subsequence & re-define instance as model input 
        main_sentence      = instance[self.INSTANCE_INDEX.index('main_sentence')]
        main_coordinates   = instance[self.INSTANCE_INDEX.index('main_coordinates')]
        subsequence_stack  = instance[self.INSTANCE_INDEX.index('subsequence_stack')]
        subsequence_length = instance[self.INSTANCE_INDEX.index('subsequence_length')]
        shiptype           = instance[self.INSTANCE_INDEX.index('shiptype')]
        departure          = instance[self.INSTANCE_INDEX.index('departure')]
        destination        = instance[self.INSTANCE_INDEX.index('destination')]
        
        timeprogress_x, subsequence_x, subsequence_y, subsequence_length = self.downsample_subsequence(subsequence_stack, subsequence_length)
        spacetime_x = np.concatenate([main_coordinates, timeprogress_x], axis=-1)
        instance = [main_sentence,
                    list(spacetime_x),                
                    len(main_sentence), 
                    list(subsequence_x), 
                    list(subsequence_length), 
                    shiptype, 
                    departure, 
                    destination, 
                    list(subsequence_y), 
                    is_label]
        """
        instance = list(
            [0] main_sentence      : list(str=(L,))
            [1] spacetime_x        : list(L, np.array(float64=(L, 3)) [main_coord(lon, lat), time_progress]
            [2] main_length        : int L
            [3] subsequence_x      : list(L, np.array(float64=(sub_length, n_features=10)))
            [4] subsequence_length : list(int=(L,))
            [5] shiptype           : int 0, 1, 2
            [6] departure          : int
            [7] destination        : int
            [8] subsequence_y      : list(L, np.array(float64=(n_sublabels=4)))
            [9] is_label           : bool
        )
        """
        # apply augmentation
        if random.random() < self.aug_ratio and is_label:
            instance = self.augment(instance)
        
        # scale subsequence features
        if self.scaler is not None:
            instance[3], instance[4] = np.array(instance[3]), np.array(instance[4]) # subsequence_x, subsequence_length
            valid_idx = np.arange(self.sub_length)[None, :] < instance[4][:, None]
            instance[3][valid_idx] = self.scaler.transform(instance[3][valid_idx])
            instance[3], instance[4] = list(instance[3]), list(instance[4])
            
        # padding preprocess
        instance, n_paddings = self.pad_preprocess(instance)
        
        # encode main_sentence with pad
        instance[0] = self.encode(instance[0], n_paddings)
        
        # pad subsequence
        instance[1] += [np.zeros_like(instance[1][-1])] * n_paddings # pad main_coordinates & time_progress
        instance[3] += [np.zeros_like(instance[3][-1])] * n_paddings # pad subsequence_x
        instance[4] += [0] * n_paddings                              # pad subsequence_length
        instance[8] += [np.zeros_like(instance[8][-1])] * n_paddings # pad subsequence_y
        
        # Tensor transformation
        instance[0] = torch.LongTensor(instance[0])  # main_sentence
        instance[1] = torch.FloatTensor(instance[1]) # spacetime_x
        instance[3] = torch.FloatTensor(instance[3]) # subsequence_x
        instance[4] = torch.LongTensor(instance[4])  # subsequence_length
        instance[8] = torch.FloatTensor(instance[8]) # subsequence_y
        
        return instance
        # [0] main_sentence      : int(batch_size, max_length)
        # [1] spacetime_x        : int(batch_size, max_length, 3) [main_coord(lon, lat), time_progress]
        # [2] main_length        : int(batch_size,)
        # [3] subsequence_x      : float(batch_size, max_length, sub_length, n_features=10)
        # [4] subsequence_length : int(batch_size, max_length)
        # [5] shiptype           : int(batch_size,)
        # [6] departure          : int(batch_size,)
        # [7] destination        : int(batch_size,)
        # [8] subsequence_y      : float(batch_size, max_length, n_sublabels=4) ['eda_label', 'eta_label', next(absolute_lon, absolute_lat)]
        # [9] is_label           : bool(batch_size,)
        
        
        
         
   
        
        
    