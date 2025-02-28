import logging
import os
import sys
import torch
import torch.nn as nn
from torch.nn.modules.loss import BCELoss, CrossEntropyLoss, NLLLoss
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils import data
from torch.utils.data import DataLoader

from model import *
from dataset import *

import neptune
from easydict import EasyDict


def neptune_init(args):
    print('\nLogging on ... ', end='')
    with open('./neptune/project.txt', 'r') as f:
        project_name = f.readline()
    with open('./neptune/api.txt', 'r') as f:
        api_token    = f.readline()
    neptune.init(project_name, api_token=api_token)
    exp_name   = str(os.path.basename(os.path.normpath(os.getcwd())))
    experiment = neptune.create_experiment(name=exp_name, params=vars(args))
    if args.tag is not None:
        for tagged in args.tag:
            neptune.append_tag(str(tagged))
    
    return experiment._id

def neptune_resume(exp_id):
    import neptune.new as neptune
    with open('./neptune/project.txt', 'r') as f:
        project_name = f.readline()
    with open('./neptune/api.txt', 'r') as f:
        api_token    = f.readline()
    print(f'\nResuming Experiment {exp_id}, Logging on ... ', end='')
    resume_run = neptune.init(project_name, api_token=api_token, run=exp_id)
    return resume_run


class Trainer():
    def __init__(self, train_loader, mixed_loader, valid_loader, test_loader, args):
        self.train_loader = train_loader
        self.mixed_loader = mixed_loader
        self.valid_loader = valid_loader
        self.test_loader  = test_loader
        self.device = torch.device(f"{args.device}" if torch.cuda.is_available() else "cpu")
        
        # check if evaluation, resuming experiment or fresh start
        if args.evaluate_experiment is not None:
            self.evaluate_experiment(args.evaluate_experiment)
        else:
            if args.resume_experiment is None:
                if args.semi_quickstart_from is None:
                    self.fresh_start(args)
                else:
                    self.semi_quickstart(args)
            else:
                self.resume_experiment(args)
        print('   ::Total Params:', sum([p.nelement() for p in self.model.parameters()]))

        # define criterions
        self.criterionLTD = nn.CrossEntropyLoss(ignore_index=0) # unlabeled_port = 0
        self.criterionSTD = nn.CrossEntropyLoss(ignore_index=0) # padding_idx    = 0
        self.criterionEDA = nn.L1Loss()
        self.criterionETA = nn.L1Loss()
        self.criterionCRD = nn.MSELoss()
        
    ################################################## Initiational Functions ##################################################
    
    def fresh_start(self, args):
        self.args = args
        if self.args.logger:
            self.exp_id   = neptune_init(self.args)
            os.makedirs(f'./weights/{self.exp_id}/', exist_ok=True)
        else:
            self.exp_id   = ''
        self.epoch_start  = 0
        self.resume_run   = None
        self.val_min_loss = 1e+10
        self.val_max_acc  = -1
        
        # initialize model from arguments
        self.model     = DownstreamHead(WAY(self.args)).to(self.device); self.model.set_device(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr, betas=(self.args.beta1, self.args.beta2), weight_decay=self.args.weight_decay)
        print('== Initiation Complete, Fresh-Start from Ground-Zero ==')
            
    def semi_quickstart(self, args):
        checkpoint = torch.load(f'./weights/{args.semi_quickstart_from}/checkpoint_ep{args.init_round}_supervised.pth')
        assert checkpoint['exp_id'] == args.semi_quickstart_from
        
        # overwite semi-relevant arguments on loaded argument
        self.args = checkpoint['args']

        # trainer actions
        self.args.semi_quickstart_from = args.semi_quickstart_from 
        self.args.resume_experiment    = None  
        self.args.n_epochs             = args.n_epochs
        self.args.batch_size           = args.batch_size
        self.args.num_workers          = args.num_workers
        self.args.device               = args.device

        # semi-supervised learning
        self.args.init_round           = args.init_round  
        self.args.mix_portion          = args.mix_portion
        self.args.mix_extra            = args.mix_extra
        self.args.conf_thresh          = args.conf_thresh

        # optimizer
        self.args.STD_ratio            = args.STD_ratio
        self.args.EDA_ratio            = args.EDA_ratio
        self.args.ETA_ratio            = args.ETA_ratio
        self.args.CRD_ratio            = args.CRD_ratio

        # minor-replacement
        self.args.config_update        = args.config_update  
        self.args.tag                  = args.tag
                      
        if args.logger:
            self.args.logger = True
            self.exp_id      = neptune_init(self.args)
            os.makedirs(f'./weights/{self.exp_id}/', exist_ok=True)
            
            # repeat quick-start loggings for init round
            for _ in range(self.args.init_round):
                for key in checkpoint['logging_dict'].keys():
                    neptune.log_metric(key, checkpoint['logging_dict'][key])
                    
        else:
            self.args.logger = False
            self.exp_id   = ''
        self.epoch_start  = checkpoint['epoch_no']
        self.resume_run   = None
        self.val_min_loss = checkpoint['valid_loss']
        self.val_max_acc  = checkpoint['valid_acc']
        
        # initialize model from checkpoint
        self.model     = checkpoint['model'].to(self.device); self.model.set_device(self.device)
        self.optimizer = optim.Adam(self.model.parameters())
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        
        # set up rng environs
        self.rng_state = checkpoint['rng_state']
        random.setstate(self.rng_state['rand_state'])
        np.random.set_state(self.rng_state['np_rand_state'])
        torch.random.set_rng_state(self.rng_state['torch_rng_state'])        
        torch.cuda.set_rng_state(self.rng_state['cuda_rng_state'], device=self.device)
        os.environ['PYTHONHASHSEED'] = str(self.rng_state['os_hash_seed'])
        
        print(f'== Initiation Complete, Semi Quick Start from {self.args.semi_quickstart_from}, {self.args.init_round} init round ==')
        
    def resume_experiment(self, args):
        checkpoint = torch.load(f'./weights/{args.resume_experiment}/last_checkpoint.pth')
        assert checkpoint['exp_id'] == args.resume_experiment
        
        self.args = checkpoint['args']
        
        # while resuming experiment on neptune, logger always True
        self.exp_id       = checkpoint['exp_id']
        self.epoch_start  = checkpoint['epoch_no']
        self.resume_run   = neptune_resume(self.exp_id)
        self.val_min_loss = checkpoint['valid_loss']
        self.val_max_acc  = checkpoint['valid_acc']        
        
        # overwrite total epoch if experiment extenstion (new total epoch > previous total epoch)
        if self.args.n_epochs < args.n_epochs:
            self.args.n_epochs = args.n_epochs
            self.resume_run["parameters/n_epochs"] = args.n_epochs                
        
        # init model from checkpoint
        self.model     = checkpoint['model'].to(self.device); self.model.set_device(self.device)
        self.optimizer = optim.Adam(self.model.parameters())
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        
        # set up rng environs
        self.rng_state = checkpoint['rng_state']
        random.setstate(self.rng_state['rand_state'])
        np.random.set_state(self.rng_state['np_rand_state'])
        torch.random.set_rng_state(self.rng_state['torch_rng_state'])        
        torch.cuda.set_rng_state(self.rng_state['cuda_rng_state'], device=self.device)
        os.environ['PYTHONHASHSEED'] = str(self.rng_state['os_hash_seed'])

        print(f'== Initiation Complete, Resuming from {self.epoch_start} epoch ==')
        
    def evaluate_experiment(self, exp_id):
        self.eval(exp_id)
        
    ################################################## Initiational Functions ##################################################

    def _epoch(self, dataloader, valid=False, semi=False):
        running_loss, total_loss = 0.0, 0.0
        LTD_correct, LTD_count   = torch.Tensor([0, 0, 0, 0, 0]), torch.Tensor([0, 0, 0, 0, 0])
        STD_correct, STD_count   = 0, 0
        CRD_loss = 0.0
        EDA_loss, EDA_loss_all   = 0.0, 0.0
        ETA_loss, ETA_loss_all   = 0.0, 0.0
        
        for i, instance in enumerate(tqdm(dataloader)):
            """
            instance = list(
                [0] main_sentence      : list(str=(L,))
                [1] spacetime_x        : list(L, np.array(float64=(L, 3)) [main_coord(lon, lat), time_progress]
                [2] main_length        : int L
                [3] subsequence_x      : list(L, np.array(float64=(sub_length, n_features=9)))
                [4] subsequence_length : list(int=(L,))
                [5] shiptype           : int 0, 1, 2
                [6] departure          : int
                [7] destination        : int
                [8] subsequence_y      : list(L, np.array(float64=(n_sublabels=4)))
                [9] is_label           : bool
            )
            """
            # model input
            main_sentence      = instance[0].long().to(self.device)
            spacetime_x        = instance[1].to(self.device)
            main_length        = instance[2].long().to(self.device)
            subsequence_x      = instance[3].to(self.device)
            subsequence_length = instance[4].long()
            shiptype           = instance[5].long().to(self.device)
            departure          = instance[6].long().to(self.device)
            """
            main_sentence      : long (batch_size, max_length)
            spacetime_x        : float(batch_size, max_length, 3)
            main_length        : long (batch_size)
            subsequence_x      : float(batch_size max_length, sub_length, n_features=9)
            subsequence_length : long (batch_size, max_length)
            shiptype           : long (batch_size)
            departure          : long (batch_size)
            """
            # destination & ETA labels
            LTD_label = instance[7].long().to(self.device)
            STD_label = main_sentence[:, 1:].contiguous()
            CRD_label = instance[8][:, :, -2:].to(self.device)
            EDA_label = instance[8][:, :, [0]].to(self.device)
            ETA_label = instance[8][:, :, [1]].to(self.device)
            """
            LTD_label : long (batch_size)
            STD_label : long (batch_size, max_length-1)
            CRD_label : float(batch_size, max_length, 2)
            EDA_label : float(batch_size, max_length, 1)
            ETA_label : float(batch_size, max_length, 1)
            """
            # labeled & unlabeled index
            labeled_index   = instance[9].bool()
            unlabeled_index = instance[7].eq(0)
            """
            labeled_index   : bool(batch_size)
            unlabeled_index : bool(batch_size)
            """
            
            # Semi-Supervised Pseudo-Labelling [Dynamic Wrapper]
            if semi:
                # model forward without gradient
                self.model.eval()
                LTD_logit, STD_logit, CRD_logit, EDA_logit, ETA_logit, chattn_list, sfattn_list = self.model(spacetime_x, shiptype, departure, subsequence_x, subsequence_length)
                pseudo_conf, pseudo_label = F.softmax(LTD_logit[range(len(LTD_logit)), main_length-1], dim=-1).max(dim=1)
                pseudo_label = pseudo_label.masked_fill(pseudo_conf < self.args.conf_thresh, 0)
                LTD_label[unlabeled_index] = pseudo_label[unlabeled_index]
                self.model.train()
                """
                pseudo_conf  : float(batch_size)
                pseudo_label : long (batch_size)
                LTD_label    : long (batch_size)
                """
                
            # model forward with gradient
            LTD_logit, STD_logit, CRD_logit, EDA_logit, ETA_logit, chattn_list, sfattn_list = self.model(spacetime_x, shiptype, departure, subsequence_x, subsequence_length)
            """
            LTD_logit   : (batch_size, max_length, n_class)
            STD_logit   : (batch_size, max_length, vocab_size)
            CRD_logit   : (batch_size, max_length, 2)
            EDA_logit   : (batch_size, max_length, 1)
            ETA_logit   : (batch_size, max_length, 1)
            sfattn_list : (n_blocks, batch_size, n_heads, max_length, max_length)
            """
            
            # [LTD PREDICTION] : M2M label generation
            LTD_label = (LTD_label.unsqueeze(1).repeat(1, main_sentence.size(1))
                         * (torch.arange(main_sentence.size(1), device=self.device)[None, :] < main_length[:, None]))
            if not valid and self.args.grad_drop:
                # Drop Gradient by Sampling Prediction Steps within Originals' log-scaled batch-wise Length Distribution
                sample_rate = main_length.float().log()
                sample_rate = (sample_rate.max() - sample_rate + sample_rate.min()) / (sample_rate.max() - sample_rate.min() + sample_rate.min())
                LTD_targets = LTD_label * (torch.rand(LTD_label.size(), device=self.device) < sample_rate.unsqueeze(1))
            else:
                LTD_targets = LTD_label
            lossLTD = self.criterionLTD(LTD_logit.transpose(1, 2).contiguous(), LTD_targets)
            """
            LTD_label    : (batch_size, max_length)
            LTD_logit    : (batch_size, max_length, n_class)
            masked_label : (batch_size, max_length)
            """
            
            # [STD PREDICTION] : shifted right
            STD_logit = STD_logit[:, :-1].contiguous()
            lossSTD = self.criterionSTD(STD_logit.transpose(1, 2).contiguous(), STD_label)
            """
            STD_label : (batch_size, max_length-1)
            STD_logit : (batch_size, max_length-1, vocab_size)
            """

            # [CRD PREDICTION] : next coordinates
            CRD_label = CRD_label[LTD_label!=0]
            CRD_logit = CRD_logit[LTD_label!=0]
            lossCRD = self.criterionCRD(CRD_logit, CRD_label)
            """
            CRD_label : (n_non_padded, 2)
            CRD_logit : (n_non_padded, 2)
            """   

            # [EDA / ETA PREDICTION] : current eda within correct LTD prediction
            ETA_label   = ETA_label.masked_fill(ETA_label < 0, 0.0)
            EDA_EDA_targets = ((LTD_logit.argmax(dim=-1)==LTD_label) & (LTD_label!=0))[labeled_index]
            lossEDA     = torch.Tensor([0.0]).to(self.device)
            lossETA     = torch.Tensor([0.0]).to(self.device)
            if EDA_EDA_targets.sum():
                lossEDA = self.criterionEDA(EDA_logit[EDA_EDA_targets], EDA_label[EDA_EDA_targets]) 
                lossETA = self.criterionETA(ETA_logit[EDA_EDA_targets], ETA_label[EDA_EDA_targets])
            """
            EDA_ETA_targets : (n_labeled, max_length)
            ETA/EDA_label   : (n_correct, 1)
            ETA/EDA_logit   : (n_correct, 1)
            """         
            
            # [Loss Backward]
            loss = (lossLTD 
                    + (lossSTD * self.args.STD_ratio) 
                    + (lossCRD * self.args.CRD_ratio)                    
                    + (lossEDA * self.args.EDA_ratio) 
                    + (lossETA * self.args.ETA_ratio))
            if not valid:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            # [Evaluation]
            running_loss += loss.item()
            total_loss   += lossLTD.item() + lossSTD.item() + lossCRD.item() + lossEDA.item() + lossETA.item()
            
            LTD_pred = LTD_logit.argmax(dim=-1)
            for quater_no in [1, 2, 3, 4]:
                quater_mask = ((torch.arange(main_sentence.size(1), device=self.device)[None, :] > ((main_length[:, None]-1)/4*(quater_no-1)).long())
                                & (torch.arange(main_sentence.size(1), device=self.device)[None, :] <= ((main_length[:, None]-1)/4*quater_no).long()))
                quater_mask[:, 0] = True if quater_no == 1 else False
                """
                quater_mask : (batch_size, max_len)
                """
                LTD_correct[quater_no] += ((LTD_pred==(LTD_label * quater_mask)) & (quater_mask!=0))[labeled_index].sum().item()
                LTD_count[quater_no]   += (quater_mask!=0)[labeled_index].sum().item()
            LTD_correct[0] += ((LTD_pred==LTD_label) & (LTD_label!=0))[labeled_index].sum().item()
            LTD_count[0]   += (LTD_label!=0)[labeled_index].sum().item()
            
            STD_pred = STD_logit.argmax(dim=-1)
            STD_correct += ((STD_pred==STD_label) & (STD_label!=0)).sum().item()
            STD_count   += (STD_label!=0).sum().item()
            
            CRD_loss += torch.sqrt(lossCRD).item()
            
            EDA_loss += lossEDA.item()
            EDA_loss_all += self.criterionEDA(EDA_logit[labeled_index][LTD_label[labeled_index]!=0], EDA_label[labeled_index][LTD_label[labeled_index]!=0])

            ETA_loss += lossETA.item()
            ETA_loss_all += self.criterionETA(ETA_logit[labeled_index][LTD_label[labeled_index]!=0], ETA_label[labeled_index][LTD_label[labeled_index]!=0])
            
        return (running_loss/(i+1), 
                total_loss/(i+1), 
                LTD_correct/LTD_count*100, 
                STD_correct/STD_count*100, 
                CRD_loss/(i+1),
                EDA_loss/(i+1),
                EDA_loss_all/(i+1),                
                ETA_loss/(i+1), 
                ETA_loss_all/(i+1))
    
    def train(self):
        for epoch_no in range(self.epoch_start, self.args.n_epochs):
            
            ##################################################################################
            # recover train phase rng state if 'rng_state' variable exists 
            if hasattr(self, 'rng_state'):
                # print('recover from evalseed')
                random.setstate(self.rng_state['rand_state'])
                np.random.set_state(self.rng_state['np_rand_state'])
                torch.random.set_rng_state(self.rng_state['torch_rng_state'])
                torch.cuda.set_rng_state(self.rng_state['cuda_rng_state'], device=self.device)
                os.environ['PYTHONHASHSEED'] = str(self.rng_state['os_hash_seed'])
            ##################################################################################       
             
            logging_dict = {item: 0.0 for item in ['train loss', 'train total loss', 
                                                   'train LTD',  'train LTD 1Q', 'train LTD 2Q', 'train LTD 3Q', 'train LTD 4Q', 
                                                   'train STD',  'train CRD', 'train EDA', 'train total EDA', 'train ETA', 'train total ETA',
                                                   
                                                   'valid loss', 'valid total loss', 
                                                   'valid LTD',  'valid LTD 1Q', 'valid LTD 2Q', 'valid LTD 3Q', 'valid LTD 4Q', 
                                                   'valid STD',  'valid CRD', 'valid EDA', 'valid total EDA', 'valid ETA', 'valid total ETA',
                                                   
                                                   'test loss', 'test total loss', 
                                                   'test LTD',  'test LTD 1Q', 'test LTD 2Q', 'test LTD 3Q', 'test LTD 4Q', 
                                                   'test STD',  'test CRD', 'test EDA', 'test total EDA', 'test ETA', 'test total ETA']}
            
            self.model.train()
            if (epoch_no < self.args.init_round) or (self.args.init_round < 0):
                # supervised learning
                print('supervised epoch:%03d on progress...' % (epoch_no + 1))
                running_loss, total_loss, LTD, STD, CRD, EDA, total_EDA, ETA, total_ETA = self._epoch(self.train_loader)
            else:
                # semi_supervised learning
                print('semi-supervised epoch:%03d on progress...' % (epoch_no + 1))
                running_loss, total_loss, LTD, STD, CRD, EDA, total_EDA,  ETA, total_ETA = self._epoch(self.mixed_loader, semi=True)

            # log train results                
            logging_dict['train loss'], logging_dict['train total loss'] = running_loss, total_loss
            logging_dict['train LTD'], logging_dict['train STD'], logging_dict['train CRD'] = LTD[0], STD, CRD
            for quater_no in [1, 2, 3, 4]:
                logging_dict[f'train LTD {quater_no}Q'] = LTD[quater_no]
            logging_dict['train EDA'], logging_dict['train total EDA']   = EDA, total_EDA            
            logging_dict['train ETA'], logging_dict['train total ETA']   = ETA, total_ETA
                
            ################################################################################
            # save train phase rng state & fix (valid / test) phase rng state 
            self.rng_state = {'rand_state'      : random.getstate(),
                              'np_rand_state'   : np.random.get_state(),
                              'torch_rng_state' : torch.random.get_rng_state(),
                              'cuda_rng_state'  : torch.cuda.get_rng_state(),
                              'os_hash_seed'    : str(os.environ['PYTHONHASHSEED'])}            
            
            random.seed(self.args.evalseed)  
            np.random.seed(self.args.evalseed)
            torch.manual_seed(self.args.evalseed)
            torch.cuda.manual_seed(self.args.evalseed)
            torch.cuda.manual_seed_all(self.args.evalseed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            os.environ['PYTHONHASHSEED'] = str(self.args.evalseed)
            ################################################################################           
            
            self.model.eval()
            with torch.no_grad():
                for dataloader, set_type in zip([self.valid_loader, self.test_loader], ['valid', 'test']):
                    running_loss, total_loss, LTD, STD, CRD, EDA, total_EDA, ETA, total_ETA = self._epoch(dataloader, valid=True)
                    
                    # log evaluate results
                    logging_dict[f'{set_type} loss'], logging_dict[f'{set_type} total loss'] = running_loss, total_loss
                    logging_dict[f'{set_type} LTD'], logging_dict[f'{set_type} STD'], logging_dict[f'{set_type} CRD'] = LTD[0], STD, CRD
                    for quater_no in [1, 2, 3, 4]:
                        logging_dict[f'{set_type} LTD {quater_no}Q'] = LTD[quater_no]
                    logging_dict[f'{set_type} EDA'], logging_dict[f'{set_type} total EDA']   = EDA, total_EDA            
                    logging_dict[f'{set_type} ETA'], logging_dict[f'{set_type} total ETA']   = ETA, total_ETA
                    
            # save model
            self.archive_model(epoch_no, logging_dict)
            
            # print-out train result
            for set_type in ['train', 'valid', 'test']:
                print(f'   :: {set_type} result ::')
                for key, item in logging_dict.items():
                    if set_type in key:
                        if 'LTD' in key or 'STD' in key:
                            print('\t %s: %.04f%%' % (key, item), end=' ')
                        else:    
                            print('\t %s: %.04f' % (key, item), end=' ')
                        if 'total' in key or '4Q' in key or 'STD' in key or 'CRD' in key:
                            print()             

        print(f'\nStart Final Evaluation...')                    
        self.eval(self.exp_id)

        if self.args.logger:
            neptune.stop()
                    
    def eval(self, exp_id, criteria='acc'):
        checkpoint = torch.load(f'./weights/{exp_id}/best_valid_{criteria}.pth')
        assert checkpoint['exp_id'] == exp_id
        
        self.args = checkpoint['args']
        
        # set final loggings
        self.exp_id       = checkpoint['exp_id']
        self.epoch_start  = None
        self.resume_run   = None
        self.val_min_loss = checkpoint['valid_loss']
        self.val_max_acc  = checkpoint['valid_acc']
        
        # initiate model from checkpoint
        self.model     = checkpoint['model'].to(self.device); self.model.set_device(self.device)
        self.optimizer = optim.Adam(self.model.parameters())
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        
        # set up rng environs
        random.seed(self.args.evalseed)  
        np.random.seed(self.args.evalseed)
        torch.manual_seed(self.args.evalseed)
        torch.cuda.manual_seed(self.args.evalseed)
        torch.cuda.manual_seed_all(self.args.evalseed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ['PYTHONHASHSEED'] = str(self.args.evalseed)
        
        print(f'== Initiation Complete, Evaluate Experiment {self.exp_id}, Criteria is {criteria} ==')
        print(f'checking records...\n\t valid total loss: {self.val_min_loss}\n\t valid LTD: {self.val_max_acc}')        
        
        self.model.eval()
        with torch.no_grad():
            for dataloader, set_type in zip([self.valid_loader, self.test_loader], ['valid', 'test']):
                print(f'\n{set_type} set evaluation on progress...')
                eval_summary = {item: torch.Tensor() for item in [f'{set_type} loss', f'{set_type} total loss', 
                                                                  f'{set_type} LTD',  f'{set_type} STD',  f'{set_type} CRD', 
                                                                  f'{set_type} EDA',  f'{set_type} total EDA', 
                                                                  f'{set_type} ETA',  f'{set_type} total ETA']}
                for _ in range(10):
                    eval_result = self._epoch(dataloader, valid=True)
                    for i, result in enumerate(eval_result):
                        key = list(eval_summary.keys())[i]
                        result = torch.Tensor([result]).cpu() if type(result) != type(torch.Tensor()) else result.cpu()
                        eval_summary[key] = torch.cat([eval_summary[key], result.view(1, -1)], axis=0)
                for key, item in eval_summary.items():
                    if 'LTD' in key or 'STD' in key:
                        if 'LTD' in key:
                            print('\t %s: %.04f%% +-%.04f' % (key, item.mean(dim=0).tolist()[0], item.std(dim=0).tolist()[0]), end=' ')
                            for m, std, q in zip(item.mean(dim=0).tolist()[1:], item.std(dim=0).tolist()[1:], [1,2,3,4]):
                                print('\t %sQ: %.04f%% +-%.04f' % (q, m, std), end=' ')
                        else:
                            print('\t %s: %.04f%% +-%.04f' % (key, item.mean(dim=0).tolist()[0], item.std(dim=0).tolist()[0]), end=' ')
                    else:
                        print('\t %s: %.04f +-%.04f' % (key, item.mean(dim=0).tolist()[0], item.std(dim=0).tolist()[0]), end=' ')
                    if 'total' in key or 'LTD' in key or 'STD' in key or 'CRD' in key:
                        print()
            
    def archive_model(self, epoch_no, logging_dict):
        epoch_no = epoch_no + 1
        checkpoint = {'args'            : self.args,
                      'exp_id'          : self.exp_id,
                      'epoch_no'        : epoch_no,
                      'model'           : self.model.cpu(),
                      'state_dict'      : self.model.state_dict(),
                      'optimizer'       : self.optimizer.state_dict(),
                      'rng_state'       : self.rng_state,
                      'logging_dict'    : logging_dict,
                      'valid_loss'      : logging_dict['valid loss'],
                      'valid_acc'       : logging_dict['valid LTD']}
        
        # save the best loss
        if self.val_min_loss > checkpoint['valid_loss']:
            self.val_min_loss = checkpoint['valid_loss']
            torch.save(checkpoint, f'./weights/{self.exp_id}/best_valid_loss.pth')
            
        # save the best LTD accuracy
        if self.val_max_acc < checkpoint['valid_acc']:
            self.val_max_acc = checkpoint['valid_acc']
            torch.save(checkpoint, f'./weights/{self.exp_id}/best_valid_acc.pth')
            
        # log train result and save the checkpoint
        if self.args.logger:
            if epoch_no % 10 == 0:
                torch.save(checkpoint, f'./weights/{self.exp_id}/checkpoint_ep{epoch_no}.pth')
            
            for key in logging_dict.keys():
                if self.resume_run is None:
                    # fresh start / semi quick start
                    neptune.log_metric(key, logging_dict[key])
                else:
                    # resume experiment
                    self.resume_run["logs/"+key].log(logging_dict[key])
                    
        # saving the last checkpoint for resuming experiment
        torch.save(checkpoint, f'./weights/{self.exp_id}/last_checkpoint.pth')
        
        self.model.to(self.device)
        
                