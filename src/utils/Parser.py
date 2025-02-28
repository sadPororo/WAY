import argparse
import easydict
from ast import literal_eval

def load_config():
    parser = argparse.ArgumentParser()

    # trainer actions
    parser.add_argument('--semi_quickstart_from', type=str,  default=None,     help='quick start from "init round" checkpoint, choose exp_id from weights folder :: default is None')
    parser.add_argument('--resume_experiment',    type=str,  default=None,     help='resume experiment from checkpoint, choose exp_id from neptune.ai" :: default is None')
    parser.add_argument('--evaluate_experiment',  type=str,  default=None,     help='evaluate experiment from best accuracy, choose exp_id from neptune.ai" :: default is None')    
    parser.add_argument('--logger',               type=bool, default=False,    help='choose from {0, 1} :: default is 0(False)')
    parser.add_argument('--n_epochs',             type=int,  default=100,      help='number of epochs to train :: default is 100')
    parser.add_argument('--batch_size',           type=int,  default=32,       help='dataloader batch size :: default is 32')
    parser.add_argument('--num_workers',          type=int,  default=4,        help='dataloader num cpu-core workers :: default is 4')
    parser.add_argument('--device',               type=str,  default='cuda:0', help='local titan-"cuda:0", 3090-"cuda:0", 224server-"cuda:0" or "cuda:1" or "cuda:2" :: default is "cuda:0"')
    parser.add_argument('--seed',                 type=int,  default=3407,     help='random seed init :: default is 3407')
    parser.add_argument('--evalseed',             type=int,  default=32,       help='random seed init for evaluation :: default is 32')
    parser.add_argument('--grad_drop',            type=int,  default=0,        help='lengthwise gradient sample/drop technique :: default is 0 - False')

    # semi-supervised learning
    parser.add_argument('--init_round',    type=int,   default=-1,    help='supervised training step before semi-supervised learning :: default is -1')
    parser.add_argument('--mix_portion',   type=float, default=0.25,  help='unlabeled mixing portion when semi-supervised learning :: default is 0.25')
    parser.add_argument('--mix_extra',     type=bool,  default=True,  help='if True, resample unlabeled portion per each semi-supervised epoch :: default is True')
    parser.add_argument('--conf_thresh',   type=float, default=0.85,  help='Confidence Threshold for Pseudo Labelling :: default is 0.85')
    
    # dataset
    parser.add_argument('--max_length',    type=int,   default=512,   help='max_length through Dataset, Models :: default is 512')
    parser.add_argument('--vocab_size',    type=int,   default=29756, help='vocab_size covering unlabeled, train dataset :: default is 29756 within WGRS_1degree grid cell')
    parser.add_argument('--padding_idx',   type=int,   default=0,     help='Index of "unlabeled_port", "[PAD] token_id" :: default is 0')
    parser.add_argument('--cls_idx',       type=int,   default=1,     help='index of "[CLS] token_id" :: default is 1')
    parser.add_argument('--abs_coord',     type=bool,  default=False, help='use real absolute coordinates rather than relative coordinates if True :: default is False')

    # contrastive augmentation
    parser.add_argument('--aug_ratio',     type=float, default=0.0,  help='Augmentation ratio per instance :: default is 0.0')
    parser.add_argument('--aug_lambda',    type=float, default=1.5,   help='Number of Augmentation, Randomly Chosen within Poisson Distribution :: default is 1.5')
    parser.add_argument('--omitting_rate', type=float, default=0.15,  help='Omit few steps from the main sentence, choose from [0.0~1.0] :: default is 0.15')
    parser.add_argument('--tweaking_rate', type=float, default=0.15,  help='Tweak some navigational routes, choose from [0.0~1.0] :: default is 0.15')

    # model
    parser.add_argument('--n_blocks',    type=int,   default=4,       help='Transformer Block size :: default is 4')
    parser.add_argument('--d_model',     type=int,   default=128,     help='Transformer Embed Size :: default is 128')
    parser.add_argument('--d_ffn',       type=int,   default=256,     help='Transformer FeedForward Inner Embed Size :: default is 256')
    parser.add_argument('--n_heads',     type=int,   default=4,       help='Transformer MultiHeaded Attention Head number :: default is 4')
    parser.add_argument('--d_k',         type=int,   default=64,      help='Transformer Embed Size for each Attention Head :: default is 64')
    parser.add_argument('--temperature', type=float, default=64**0.5, help='Attention Temperature(Div Term), d_k ** 0.5 recommended :: default is 8')
    parser.add_argument('--dropout',     type=float, default=0.3,     help='Model Dropout Probability :: default is 0.3')
    parser.add_argument('--n_class',     type=int,   default=3244,    help='Globally 3243 Ports, 0 for unlabeled :: default is 3244')

    # aggregation
    parser.add_argument('--n_features',  type=int,   default=9,      help='Number of Sub-sequential Features :: default is 9')
    parser.add_argument('--sub_length',  type=int,   default=5,      help='Number of Sampled Points for Sub-sequence :: default is 3')
    parser.add_argument('--n_ship',      type=int,   default=3,      help='Number of Ship Types (tanker02-0, container03-1, bulk04-2) :: default is 3')

    # optimizer
    parser.add_argument('--lr',                type=float, default=1e-4,  help='learning rate :: default is 0.0001')
    parser.add_argument('--beta1',             type=float, default=0.9,   help='beta1 for Adam :: default is 0.9')
    parser.add_argument('--beta2',             type=float, default=0.999, help='beta2 for Adam :: default is 0.999')
    parser.add_argument('--weight_decay',      type=float, default=0.0,   help='weight decay :: default is 0')
    parser.add_argument('--STD_ratio',         type=float, default=0.0,   help='STD learning ratio :: default is 0.0')
    parser.add_argument('--CRD_ratio',         type=float, default=0.0,   help='CRD learning ratio :: default is 0.0')
    parser.add_argument('--EDA_ratio',         type=float, default=0.0,   help='EDA learning ratio :: default is 0.0')
    parser.add_argument('--ETA_ratio',         type=float, default=0.0,   help='ETA learning ratio :: default is 0.0')

    # minor-replacement
    parser.add_argument('--config_update', type=str, default="{}", help='minor config update string to dict :: default is "{}"')
    parser.add_argument('--tag',           type=str, nargs='+', default=None, help="Experiment Tags:: Default is None")

    args = parser.parse_args()

    args.config_update = literal_eval(args.config_update)
    for key in args.config_update.keys():
        setattr(args, key, args.config_update[key])

    print(args)
    return args