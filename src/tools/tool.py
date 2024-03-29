import argparse

import torch
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
            description="LP-PN")

    parser.add_argument("--data_path", type=str,
                        default="data/reuters.json",
                        help="path to dataset")
    parser.add_argument("--dataset", type=str, default="reuters",
                        help="name of the dataset. "
                        "Options: [20newsgroup, amazon, huffpost, "
                        "reuters]")
    parser.add_argument("--n_train_class", type=int, default=15,
                        help="number of meta-train classes")
    parser.add_argument("--n_val_class", type=int, default=5,
                        help="number of meta-val classes")
    parser.add_argument("--n_test_class", type=int, default=11,
                        help="number of meta-test classes")

    parser.add_argument("--n_workers", type=int, default=10,
                        help="Num. of cores used for loading data. Set this "
                        "to zero if you want to use all the cpus.")

    parser.add_argument("--way", type=int, default=5,
                        help="#classes for each task")
    parser.add_argument("--shot", type=int, default=5,
                        help="#support examples for each class for each task")
    parser.add_argument("--query", type=int, default=100,
                        help="#query examples for each class for each task")

    parser.add_argument("--train_epochs", type=int, default=1000,
                        help="max num of training epochs")
    parser.add_argument("--train_episodes", type=int, default=100,
                        help="#tasks sampled during each training epoch")
    parser.add_argument("--val_episodes", type=int, default=100,
                        help="#asks sampled during each validation epoch")
    parser.add_argument("--test_episodes", type=int, default=1000,
                        help="#tasks sampled during each testing epoch")

    parser.add_argument("--wv_path", type=str,
                        default='pretrain_wordvec',
                        help="path to word vector cache")
    parser.add_argument("--word_vector", type=str, default='pretrain_wordvec/wiki.en.vec',
                        help=("Name of pretrained word embeddings."))
    parser.add_argument("--finetune_ebd", action="store_true", default=False,
                        help=("Finetune embedding during meta-training"))

    parser.add_argument("--embedding", type=str, default="bilstm",
                        help=("document embedding method."))
    

    parser.add_argument("--seed", type=int, default=123, help="seed")
    parser.add_argument("--dropout", type=float, default=0.1, help="drop rate")
    parser.add_argument("--patience", type=int, default=20, help="patience")
    parser.add_argument("--clip_grad", type=float, default=None,
                        help="gradient clipping")
    parser.add_argument("--cuda", type=int, default=-1,
                        help="cuda device, -1 for cpu")
    parser.add_argument("--mode", type=str, default="test",
                        help=("Running mode."
                              "Options: [train, test]"
                              "[Default: test]"))
    parser.add_argument("--save", action="store_true", default=False,
                        help="train the model")
    parser.add_argument("--notqdm", action="store_true", default=False,
                        help="disable tqdm")
    parser.add_argument("--result_path", type=str, default="")
    parser.add_argument("--snapshot", type=str, default="",
                        help="path to the pretraiend weights")
    parser.add_argument("--pretrain", type=str, default=None, help="path to the pretraiend weights for bilstm")
    parser.add_argument("--n", type=int, default=None, help="Number of iterations of the model")
    parser.add_argument("--lr_g", type=float, default=1e-3, help="learning rate of G")
    parser.add_argument("--lr_d", type=float, default=1e-3, help="learning rate of D")
    parser.add_argument("--lr_scheduler", type=str, default=None, help="lr_scheduler")
  
    parser.add_argument("--train_mode", type=str, default=None, help="you can choose t_add_v or None")
 
    parser.add_argument("--path_drawn_data", type=str, default="reuters_False_data.json", help="path_drawn_data")

    parser.add_argument("--id2word", default=None, help="id2word")

    parser.add_argument("--bert", default=False, action="store_true",
                        help=("set true if use bert embeddings "))
  
  


    parser.add_argument('--g',          type=int,   default=20,         metavar='G',
                    help="top g in constructing the graph W")
    parser.add_argument('--rn',         type=int,   default=300,        metavar='RN',
                    help="graph construction types: "
                    "300: alpha is fixed" +
                    "30:  alpha learned")
    parser.add_argument('--alpha',      type=float, default=0.99,       metavar='ALPHA',
                    help="Initial alpha in label propagation")
    
    parser.add_argument('--k',          type=int,   default=20,         metavar='K',
                    help="top k in label propagation to Prototypes")
    
    parser.add_argument('--mu',          type=int,   default=1,         metavar='mu',
                    help="Loss function trade off parameters")
    
    
    

    return parser.parse_args()


def print_args(args):
    """
        Print arguments (only show the relevant arguments)
    """
    print("\nParameters:")
    for attr, value in sorted(args.__dict__.items()):
        print("\t{}={}".format(attr.upper(), value))

def set_seed(seed):
    """
        Setting random seeds
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)


def load_model_state_dict(model, model_path):
    model_dict = model.state_dict()
    pretrained_dict = torch.load(model_path)
    keys = []
    for k, v in pretrained_dict.items():
           keys.append(k)

    i = 0

    print("_____________pretrain_parameters______________________________")
    for k, v in model_dict.items():
        if v.size() == pretrained_dict[keys[i]].size():
            model_dict[k] = pretrained_dict[keys[i]]
            print(model_dict[k])
            i = i + 1
        # print(model_dict[k])
    print("___________________________________________________________")
    model.load_state_dict(model_dict)
    return model