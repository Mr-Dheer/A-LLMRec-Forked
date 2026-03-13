import os
import sys
import argparse

from utils import *
from train_model import (
    train_model_phase1,
    train_model_phase2,
    inference,
    train_model_phase2_id,
    inference_id,
)

from pre_train.sasrec.data_preprocess import preprocess

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # GPU train options
    parser.add_argument("--multi_gpu", action='store_true')
    parser.add_argument('--gpu_num', type=int, default=0)
    
    # model setting
    parser.add_argument("--llm", type=str, default='opt', help='flan_t5, opt, vicuna')
    parser.add_argument("--recsys", type=str, default='sasrec')
    
    # dataset setting
    parser.add_argument("--rec_pre_trained_data", type=str, default='Movies_and_TV')
    
    # train phase setting
    parser.add_argument("--pretrain_stage1", action='store_true')
    parser.add_argument("--pretrain_stage2", action='store_true')
    parser.add_argument("--inference", action='store_true')
    # ID-prediction mode: replaces text-generation objective with item-ID CE loss.
    parser.add_argument("--id_prediction", action='store_true',
                        help="Use ID-prediction head instead of text-generation for Stage 2.")
    
    # hyperparameters options
    parser.add_argument('--batch_size1', default=32, type=int)
    parser.add_argument('--batch_size2', default=1, type=int)
    parser.add_argument('--batch_size_infer', default=2, type=int)
    parser.add_argument('--maxlen', default=50, type=int)
    parser.add_argument('--num_epochs', default=10, type=int)
    parser.add_argument("--stage1_lr", type=float, default=0.0001)
    parser.add_argument("--stage2_lr", type=float, default=0.0001)
    
    args = parser.parse_args()
    
    args.device = 'cuda:' + str(args.gpu_num)
    
    if args.pretrain_stage1:
        train_model_phase1(args)
    elif args.pretrain_stage2 and args.id_prediction:
        train_model_phase2_id(args)
    elif args.pretrain_stage2:
        train_model_phase2(args)
    elif args.inference and args.id_prediction:
        inference_id(args)
    elif args.inference:
        inference(args)