import os
import sys
import argparse

from utils import *
from train_model import *

from pre_train.sasrec.data_preprocess import preprocess

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # GPU train options
    parser.add_argument("--multi_gpu", action='store_true')
    parser.add_argument('--gpu_num', type=int, default=0)
    
    # model setting
    parser.add_argument("--llm", type=str, default='opt', help='flan_t5, opt, vicuna, smolvlm')
    parser.add_argument("--recsys", type=str, default='sasrec')
    parser.add_argument(
        "--load_in_4bit",
        action="store_true",
        help="Load LLM with 4-bit quantization to reduce GPU memory.",
    )
    
    # dataset setting
    parser.add_argument("--rec_pre_trained_data", type=str, default='Movies_and_TV')
    
    # train phase setting
    parser.add_argument("--pretrain_stage1", action='store_true')
    parser.add_argument("--pretrain_stage2", action='store_true')
    parser.add_argument("--inference", action='store_true')
    parser.add_argument(
        "--inference_output_file",
        type=str,
        default="./results/smol/recommendation_output_smol_v1_2B.txt",
        help="Path to save inference outputs.",
    )
    
    # hyperparameters options
    parser.add_argument('--batch_size1', default=32, type=int)
    parser.add_argument('--batch_size2', default=2, type=int)
    parser.add_argument('--batch_size_infer', default=32, type=int)
    parser.add_argument('--maxlen', default=50, type=int)
    parser.add_argument('--num_epochs', default=10, type=int)
    parser.add_argument("--stage1_lr", type=float, default=0.0001)
    parser.add_argument("--stage2_lr", type=float, default=0.0001)
    parser.add_argument("--use_wandb", action="store_true", help="Enable Weights & Biases logging.")
    parser.add_argument("--wandb_project", type=str, default="a-llmrec", help="W&B project name.")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="Optional W&B run name.")
    parser.add_argument("--wandb_log_interval", type=int, default=1, help="Log to W&B every N steps.")
    
    args = parser.parse_args()
    
    args.device = 'cuda:' + str(args.gpu_num)
    
    if args.pretrain_stage1:
        train_model_phase1(args)
    elif args.pretrain_stage2:
        train_model_phase2(args)
    elif args.inference:
        inference(args)