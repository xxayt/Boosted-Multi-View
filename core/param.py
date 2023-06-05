import argparse
import random
import numpy as np
import torch


def parse_args():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--name", required=True, type=str, help="Name of this run. Used for monitoring.")
    parser.add_argument('--save_dir', default='./logs', type=str)
    parser.add_argument('--device_name', type=str, default='torch.cuda.get_device_name(0)')
    # data
    parser.add_argument("--data", choices=["Caltech101-7", "Caltech101-20"], default="Caltech101-7", help="Which downstream task.")
    parser.add_argument('--data_path', type=str, default="./data")
    parser.add_argument('--num_classes', type=int, default=2)
    # train
    parser.add_argument("--warmup_epochs", default=3, type=int, help="unimodal warmup epoch.")
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument("--train_batch_size", default=32, type=int, help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=32, type=int, help="Total batch size for eval.")
    parser.add_argument("--lr", default=5e-2, type=float, help="The initial learning rate for SGD.")
    parser.add_argument("--max_accuracy", default=0.0, type=float)
    parser.add_argument("--TwoClass", action='store_true', help="choose two class.")
    parser.add_argument("--seed", default=42, type=int, help="Random seed for initialization.")
    parser.add_argument("--weight_decay", default=0, type=float, help="Weight deay if we apply some.")
    parser.add_argument("--decay_type", choices=["cosine", "linear"], default="cosine", help="How to decay the learning rate.")
    # model
    # parser.add_argument("--pretrain_dir", type=str, default="./pretrain", help="Where to search for pretrained ViT models.")
    # parser.add_argument('--pretrain', type=str, default="ViT-B_16.npz", help='vit_base_patch16_224_in21k.pth')
    # parser.add_argument('--model_file', type=str, default='modeling')
    # parser.add_argument("--model_name", default="ViT-B_16", help="Which variant to use.")

    # Training configuration
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_workers", default=8, type=int, help="Number of workers to use.")

    # Parse the arguments.
    args = parser.parse_args()

    # Set seeds
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    return args


args = parse_args()
