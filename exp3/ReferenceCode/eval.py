"""Evaluate GRU translation model with PyTorch."""

import argparse
import os

import torch

from src.config import cfg
from src.dataset import create_dataset
from src.preprocess import convert_to_mindrecord
from src.seq2seq import InferCell, Seq2Seq


def _ensure_dataset(dataset_path):
    train_file = os.path.join(dataset_path, "gru_train.npz")
    eval_file = os.path.join(dataset_path, "gru_eval.npz")
    en_vocab = os.path.join(dataset_path, "en_vocab.txt")
    ch_vocab = os.path.join(dataset_path, "ch_vocab.txt")
    if all(os.path.exists(p) for p in [train_file, eval_file, en_vocab, ch_vocab]):
        return
    convert_to_mindrecord("src/cmn_zhsim.txt", dataset_path, cfg.max_seq_length)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch GRU Example")
    parser.add_argument("--dataset_path", type=str, default="./preprocess", help="dataset path.")
    parser.add_argument("--checkpoint_path", type=str, default="", help="checkpoint path.")
    args = parser.parse_args()

    os.makedirs(args.dataset_path, exist_ok=True)
    _ensure_dataset(args.dataset_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds_eval = create_dataset(args.dataset_path, cfg.eval_batch_size, is_training=False)

    network = Seq2Seq(cfg, is_train=False)
    network = InferCell(network, cfg).to(device)
    network.eval()

    checkpoint_path = args.checkpoint_path or cfg.checkpoint_path
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    network.network.load_state_dict(state_dict)

    with open(os.path.join(args.dataset_path, "en_vocab.txt"), "r", encoding="utf-8") as f:
        en_vocab = f.read().split("\n")

    with open(os.path.join(args.dataset_path, "ch_vocab.txt"), "r", encoding="utf-8") as f:
        ch_vocab = f.read().split("\n")

    with torch.no_grad():
        for data in ds_eval:
            src = data["encoder_data"].to(device)
            dst = data["decoder_data"].to(device)

            en_data = ""
            ch_data = ""

            for x in data["encoder_data"][0].tolist():
                if x == 0:
                    break
                en_data += en_vocab[x]
                en_data += " "

            for x in data["decoder_data"][0].tolist():
                if x == 0:
                    break
                if x == 1:
                    continue
                ch_data += ch_vocab[x]

            output = network(src, dst)
            print("English:", en_data)
            print("expect Chinese:", ch_data)

            out = ""
            for x in output[0].tolist():
                if x == 0:
                    break
                out += ch_vocab[x]
            print("predict Chinese:", out)
            print(" ")
