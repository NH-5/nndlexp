"""Train GRU translation model with PyTorch."""

import argparse
import os
import time

import torch

from src.config import cfg
from src.dataset import create_dataset
from src.preprocess import convert_to_mindrecord
from src.seq2seq import Seq2Seq, WithLossCell


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
    parser.add_argument("--ckpt_save_path", type=str, default="./ckpt", help="checkpoint save path.")
    args = parser.parse_args()

    os.makedirs(args.dataset_path, exist_ok=True)
    os.makedirs(args.ckpt_save_path, exist_ok=True)
    _ensure_dataset(args.dataset_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds_train = create_dataset(args.dataset_path, cfg.batch_size)

    network = Seq2Seq(cfg) #根据预设参数构建模型
    network = WithLossCell(network, cfg).to(device) #记录单个批尺寸数据集的损失值
    optimizer = torch.optim.Adam(network.parameters(), lr=cfg.learning_rate, betas=(0.9, 0.98)) #使用Adam优化器

    saved_ckpts = []
    for epoch in range(1, cfg.num_epochs + 1):
        network.train()
        epoch_start = time.time()

        for step, data in enumerate(ds_train, start=1):
            src = data["encoder_data"].to(device)
            dst = data["decoder_data"].to(device)
            label = data["target_data"].to(device)

            optimizer.zero_grad()
            loss = network(src, dst, label)
            loss.backward()
            optimizer.step()

            if step % cfg.save_checkpoint_steps == 0:
                save_name = f"gru-{epoch}_{step}.pt"
                save_path = os.path.join(args.ckpt_save_path, save_name)
                torch.save({"model_state_dict": network._backbone.state_dict()}, save_path)
                saved_ckpts.append(save_path)
                if cfg.keep_checkpoint_max and len(saved_ckpts) > cfg.keep_checkpoint_max:
                    old_path = saved_ckpts.pop(0)
                    if os.path.exists(old_path):
                        os.remove(old_path)
                print(f"epoch: {epoch} step: {step}, loss is {loss.item():.7f}")

        epoch_time_ms = (time.time() - epoch_start) * 1000.0
        per_step_time_ms = epoch_time_ms / max(len(ds_train), 1)
        print(f"epoch time: {epoch_time_ms:.3f} ms, per step time: {per_step_time_ms:.3f} ms")
