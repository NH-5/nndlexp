import os
import random
import re
import unicodedata

import numpy as np

EOS = "<eos>"
SOS = "<sos>"
MAX_SEQ_LEN = 10


def unicodeToAscii(s):
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def normalizeString(s):
    s = s.lower().strip()
    s = unicodeToAscii(s)
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


def _pad_or_truncate(ids, max_seq_len):
    max_total_len = max_seq_len + 1
    if len(ids) <= max_total_len:
        return ids + [0] * (max_total_len - len(ids))
    return ids[:max_seq_len] + [0]


def _save_vocab(vocab_path, tokens):
    with open(vocab_path, "w", encoding="utf-8") as f:
        f.write("\n".join(tokens))


def prepare_data(data_path, vocab_save_path, max_seq_len):
    with open(data_path, "r", encoding="utf-8") as f:
        data = f.read().split("\n")

    pairs = []
    for line in data:
        if "\t" not in line:
            continue
        src, tgt = line.split("\t", 1)
        pairs.append((normalizeString(src), tgt.strip()))

    pairs = pairs[:2000]
    en_data = [x[0] for x in pairs]
    ch_data = [x[1] for x in pairs]

    en_vocab = set()
    for line in en_data:
        en_vocab.update(token for token in line.split(" ") if token)
    id2en = [EOS, SOS] + list(en_vocab)
    en2id = {c: i for i, c in enumerate(id2en)}
    en_vocab_size = len(id2en)

    ch_vocab = set("".join(ch_data))
    id2ch = [EOS, SOS] + list(ch_vocab)
    ch2id = {c: i for i, c in enumerate(id2ch)}
    ch_vocab_size = len(id2ch)

    os.makedirs(vocab_save_path, exist_ok=True)
    _save_vocab(os.path.join(vocab_save_path, "en_vocab.txt"), id2en)
    _save_vocab(os.path.join(vocab_save_path, "ch_vocab.txt"), id2ch)

    en_num_data = []
    for line in en_data:
        ids = [1] + [int(en2id[token]) for token in line.split(" ") if token] + [0]
        en_num_data.append(_pad_or_truncate(ids, max_seq_len))

    ch_num_data = []
    for line in ch_data:
        ids = [1] + [int(ch2id[ch]) for ch in line] + [0]
        ch_num_data.append(_pad_or_truncate(ids, max_seq_len))

    return (
        np.asarray(en_num_data, dtype=np.int64),
        np.asarray(ch_num_data, dtype=np.int64),
        en_vocab_size,
        ch_vocab_size,
    )


def convert_to_mindrecord(data_path, mindrecord_save_path, max_seq_len):
    en_num_data, ch_num_data, en_vocab_size, ch_vocab_size = prepare_data(
        data_path, mindrecord_save_path, max_seq_len
    )

    total = len(en_num_data)
    eval_size = min(20, total)
    eval_indices = random.sample(range(total), eval_size) if eval_size > 0 else []

    train_path = os.path.join(mindrecord_save_path, "gru_train.npz")
    eval_path = os.path.join(mindrecord_save_path, "gru_eval.npz")
    np.savez(train_path, encoder_data=en_num_data, decoder_data=ch_num_data)

    if eval_indices:
        np.savez(
            eval_path,
            encoder_data=en_num_data[eval_indices],
            decoder_data=ch_num_data[eval_indices],
        )
    else:
        np.savez(eval_path, encoder_data=en_num_data, decoder_data=ch_num_data)

    print("en_vocab_size:", en_vocab_size)
    print("ch_vocab_size:", ch_vocab_size)
    return en_vocab_size, ch_vocab_size


if __name__ == "__main__":
    convert_to_mindrecord("src/cmn_zhsim.txt", "./preprocess", MAX_SEQ_LEN)
