import random
import re
import unicodedata
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


RUN_CONFIG = {
    "command": "train",  # preprocess / train / eval / predict
    "data_file": "./exp3/ReferenceCode/src/cmn_zhsim.txt",
    "preprocess_dir": "./exp3/preprocess_pytorch",
    "checkpoint_dir": "./exp3/checkpoints_pytorch",
    "checkpoint_name": "seq2seq_gru.pt",
    "device": "auto",  # auto / cuda / mps / cpu
    "seed": 42,
    "num_samples": 2000,
    "max_seq_length": 10,
    "hidden_size": 512,
    "batch_size": 16,
    "eval_batch_size": 1,
    "learning_rate": 1e-3,
    "num_epochs": 15,
    "train_split": 0.99,
    "print_every": 50,
    "predict_sentence": "i am a student .",
}

PAD_ID = 0
SOS_ID = 1
EOS_TOKEN = "<eos>"
SOS_TOKEN = "<sos>"


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_name)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def unicode_to_ascii(text: str) -> str:
    return "".join(
        char for char in unicodedata.normalize("NFD", text) if unicodedata.category(char) != "Mn"
    )


def normalize_english(text: str) -> str:
    text = unicode_to_ascii(text.lower().strip())
    text = re.sub(r"([.!?])", r" \1", text)
    text = re.sub(r"[^a-zA-Z.!?]+", r" ", text)
    return re.sub(r"\s+", " ", text).strip()


def pad_or_truncate(ids, max_seq_len):
    max_total_len = max_seq_len + 1
    if len(ids) <= max_total_len:
        return ids + [PAD_ID] * (max_total_len - len(ids))
    return ids[:max_seq_len] + [PAD_ID]


def build_args(config):
    normalized = dict(config)
    for key in {"data_file", "preprocess_dir", "checkpoint_dir"}:
        normalized[key] = Path(normalized[key])
    return type("Config", (), normalized)()


def save_vocab(vocab_path: Path, tokens) -> None:
    vocab_path.write_text("\n".join(tokens), encoding="utf-8")


def load_vocab(vocab_path: Path):
    return vocab_path.read_text(encoding="utf-8").splitlines()


def preprocess_dataset(args):
    data_file = args.data_file
    output_dir = args.preprocess_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    lines = data_file.read_text(encoding="utf-8").splitlines()
    pairs = []
    for line in lines:
        if "\t" not in line:
            continue
        src, tgt = line.split("\t", 1)
        src = normalize_english(src)
        tgt = tgt.strip()
        if src and tgt:
            pairs.append((src, tgt))

    pairs = pairs[: args.num_samples]
    english_sentences = [pair[0] for pair in pairs]
    chinese_sentences = [pair[1] for pair in pairs]

    en_vocab = sorted({token for sentence in english_sentences for token in sentence.split() if token})
    ch_vocab = sorted(set("".join(chinese_sentences)))
    id2en = [EOS_TOKEN, SOS_TOKEN] + en_vocab
    id2ch = [EOS_TOKEN, SOS_TOKEN] + ch_vocab
    en2id = {token: idx for idx, token in enumerate(id2en)}
    ch2id = {token: idx for idx, token in enumerate(id2ch)}

    encoder_data = []
    for sentence in english_sentences:
        ids = [SOS_ID] + [en2id[token] for token in sentence.split()] + [PAD_ID]
        encoder_data.append(pad_or_truncate(ids, args.max_seq_length))

    decoder_data = []
    for sentence in chinese_sentences:
        ids = [SOS_ID] + [ch2id[token] for token in sentence] + [PAD_ID]
        decoder_data.append(pad_or_truncate(ids, args.max_seq_length))

    encoder_data = np.asarray(encoder_data, dtype=np.int64)
    decoder_data = np.asarray(decoder_data, dtype=np.int64)

    total = len(encoder_data)
    train_size = max(1, int(total * args.train_split))
    train_size = min(train_size, total - 1) if total > 1 else total
    train_encoder = encoder_data[:train_size]
    train_decoder = decoder_data[:train_size]
    eval_encoder = encoder_data[train_size:] if total > train_size else encoder_data[: min(20, total)]
    eval_decoder = decoder_data[train_size:] if total > train_size else decoder_data[: min(20, total)]

    np.savez(output_dir / "gru_train.npz", encoder_data=train_encoder, decoder_data=train_decoder)
    np.savez(output_dir / "gru_eval.npz", encoder_data=eval_encoder, decoder_data=eval_decoder)
    save_vocab(output_dir / "en_vocab.txt", id2en)
    save_vocab(output_dir / "ch_vocab.txt", id2ch)

    print(f"preprocess done: {output_dir}")
    print(f"total pairs: {total}")
    print(f"train pairs: {len(train_encoder)}")
    print(f"eval pairs: {len(eval_encoder)}")
    print(f"en vocab size: {len(id2en)}")
    print(f"ch vocab size: {len(id2ch)}")


def ensure_preprocessed(args):
    required = [
        args.preprocess_dir / "gru_train.npz",
        args.preprocess_dir / "gru_eval.npz",
        args.preprocess_dir / "en_vocab.txt",
        args.preprocess_dir / "ch_vocab.txt",
    ]
    if all(path.exists() for path in required):
        return
    preprocess_dataset(args)


def create_dataloader(args, is_training: bool):
    data_path = args.preprocess_dir / ("gru_train.npz" if is_training else "gru_eval.npz")
    with np.load(data_path) as data:
        encoder_data = data["encoder_data"]
        decoder_data = data["decoder_data"]

    if is_training:
        encoder_input = encoder_data[:, 1:]
        decoder_input = decoder_data[:, :-1]
        decoder_target = decoder_data[:, 1:]
        dataset = TensorDataset(
            torch.from_numpy(encoder_input),
            torch.from_numpy(decoder_input),
            torch.from_numpy(decoder_target),
        )

        def collate_fn(batch):
            return {
                "encoder_data": torch.stack([item[0] for item in batch]),
                "decoder_data": torch.stack([item[1] for item in batch]),
                "target_data": torch.stack([item[2] for item in batch]),
            }

        return DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=collate_fn,
        )

    encoder_input = encoder_data[:, 1:]
    decoder_input = decoder_data[:, :-1]
    dataset = TensorDataset(torch.from_numpy(encoder_input), torch.from_numpy(decoder_input))

    def collate_fn(batch):
        return {
            "encoder_data": torch.stack([item[0] for item in batch]),
            "decoder_data": torch.stack([item[1] for item in batch]),
        }

    return DataLoader(
        dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_fn,
    )


class Seq2SeqGRU(nn.Module):
    def __init__(self, en_vocab_size, ch_vocab_size, hidden_size, max_seq_length):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_seq_length = max_seq_length
        self.encoder_embedding = nn.Embedding(en_vocab_size, hidden_size)
        self.decoder_embedding = nn.Embedding(ch_vocab_size, hidden_size)
        self.encoder_gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.decoder_gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.output_layer = nn.Linear(hidden_size, ch_vocab_size)

    def encode(self, src):
        embedded = self.encoder_embedding(src)
        _, hidden = self.encoder_gru(embedded)
        return hidden

    def forward(self, src, tgt):
        hidden = self.encode(src)
        embedded = self.decoder_embedding(tgt)
        output, _ = self.decoder_gru(embedded, hidden)
        logits = self.output_layer(output)
        return logits

    def greedy_decode(self, src, max_len):
        hidden = self.encode(src)
        decoder_input = torch.full((src.size(0), 1), SOS_ID, dtype=torch.long, device=src.device)
        outputs = []
        for _ in range(max_len):
            embedded = self.decoder_embedding(decoder_input)
            output, hidden = self.decoder_gru(embedded, hidden)
            logits = self.output_layer(output[:, -1:, :])
            next_token = logits.argmax(dim=-1)
            outputs.append(next_token)
            decoder_input = next_token
        return torch.cat(outputs, dim=1)


def load_meta(args):
    en_vocab = load_vocab(args.preprocess_dir / "en_vocab.txt")
    ch_vocab = load_vocab(args.preprocess_dir / "ch_vocab.txt")
    return en_vocab, ch_vocab


def build_model(args):
    en_vocab, ch_vocab = load_meta(args)
    model = Seq2SeqGRU(
        en_vocab_size=len(en_vocab),
        ch_vocab_size=len(ch_vocab),
        hidden_size=args.hidden_size,
        max_seq_length=args.max_seq_length,
    )
    return model, en_vocab, ch_vocab


def save_checkpoint(model, args, en_vocab, ch_vocab):
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.checkpoint_dir / args.checkpoint_name
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "en_vocab_size": len(en_vocab),
            "ch_vocab_size": len(ch_vocab),
            "hidden_size": args.hidden_size,
            "max_seq_length": args.max_seq_length,
        },
        checkpoint_path,
    )
    print(f"checkpoint saved to: {checkpoint_path}")


def load_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    return checkpoint


def ids_to_english(ids, vocab):
    tokens = []
    for idx in ids:
        if idx == PAD_ID:
            break
        if idx == SOS_ID:
            continue
        tokens.append(vocab[idx])
    return " ".join(tokens)


def ids_to_chinese(ids, vocab):
    chars = []
    for idx in ids:
        if idx == PAD_ID:
            break
        if idx == SOS_ID:
            continue
        chars.append(vocab[idx])
    return "".join(chars)


def train(args):
    set_seed(args.seed)
    ensure_preprocessed(args)
    device = resolve_device(args.device)
    print(f"using device: {device}")

    train_loader = create_dataloader(args, is_training=True)
    model, en_vocab, ch_vocab = build_model(args)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.98))

    for epoch in range(1, args.num_epochs + 1):
        model.train()
        total_loss = 0.0
        for step, batch in enumerate(train_loader, start=1):
            src = batch["encoder_data"].to(device)
            tgt = batch["decoder_data"].to(device)
            label = batch["target_data"].to(device)

            optimizer.zero_grad()
            logits = model(src, tgt)
            loss = criterion(logits.reshape(-1, logits.size(-1)), label.reshape(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if step % args.print_every == 0 or step == len(train_loader):
                avg_loss = total_loss / step
                print(f"epoch {epoch}/{args.num_epochs} step {step}/{len(train_loader)} loss {avg_loss:.6f}")

    save_checkpoint(model, args, en_vocab, ch_vocab)


@torch.no_grad()
def evaluate(args):
    ensure_preprocessed(args)
    device = resolve_device(args.device)
    print(f"using device: {device}")

    eval_loader = create_dataloader(args, is_training=False)
    model, en_vocab, ch_vocab = build_model(args)
    model = model.to(device)
    checkpoint_path = args.checkpoint_dir / args.checkpoint_name
    load_checkpoint(model, checkpoint_path, device)
    model.eval()

    for batch in eval_loader:
        src = batch["encoder_data"].to(device)
        tgt = batch["decoder_data"].to(device)
        pred = model.greedy_decode(src, args.max_seq_length).cpu()

        english = ids_to_english(batch["encoder_data"][0].tolist(), en_vocab)
        expect_chinese = ids_to_chinese(batch["decoder_data"][0].tolist(), ch_vocab)
        predict_chinese = ids_to_chinese(pred[0].tolist(), ch_vocab)

        print("English:", english)
        print("expect Chinese:", expect_chinese)
        print("predict Chinese:", predict_chinese)
        print()


@torch.no_grad()
def predict(args):
    ensure_preprocessed(args)
    device = resolve_device(args.device)
    print(f"using device: {device}")

    model, en_vocab, ch_vocab = build_model(args)
    model = model.to(device)
    checkpoint_path = args.checkpoint_dir / args.checkpoint_name
    load_checkpoint(model, checkpoint_path, device)
    model.eval()

    sentence = normalize_english(args.predict_sentence)
    en2id = {token: idx for idx, token in enumerate(en_vocab)}
    tokens = sentence.split()
    ids = [SOS_ID] + [en2id.get(token, PAD_ID) for token in tokens] + [PAD_ID]
    ids = pad_or_truncate(ids, args.max_seq_length)
    src = torch.tensor([ids[1:]], dtype=torch.long, device=device)
    pred = model.greedy_decode(src, args.max_seq_length).cpu()[0].tolist()

    print("input English:", sentence)
    print("predict Chinese:", ids_to_chinese(pred, ch_vocab))


def main():
    args = build_args(RUN_CONFIG)
    if args.command == "preprocess":
        preprocess_dataset(args)
    elif args.command == "train":
        train(args)
    elif args.command == "eval":
        evaluate(args)
    elif args.command == "predict":
        predict(args)
    else:
        raise ValueError(f"unknown command: {args.command}")


if __name__ == "__main__":
    main()
