"""Network config settings."""


class EasyDict(dict):
    """A tiny EasyDict replacement for attribute-style access."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = value


cfg = EasyDict(
    {
        'en_vocab_size': 1154, #英文字典的大小，也就是英文的 subword 的个数
        'ch_vocab_size': 1116, #中文字典的大小
        'max_seq_length': 10, #字数的个数
        'hidden_size': 1024, #隐藏单元数
        'batch_size': 16, #批尺寸大小
        'eval_batch_size': 1,
        'learning_rate': 0.001, #学习率
        'momentum': 0.9, #动量优化器参数
        'num_epochs': 15,#训练全部数据集迭代次数
        'save_checkpoint_steps': 125, #每隔这么多步骤保存检查点
        'keep_checkpoint_max': 10, #要保留的最近检查点文件的最大数量.当新文件被创建时,旧文件被删除.如果为None或0,则保留所有检查点文件.默认为5(也就是保留5个最近的检查点文件.)
        'dataset_path':'./preprocess', #预处理路径
        'ckpt_save_path':'./ckpt', #储存模型的位置
        'checkpoint_path':'./ckpt/gru-15_125.ckpt' #储存检查点的位置
    }
)
