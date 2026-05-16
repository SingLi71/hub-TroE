import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# 第一步：生成数据集和词表
# 我们要写代码自动生成这些“假数据”，并构建一个简单的词表。
# 1. 生成随机字符串和标签
def build_sample():
    # 准备一个简单的字库（除了'你'之外的字）
    chars = "我爱北京天安门吃喝玩乐春夏秋冬"
    # 随机选4个不重复的字
    sampled_chars = random.sample(chars, 4)
    # 随机决定"你"字的位置 (0 到 4)
    target_pos = random.randint(0, 4)

    # 拼接出包含5个字的字符串
    # 先把"你"字放进去
    sampled_chars.insert(target_pos, "你")
    sentence = "".join(sampled_chars)

    return sentence, target_pos


# 2. 生成多条数据
def build_dataset(total_sample_num):
    data = []
    for _ in range(total_sample_num):
        data.append(build_sample())
    return data


# 3. 构建词表 (给每个字一个ID)
def build_vocab():
    chars = "你我爱北京天安门吃喝玩乐春夏秋冬"
    vocab = {"<PAD>": 0, "<UNK>": 1}  # 养成好习惯，留出特殊占位符
    for index, char in enumerate(chars):
        vocab[char] = index + 2  # ID从2开始
    return vocab


# 测试一下生成的数据
sample_sentence, sample_label = build_sample()
print(f"生成的句子: {sample_sentence}, 标签(类别): {sample_label}")
my_vocab = build_vocab()
print(f"生成的词表: {list(my_vocab.items())}")

# 第二步：将文本转换为模型可用的 Tensor
# 模型需要的是数字构成的矩阵（Tensor）。我们需要把刚刚生成的汉字字符串，根据词表转换成整数 ID 序列。
class TextDataset(Dataset):
    def __init__(self, data, vocab):
        self.X = []
        self.y = []
        for sentence, label in data:
            # 将句子里的每个字转换成对应的ID，如果不认识的字就用 <UNK> 的ID 1
            sentence_ids = [vocab.get(char, vocab["<UNK>"]) for char in sentence]
            self.X.append(sentence_ids)
            self.y.append(label)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        # PyTorch 的交叉熵损失函数要求标签是 Long 类型 (整数)
        return torch.LongTensor(self.X[idx]), torch.LongTensor([self.y[idx]])

# 测试一下转换过程
dataset = build_dataset(10)
vocab = build_vocab()
text_dataset = TextDataset(dataset, vocab)
x_sample, y_sample = text_dataset[0]
print(f"\n转换后的输入 X (ID序列): {x_sample}")
print(f"转换后的标签 y: {y_sample}")

# 第三步：构建包含 Embedding 和 RNN 的模型
# 这是最核心的部分。数据流向是这样的：
# 整数 ID 序列 -> Embedding 层 (变成词向量) -> RNN 层 (提取序列特征) -> 全连接层 (分类输出) -> 损失函数
class RnnClassificationModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, num_classes):
        super(RnnClassificationModel, self).__init__()

        # 1. Embedding 层: 将词汇ID映射为稠密向量
        # padding_idx=0 意味着词表中ID为0的那个词（我们设为<PAD>）的向量不会被更新
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim, padding_idx=0)

        # 2. RNN 层: 提取时序特征
        # batch_first=True 意思是输入数据的形状是 (batch_size, sequence_length, embed_dim)
        self.rnn = nn.RNN(input_size=embed_dim, hidden_size=hidden_size, batch_first=True)
        # 注意：上面的 nn.RNN 和 nn.LSTM 的输入输出格式在这里几乎一样。

        # 3. 线性分类层
        # RNN 在跑完整个序列后，我们只取它最后一个时间步的隐藏状态来做分类
        self.linear = nn.Linear(in_features=hidden_size, out_features=num_classes)

        # 4. 损失函数: 因为是多分类，所以用交叉熵
        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        # x的形状: (batch_size, seq_length=5)

        # 经过Embedding
        embed_x = self.embedding(x)
        # embed_x的形状: (batch_size, seq_length=5, embed_dim)

        # 经过RNN
        # RNN的输出有两个:
        # output: 每个时间步的隐藏状态 (batch_size, seq_length, hidden_size)
        # h_n: 最后一个时间步的隐藏状态 (num_layers, batch_size, hidden_size)
        rnn_out, hidden_state = self.rnn(embed_x)

        # 我们做整句分类，通常只取最后一个时间步的特征
        # 因为我们把 batch 放在了第一维 (batch_first=True)
        # 所以可以通过 rnn_out[:, -1, :] 拿到每个样本的最后一个时间步输出
        # 它和 hidden_state.squeeze(0) 在这里是等价的
        last_step_out = rnn_out[:, -1, :]
        # last_step_out的形状: (batch_size, hidden_size)

        # 经过线性分类层
        logits = self.linear(last_step_out)
        # logits的形状: (batch_size, num_classes=5)

        # 计算损失
        if y is not None:
            # 注意：CrossEntropyLoss 期望的 y 是一维的 (batch_size,)
            # 但我们的 DataLoader 通常给出的是 (batch_size, 1)，所以要 squeeze 一下去掉多余的维度
            loss = self.loss_func(logits, y.squeeze(-1))
            return loss
        else:
            return logits


# 我们来过一遍数据看看维度是否正确
vocab = build_vocab()
# 词表大小，embedding维度设为16，隐藏层维度设为32，分为5类
model = RnnClassificationModel(vocab_size=len(vocab), embed_dim=16, hidden_size=32, num_classes=5)

# 假设拿两条数据(batch_size=2)扔进模型
dummy_x = torch.LongTensor([[2, 3, 4, 5, 6], [7, 8, 2, 9, 10]])
dummy_y = torch.LongTensor([[0], [2]])

# 前向传播看看能不能算出loss
loss = model(dummy_x, dummy_y)
print(f"\n模型测试跑通，初始损失 Loss: {loss.item():.4f}")

# 第四步：完整的训练循环与预测
def evaluate(model, val_loader):
    model.eval()  # 切换到评估模式 (关闭Dropout等)
    correct = 0
    total = 0
    with torch.no_grad():  # 评估时不计算梯度，省内存
        for x, y in val_loader:
            logits = model(x)
            # 使用 argmax 找到得分最高的那一类的索引
            predicted_class = torch.argmax(logits, dim=1)
            # 统计正确个数
            correct += (predicted_class == y.squeeze(-1)).sum().item()
            total += y.size(0)

    accuracy = correct / total
    return accuracy


def main():
    # 超参数设置
    EPOCH_NUM = 15
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3
    TRAIN_SAMPLE_NUM = 5000
    VAL_SAMPLE_NUM = 500
    EMBED_DIM = 16
    HIDDEN_SIZE = 32

    print("开始生成数据...")
    vocab = build_vocab()
    train_data = build_dataset(TRAIN_SAMPLE_NUM)
    val_data = build_dataset(VAL_SAMPLE_NUM)

    train_dataset = TextDataset(train_data, vocab)
    val_dataset = TextDataset(val_data, vocab)

    # DataLoader 帮我们自动分批次 (Batch)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 初始化模型
    model = RnnClassificationModel(vocab_size=len(vocab),
                                   embed_dim=EMBED_DIM,
                                   hidden_size=HIDDEN_SIZE,
                                   num_classes=5)  # 0到4位，共5类

    # 优化器，也就是第三周讲到的 Adam
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("开始训练...")
    for epoch in range(EPOCH_NUM):
        model.train()  # 切换到训练模式
        total_loss = 0

        for x, y in train_loader:
            # 1. 梯度清零
            optimizer.zero_grad()
            # 2. 前向传播算损失
            loss = model(x, y)
            # 3. 反向传播算梯度
            loss.backward()
            # 4. 优化器更新参数
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        val_accuracy = evaluate(model, val_loader)

        print(f"Epoch {epoch + 1:2d}/{EPOCH_NUM} - Loss: {avg_loss:.4f} - Val Accuracy: {val_accuracy:.4f}")

    print("\n训练完成，进行人工测试...")
    test_strings = ["北京你玩乐", "你吃春夏秋", "我爱你中国", "冬天吃烤你", "天安门你呀"]
    model.eval()
    with torch.no_grad():
        for s in test_strings:
            # 手动编码
            s_ids = [vocab.get(char, vocab["<UNK>"]) for char in s]
            x_tensor = torch.LongTensor([s_ids])  # 加上 batch 维度

            logits = model(x_tensor)
            pred_class = torch.argmax(logits, dim=1).item()
            print(f"句子: [{s}] -> 模型预测'你'在位置: {pred_class}")


if __name__ == "__main__":
    main()
