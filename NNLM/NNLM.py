import paddle
import paddle.nn as nn
import paddle.optimizer as optim


def make_batch():
    input_batch = []
    target_batch = []

    for sen in sentences:
        word = sen.split()  # space tokenizer
        input = [word_dict[n] for n in word[:-1]]  # create (1~n-1) as input
        target = word_dict[word[-1]]  # create (n) as target, We usually call this 'casual language model'

        input_batch.append(input)
        target_batch.append(target)

    return input_batch, target_batch


class NNLM(paddle.nn.Layer):
    def __init__(self):
        super(NNLM, self).__init__()
        self.C = paddle.nn.Embedding(num_embeddings=n_class, embedding_dim=m)
        self.H = paddle.nn.Linear(in_features=n_step * m, out_features=n_hidden, bias_attr=False)
        self.d = self.create_parameter(shape=[n_hidden], default_initializer=paddle.nn.initializer.Constant(value=1.0))
        self.U = paddle.nn.Linear(in_features=n_hidden, out_features=n_class, bias_attr=False)
        self.W = paddle.nn.Linear(in_features=n_step * m, out_features=n_class, bias_attr=False)
        self.b = self.create_parameter(shape=[n_class], default_initializer=paddle.nn.initializer.Constant(value=1.0))

    def forward(self, X):
        X = self.C(X)  # X : [batch_size, n_step, m]
        X = paddle.reshape(X, shape=[-1, n_step * m])  # [batch_size, n_step * m]
        tanh = paddle.tanh(self.d + self.H(X))  # [batch_size, n_hidden]
        output = self.b + self.W(X) + self.U(tanh)  # [batch_size, n_class]
        return output


if __name__ == '__main__':
    n_step = 2  # number of steps, n-1 in paper
    n_hidden = 2  # number of hidden size, h in paper
    m = 2  # embedding size, m in paper

    sentences = ["i like cat", "i love dog", "i hate milk"]

    word_list = " ".join(sentences).split()
    word_list = list(set(word_list))
    word_dict = {w: i for i, w in enumerate(word_list)}
    number_dict = {i: w for i, w in enumerate(word_list)}
    n_class = len(word_dict)  # number of Vocabulary

    model = NNLM()
    print(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(parameters=model.parameters(), learning_rate=0.001)

    input_batch, target_batch = make_batch()
    input_batch = paddle.to_tensor(input_batch, dtype=paddle.int64, )
    target_batch = paddle.to_tensor(target_batch, dtype=paddle.int64)

    # Training
    for epoch in range(5000):
        optimizer.clear_grad()
        output = model(input_batch)

        # output : [batch_size, n_class], target_batch : [batch_size]
        loss = criterion(output, target_batch)
        if (epoch + 1) % 1000 == 0:
            print('Epoch: %04d, cost = %.6f' % (epoch + 1, loss.numpy()))

        loss.backward()
        optimizer.step()

    # Predict
    predict = paddle.argmax(model(input_batch), axis=1)
    # predict = model(input_batch).data.max(1, keepdim=True)[1]

    # Test
    print([sen.split()[:2] for sen in sentences], '->', [number_dict[n.item()] for n in predict.squeeze()])
