import paddle

if __name__ == '__main__':

    paddle.enable_static()
    W = paddle.static.create_parameter(shape=[784, 200], dtype='float32')
    print(W)