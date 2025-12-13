import numpy as np

class Conv2D:
    def __init__(self, num_filters, filter_size, input_depth):
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.input_depth = input_depth

        scale = np.sqrt(2.0 / (filter_size * filter_size * input_depth))
        self.W = np.random.randn(
            num_filters, input_depth, filter_size, filter_size
        ) * scale
        self.b = np.zeros((num_filters, 1))

    def im2col(self, x):
        N, C, H, W = x.shape
        F = self.filter_size
        out_h = H - F + 1
        out_w = W - F + 1

        cols = np.zeros((C * F * F, N * out_h * out_w))
        idx = 0

        for i in range(out_h):
            for j in range(out_w):
                patch = x[:, :, i:i+F, j:j+F]
                cols[:, idx:idx+N] = patch.reshape(N, -1).T
                idx += N

        return cols, out_h, out_w

    def forward(self, x):
        self.x = x
        self.cols, self.out_h, self.out_w = self.im2col(x)

        W_col = self.W.reshape(self.num_filters, -1)
        out = (W_col @ self.cols + self.b).T

        out = out.reshape(self.out_h, self.out_w, x.shape[0], self.num_filters)
        return out.transpose(2, 3, 0, 1)

    def backward(self, d_out, lr):
        N = self.x.shape[0]
        d_out = d_out.transpose(2, 3, 0, 1).reshape(-1, self.num_filters)

        dW = d_out.T @ self.cols.T
        db = np.sum(d_out, axis=0).reshape(self.num_filters, 1)

        self.W -= lr * dW.reshape(self.W.shape)
        self.b -= lr * db

        return np.zeros_like(self.x)  # input gradient not needed further
