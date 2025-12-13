import numpy as np

class MaxPool2D:
    def __init__(self, size=2):
        self.size = size

    def forward(self, x):
        self.x = x
        N, C, H, W = x.shape
        S = self.size

        out_h = H // S
        out_w = W // S

        # Reshape to pooling windows
        x_reshaped = x.reshape(N, C, out_h, S, out_w, S)

        # Compute max
        out = x_reshaped.max(axis=(3, 5))

        # Store argmax indices (CRITICAL FIX)
        self.argmax = x_reshaped.reshape(
            N, C, out_h, out_w, S * S
        ).argmax(axis=4)

        return out

    def backward(self, d_out):
        N, C, out_h, out_w = d_out.shape
        S = self.size

        dx = np.zeros_like(self.x)

        # Flatten pooling windows
        dx_reshaped = dx.reshape(
            N, C, out_h, S, out_w, S
        )

        for i in range(out_h):
            for j in range(out_w):
                idx = self.argmax[:, :, i, j]
                for n in range(N):
                    for c in range(C):
                        r = idx[n, c] // S
                        c2 = idx[n, c] % S
                        dx_reshaped[n, c, i, r, j, c2] = d_out[n, c, i, j]

        return dx
