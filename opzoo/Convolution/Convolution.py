import mobula
from mobula.const import req


@mobula.op.register
class Conv2D:
    def __init__(self, channels, kernel_size, strides=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1):
        self.channels = channels
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.dilation = dilation
        assert groups == 1
        self.groups = groups

    def forward(self, x, weight, bias=None):
        # y = wx + b
        N, C, H, W = x.shape
        KH, KW = self.kernel_size
        PH, PW = self.padding
        SH, SW = self.strides
        DH, DW = self.dilation
        _, D, OH, OW = self.y.shape
        csize = C * KH * KW
        data_col = self.F.empty((csize, OH, OW))
        rweight = weight.reshape((D, csize))
        rbias = bias.reshape((-1, 1, 1)) if bias is not None else None
        for i in range(N):
            mobula.func.im2col(
                data_col.size, x[i], H, W, KH, KW, PH, PW, SH, SW, DH, DW, OH, OW, data_col)
            out = self.F.dot(rweight, data_col).reshape((D, OH, OW))
            if rbias is not None:
                out += rbias
            self.assign(self.y[i], self.req[0], out)

    def backward(self, dy):
        N, C, H, W = self.dx.shape
        KH, KW = self.kernel_size
        PH, PW = self.padding
        SH, SW = self.strides
        DH, DW = self.dilation
        _, D, OH, OW = dy.shape
        csize = C * KH * KW
        ohw = OH * OW
        weightT = self.X[1].reshape((D, csize)).T
        out = self.F.empty_like(self.dx[0])
        dw = 0
        for i in range(N):
            rdy = dy[i].reshape((D, ohw))
            data_col = self.F.dot(weightT, rdy)
            mobula.func.col2im(
                self.dx[0].size, data_col, H, W, KH, KW, PH, PW, SH, SW, DH, DW, OH, OW, out)
            self.assign(self.dX[0][i], self.req[0], out)
            mobula.func.im2col(
                data_col.size, self.x[i], H, W, KH, KW, PH, PW, SH, SW, DH, DW, OH, OW, data_col)
            dw += self.F.dot(rdy, data_col.T)
        self.assign(self.dX[1], self.req[1], dw.reshape_like(self.dX[1]))
        if len(self.X) == 3:
            self.assign(self.dX[2], self.req[2], dy.sum(1, exclude=True))

    def infer_shape(self, in_shape):
        assert 2 <= len(
            in_shape) <= 3, "The inputs should be feature map(NCHW layout), weight and bias(optional)"
        assert len(in_shape[0]) == 4, "input: NCHW"
        assert len(in_shape[1]) == 4, "weight: DCKK"
        assert len(in_shape) == 2 or len(in_shape[2]) == 1, "bias: D"
        x, weight = in_shape[:2]
        N, C, H, W = x
        KH, KW = self.kernel_size
        PH, PW = self.padding
        SH, SW = self.strides
        DH, DW = self.dilation

        def get_outsize(X, K, P, S, D):
            K += (K - 1) * (D - 1)
            return (X + 2 * P - K) // S + 1
        OH = get_outsize(H, KH, PH, SH, DH)
        OW = get_outsize(W, KW, PW, SW, DW)
        D = self.channels
        assert weight[0] == D
        assert weight[1] == C
        assert weight[2] == KH
        assert weight[3] == KW
        return in_shape, [(N, D, OH, OW)]
