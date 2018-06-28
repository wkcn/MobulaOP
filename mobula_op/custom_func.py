from . import func
from .func import IN, OUT

functions = dict(
    add = lambda n = int, a = IN, b = IN, out = OUT : None,
    sub = lambda n = int, a = IN, b = IN, out = OUT : None,
    mul = lambda n = int, a = IN, b = IN, out = OUT : None,
    div = ('div_', lambda n = int, a = IN, b = IN, out = OUT : None),
    abs = ('abs_', lambda n = int, a = IN, out = OUT : None),
    dot_add = lambda a = IN, b = IN, I = int, U = int, K = int, M = int, out = OUT : None,
    print_carray = lambda data = [float] : None,
    assign_carray = lambda data = [float], out = OUT : None,
    assign_val = lambda n = int, val = float, out = OUT : None,
    sum = lambda n = int, data = [IN], out = OUT : None,
    transpose = lambda data = IN, shape = [int], axes = [int], out = OUT : None,

    roi_align_forward = lambda n = int,
        bottom_data = IN,
        spatial_scale = float,
        channels = int,
        height = int,
        width = int,
        pooled_height = int,
        pooled_width = int,
        sampling_ratio = int,
        bottom_rois = IN,
        top_data = OUT : None,

    roi_align_backward = lambda n = int,
        top_diff = IN,
        num_rois = int,
        spatial_scale = float,
        channels = int,
        height = int,
        width = int,
        pooled_height = int,
        pooled_width = int,
        sampling_ratio = int,
        bottom_diff = OUT,
        bottom_rois = OUT : None,

    softmax_forward = lambda data = IN,
        num_classes = int,
        outer_size = int,
        inner_size = int,
        probs = OUT : None,

    softmax_loss_forward = lambda probs = IN,
        labels = IN,
        num_classes = int,
        outer_size = int,
        inner_size = int,
        losses = OUT : None,

    softmax_loss_backward = lambda probs = IN,
        labels = IN,
        num_classes = int,
        outer_size = int,
        inner_size = int,
        dX = OUT : None,

    im2col = lambda data_im = IN, channels = int, height = int, width = int, kernel_h = int, kernel_w = int, pad_h = int, pad_w = int, stride_h = int, stride_w = int, dilation_h = int, dilation_w = int, data_col = OUT : None,

    col2im = lambda data_col = IN, channels = int, height = int, width = int, kernel_h = int, kernel_w = int, pad_h = int, pad_w = int, stride_h = int, stride_w = int, dilation_h = int, dilation_w = int, data_im = OUT : None,

    linalg_gemm_ff = lambda a = IN, b = IN, I = int, U = int, J = int, out = OUT : None,
    linalg_gemm_ft = lambda a = IN, b = IN, I = int, U = int, J = int, out = OUT : None,
    linalg_gemm_tf = lambda a = IN, b = IN, I = int, U = int, J = int, out = OUT : None,
    linalg_gemm_tt = lambda a = IN, b = IN, I = int, U = int, J = int, out = OUT : None,

)
func.bind(functions)
