from . import func
from .func import IN, OUT

functions = dict(
    add = lambda n = int, a = IN, b = IN, out = OUT : None,

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
)
func.bind(functions)
