import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_scatter import scatter_max, scatter_min


def generate_input_representation(events, event_representation, shape, nr_temporal_bins=5, separate_pol=True, img=None):
    """
    Events: N x 4, where cols are x, y, t, polarity, and polarity is in {-1, 1}. x and y correspond to image
    coordinates u and v.
    """
    if event_representation == 'histogram':
        return generate_event_histogram(events, shape)
    elif event_representation == 'voxel_grid':
        return generate_voxel_grid(events, shape, nr_temporal_bins, separate_pol)
    elif event_representation == 'voxel_grid_sep':
        return generate_voxel_grid(events, shape, nr_temporal_bins, separate_pol=True)
    elif event_representation == 'AET':
        return generate_activity_enhanced_tensor(events, shape, nr_temporal_bins)
    elif event_representation == 'event_token':
        return generate_event_token(events, shape, nr_temporal_bins, separate_pol=separate_pol)
    elif event_representation == 'event_pillars':
        return generate_event_pillars(events, shape, nr_temporal_bins)
    elif event_representation == 'edge_guide_grid':
        return generate_voxel_grid_edge_weighted(events, shape, nr_temporal_bins, img, separate_pol)
    elif event_representation == 'edge_assisted_grid':
        return generate_voxel_grid_edge_weighted(events, shape, nr_temporal_bins, img, True)


def generate_activity_enhanced_tensor(events, size, channel_num):
    """
    Func:
        build an Activity-Enhanced Tensor (AET) from a set of events
    Args:
        events: N x 4, where the form of each row is [x, y, timestamp, polarity]
        size: spatial resolution of event cameras
        channel_num: number of bins in the temporal axis
    Returns:
        An Activity-Enhanced Tensor (AET) with a size of [channel_num * 2, height, width]
    """
    nr_temporal_bins = channel_num
    (height, width) = size
    voxel_grid_positive = np.zeros((nr_temporal_bins, height, width), np.float32).ravel()
    voxel_grid_negative = np.zeros((nr_temporal_bins, height, width), np.float32).ravel()

    # normalize the event timestamps so that they lie between 0 and num_bins
    last_stamp = events[-1, 2]
    first_stamp = events[0, 2]
    deltaT = last_stamp - first_stamp

    if deltaT == 0:
        deltaT = 1.0

    xs = events[:, 0].astype(np.int32)
    ys = events[:, 1].astype(np.int32)
    ts = (nr_temporal_bins - 1) * (events[:, 2] - first_stamp) / deltaT

    pols = events[:, 3]
    pols[pols == 0] = -1  # polarity should be +1 / -1

    tis = ts.astype(np.int32)
    dts = ts - tis
    vals_left = np.abs(pols) * (1.0 - dts)
    vals_right = np.abs(pols) * dts
    pos_events_indices = pols == 1

    # Positive Voxels Grid
    valid_indices_pos = np.logical_and(tis < nr_temporal_bins, pos_events_indices)
    valid_pos = (xs < width) & (xs >= 0) & (ys < height) & (ys >= 0) & (ts >= 0) & (ts < nr_temporal_bins)
    valid_indices_pos = np.logical_and(valid_indices_pos, valid_pos)

    np.add.at(voxel_grid_positive, xs[valid_indices_pos] + ys[valid_indices_pos] * width +
              tis[valid_indices_pos] * width * height, vals_left[valid_indices_pos])

    valid_indices_pos = np.logical_and((tis + 1) < nr_temporal_bins, pos_events_indices)
    valid_indices_pos = np.logical_and(valid_indices_pos, valid_pos)
    np.add.at(voxel_grid_positive, xs[valid_indices_pos] + ys[valid_indices_pos] * width +
              (tis[valid_indices_pos] + 1) * width * height, vals_right[valid_indices_pos])

    # Negative Voxels Grid
    valid_indices_neg = np.logical_and(tis < nr_temporal_bins, ~pos_events_indices)
    valid_indices_neg = np.logical_and(valid_indices_neg, valid_pos)

    np.add.at(voxel_grid_negative, xs[valid_indices_neg] + ys[valid_indices_neg] * width +
              tis[valid_indices_neg] * width * height, vals_left[valid_indices_neg])

    valid_indices_neg = np.logical_and((tis + 1) < nr_temporal_bins, ~pos_events_indices)
    valid_indices_neg = np.logical_and(valid_indices_neg, valid_pos)
    np.add.at(voxel_grid_negative, xs[valid_indices_neg] + ys[valid_indices_neg] * width +
              (tis[valid_indices_neg] + 1) * width * height, vals_right[valid_indices_neg])

    voxel_grid_positive = np.reshape(voxel_grid_positive, (nr_temporal_bins, height, width))
    voxel_grid_negative = np.reshape(voxel_grid_negative, (nr_temporal_bins, height, width))

    voxel_grid = voxel_grid_positive - voxel_grid_negative

    # Activity Map
    map = np.zeros((nr_temporal_bins, height, width), np.float32).ravel()
    count_left = np.ceil(1.0 - dts)
    count_right = np.ceil(dts)
    valid_indices_count = tis < nr_temporal_bins
    valid_count = (xs < width) & (xs >= 0) & (ys < height) & (ys >= 0) & (ts >= 0) & (ts < nr_temporal_bins)
    valid_indices_count = np.logical_and(valid_indices_count, valid_count)

    np.add.at(map, xs[valid_indices_count] + ys[valid_indices_count] * width +
              tis[valid_indices_count] * width * height, count_left[valid_indices_count])

    valid_indices_count = (tis + 1) < nr_temporal_bins
    valid_indices_count = np.logical_and(valid_indices_count, valid_count)
    np.add.at(map, xs[valid_indices_count] + ys[valid_indices_count] * width +
              (tis[valid_indices_count] + 1) * width * height, count_right[valid_indices_count])

    map = np.reshape(map, (nr_temporal_bins, height, width))

    return np.concatenate((voxel_grid, map), axis=0)


def generate_event_histogram(events, shape):
    """
    Events: N x 4, where cols are x, y, t, polarity, and polarity is in {-1, 1}. x and y correspond to image
    coordinates u and v.
    """
    events = events[events[:, 0] >= 0]
    events = events[events[:, 0] <= shape[1]]
    events = events[events[:, 1] >= 0]
    events = events[events[:, 1] <= shape[0]]
    height, width = shape
    x, y, t, p = events.T
    x = x.astype(np.int32)
    y = y.astype(np.int32)
    p[p == 0] = -1  # polarity should be +1 / -1
    img_pos = np.zeros((height * width,), dtype="float32")
    img_neg = np.zeros((height * width,), dtype="float32")

    np.add.at(img_pos, x[p == 1] + width * y[p == 1], 1)
    np.add.at(img_neg, x[p == -1] + width * y[p == -1], 1)

    histogram = np.stack([img_neg, img_pos], 0).reshape((2, height, width))

    return histogram


def normalize_voxel_grid(events):
    """Normalize event voxel grids"""
    nonzero_ev = (events != 0)
    num_nonzeros = nonzero_ev.sum()
    if num_nonzeros > 0:
        # compute mean and stddev of the **nonzero** elements of the event tensor
        # we do not use PyTorch's default mean() and std() functions since it's faster
        # to compute it by hand than applying those funcs to a masked array
        mean = events.sum() / num_nonzeros
        stddev = torch.sqrt((events ** 2).sum() / num_nonzeros - mean ** 2)
        mask = nonzero_ev.float()
        events = mask * (events - mean) / stddev

    return events


def generate_voxel_grid(events, shape, nr_temporal_bins, separate_pol=True):
    """
    Build a voxel grid with bilinear interpolation in the time domain from a set of events.
    :param events: a [N x 4] NumPy array containing one event per row in the form: [timestamp, x, y, polarity]
    :param nr_temporal_bins: number of bins in the temporal axis of the voxel grid
    :param shape: dimensions of the voxel grid
    """
    height, width = shape
    assert(events.shape[1] == 4)
    assert(nr_temporal_bins > 0)
    assert(width > 0)
    assert(height > 0)

    voxel_grid_positive = np.zeros((nr_temporal_bins, height, width), np.float32).ravel()
    voxel_grid_negative = np.zeros((nr_temporal_bins, height, width), np.float32).ravel()

    # normalize the event timestamps so that they lie between 0 and num_bins
    last_stamp = events[-1, 2]
    first_stamp = events[0, 2]
    deltaT = last_stamp - first_stamp

    if deltaT == 0:
        deltaT = 1.0

    # events[:, 2] = (nr_temporal_bins - 1) * (events[:, 2] - first_stamp) / deltaT
    xs = events[:, 0].astype(np.int32)
    ys = events[:, 1].astype(np.int32)
    # ts = events[:, 2]
    # print(ts[:10])
    ts = (nr_temporal_bins - 1) * (events[:, 2] - first_stamp) / deltaT

    pols = events[:, 3]
    pols[pols == 0] = -1  # polarity should be +1 / -1

    tis = ts.astype(np.int32)
    dts = ts - tis
    vals_left = np.abs(pols) * (1.0 - dts)
    vals_right = np.abs(pols) * dts
    pos_events_indices = pols == 1

    # Positive Voxels Grid
    valid_indices_pos = np.logical_and(tis < nr_temporal_bins, pos_events_indices)
    valid_pos = (xs < width) & (xs >= 0) & (ys < height) & (ys >= 0) & (ts >= 0) & (ts < nr_temporal_bins)
    valid_indices_pos = np.logical_and(valid_indices_pos, valid_pos)

    np.add.at(voxel_grid_positive, xs[valid_indices_pos] + ys[valid_indices_pos] * width +
              tis[valid_indices_pos] * width * height, vals_left[valid_indices_pos])

    valid_indices_pos = np.logical_and((tis + 1) < nr_temporal_bins, pos_events_indices)
    valid_indices_pos = np.logical_and(valid_indices_pos, valid_pos)
    np.add.at(voxel_grid_positive, xs[valid_indices_pos] + ys[valid_indices_pos] * width +
              (tis[valid_indices_pos] + 1) * width * height, vals_right[valid_indices_pos])

    # Negative Voxels Grid
    valid_indices_neg = np.logical_and(tis < nr_temporal_bins, ~pos_events_indices)
    valid_indices_neg = np.logical_and(valid_indices_neg, valid_pos)

    np.add.at(voxel_grid_negative, xs[valid_indices_neg] + ys[valid_indices_neg] * width +
              tis[valid_indices_neg] * width * height, vals_left[valid_indices_neg])

    valid_indices_neg = np.logical_and((tis + 1) < nr_temporal_bins, ~pos_events_indices)
    valid_indices_neg = np.logical_and(valid_indices_neg, valid_pos)
    np.add.at(voxel_grid_negative, xs[valid_indices_neg] + ys[valid_indices_neg] * width +
              (tis[valid_indices_neg] + 1) * width * height, vals_right[valid_indices_neg])

    voxel_grid_positive = np.reshape(voxel_grid_positive, (nr_temporal_bins, height, width))
    voxel_grid_negative = np.reshape(voxel_grid_negative, (nr_temporal_bins, height, width))

    if separate_pol:
        return np.concatenate([voxel_grid_positive, voxel_grid_negative], axis=0)

    voxel_grid = voxel_grid_positive - voxel_grid_negative
    return voxel_grid

# ==========================================================

def generate_event_token(events, shape, nr_temporal_bins, group_num=12, patch_size=4, separate_pol=True):
    """
    Events: N x 4, where cols are x, y, t, polarity, and polarity is in {0, 1}. x and y correspond to image
    """
    # print("Generating event token")
    # print(events[0])
    # print(f"\nThe range of Polarity: {events[:, 3].min()} - {events[:, 3].max()}")

    height, width = shape
    time_div = group_num // 2
    patch_size = patch_size if isinstance(patch_size, tuple) else (patch_size, patch_size)

    assert(events.shape[1] == 4)
    assert(nr_temporal_bins > 0)
    assert(width > 0)
    assert(height > 0)

    PH, PW = int((height) / patch_size[0]), int((width) / patch_size[1])
    Token_num, Patch_size, b = int(PH * PW), int(patch_size[0] * patch_size[1]), 1e-4
    
    y = np.zeros([time_div, 2, 2, Patch_size, Token_num], dtype=np.float32)

    if len(events):
        # 获取极性p对应的events序列掩码
        mask = events[:, 3] == 1
        wt = (events[:, 2] - events[0, 2]) / (events[-1, 2] - events[0, 2] + b)  # 归一化时间戳
        Position = np.floor(events[:, 0] / ((width-1) / PW + b)) + \
            np.floor(events[:, 1] / ((height-1) / PH + b)) * PW
        Token = np.floor(events[:, 0] % ((width-1) / PW + b)) + \
            np.floor(events[:, 1] % ((height-1) / PH + b)) * int(((width-1) + 1) / PW)
        t_double = events[:, 2].astype(np.float64)
        DTime = np.floor(time_div * (t_double - t_double[0]) / (t_double[-1] - t_double[0] + 1))

        # Mapping from 4-D to 1-D.
        bins = np.array([time_div, 2, Patch_size, Token_num], dtype=np.int32)
        x_nd = np.stack([DTime, events[:, 3], Token, Position], axis=1).astype(np.int32).T  # 转置后形状为 [4, N]
        x_1d, index = index_mapping(x_nd, bins)
        print(f"DTime: {DTime.shape}, events[:, 3]: {events[:, 3].shape}, bins: {bins}, Token: {Token.shape}, Position: {Position.shape}, x_nd: {x_nd.shape}\
              \nx_1d: {x_1d.shape}, index: {index.shape}\
              \n The range of x_1d: {x_1d.min()} - {x_1d.max()}, Shape: {x_1d.shape}")
        
        y[:, :, 0, :, :], y[:, :, 1, :, :] = get_repr(x_1d, index, bins=bins, weights=[mask, wt])

    # 将y的形状转换为 [1, -1, PH, PW]并返回
    y = y.reshape([1, -1, PH, PW])
    print("y shape:", y.shape)

    return y


        # print(f"\nEvents: {events.shape}, Mask: {mask.shape}  \
        #       \nThe range of x: {events[:, 0].min()} - {events[:, 0].max()}, Shape: {events[:, 0].shape}\
        #       \nThe range of y: {events[:, 1].min()} - {events[:, 1].max()}, Shape: {events[:, 1].shape}\
        #       \nThe range of T: {wt.min()} - {wt.max()}, Shape: {wt.shape}\
        #       \nThe range of Polarity: {events[:, 3].min()} - {events[:, 3].max()}\
        #       \nThe range of Position: {Position.min()} - {Position.max()}, Shape: {Position.shape}\
        #       \nThe range of Token: {Token.min()} - {Token.max()}, Shape: {Token.shape}\
        #       \nThe range of DTime: {DTime.min()} - {DTime.max()}, Shape: {DTime.shape}")


    # if separate_pol:
    #     return np.concatenate([voxel_grid_positive, voxel_grid_negative], axis=0)

    # voxel_grid = voxel_grid_positive - voxel_grid_negative
    # return voxel_grid



# def index_mapping(sample, bins=None):
#     """
#     Multi-index mapping method from N-D to 1-D.
#     sample: 4D tensor, shape: [4, N]
#     bins: 1D tensor, shape: [4], the number of bins in each dimension
#     """
#     device = sample.device
#     bins = torch.as_tensor(bins).to(device)
#     y = torch.max(sample, torch.zeros([], device=device, dtype=torch.int32))
#     y = torch.min(y, bins.reshape(-1, 1))
#     index = torch.ones_like(bins)
#     index[1:] = torch.cumprod(torch.flip(bins[1:], [0]), -1).int()
#     index = torch.flip(index, [0])
#     l = torch.sum((index.reshape(-1, 1)) * y, 0)
#     return l, index

def index_mapping(sample, bins=None):
    """
    Multi-index mapping method from N-D to 1-D using numpy.
    sample: 2D array, shape: [4, N]
    bins: 1D array, shape: [4], the number of bins in each dimension
    """
    bins = np.asarray(bins)
    y = np.maximum(sample, np.zeros_like(sample, dtype=np.int32))
    y = np.minimum(y, bins.reshape(-1, 1))
    index = np.ones_like(bins, dtype=np.int32)
    index[1:] = np.cumprod(bins[:0:-1])[::-1]
    l = np.sum(index.reshape(-1, 1) * y, axis=0)
    return l, index


# def get_repr(l, index, bins=None, weights=None):
#     """
#     Function to return histograms.
#     """
#     hist = torch.bincount(l, minlength=index[0] * bins[0], weights=weights[0])
#     hist = hist.reshape(tuple(bins))
#     if len(weights) > 1:
#         hist2 = torch.bincount(l, minlength=index[0] * bins[0], weights=weights[1])
#         hist2 = hist2.reshape(tuple(bins))
#     else:
#         return hist
#     if len(weights) > 2:
#         hist3 = torch.bincount(l, minlength=index[0] * bins[0], weights=weights[2])
#         hist3 = hist3.reshape(tuple(bins))
#     else:
#         return hist, hist2
#     if len(weights) > 3:
#         hist4 = torch.bincount(l, minlength=index[0] * bins[0], weights=weights[3])
#         hist4 = hist4.reshape(tuple(bins))
#     else:
#         return hist, hist2, hist3
#     return hist, hist2, hist3, hist4

def get_repr(l, index, bins=None, weights=None):
    """
    Function to return histograms using numpy.
    """
    hist = np.bincount(l, minlength=index[0] * bins[0], weights=weights[0])
    hist = hist.reshape(tuple(bins))
    if len(weights) > 1:
        hist2 = np.bincount(l, minlength=index[0] * bins[0], weights=weights[1])
        hist2 = hist2.reshape(tuple(bins))
    else:
        return hist
    if len(weights) > 2:
        hist3 = np.bincount(l, minlength=index[0] * bins[0], weights=weights[2])
        hist3 = hist3.reshape(tuple(bins))
    else:
        return hist, hist2
    if len(weights) > 3:
        hist4 = np.bincount(l, minlength=index[0] * bins[0], weights=weights[3])
        hist4 = hist4.reshape(tuple(bins))
    else:
        return hist, hist2, hist3
    return hist, hist2, hist3, hist4

# ==========================================================

# ======================EventPillars========================
def generate_event_pillars(events, shape, nr_temporal_bins=5):
    """
    Events: N x 4, where cols are x, y, t, polarity, and polarity is in {0, 1}. x and y correspond to image
    """
    height, width = shape
    assert(events.shape[1] == 4)
    assert(nr_temporal_bins > 0)
    assert(width > 0)
    assert(height > 0)

    index = torch.tensor(events[:, 0] + events[:, 1] * width).to(torch.int64)
    pos = events[events[:, 3] == 1]
    neg = events[events[:, 3] == 0]
    pos = torch.from_numpy(pos)
    neg = torch.from_numpy(neg)

    Histogram_neg = torch.bincount(neg[:, 0].long() + neg[:, 1].long() * width,
                                    minlength=height * width).reshape(height, width)
    Histogram_pos = torch.bincount(pos[:, 0].long() + pos[:, 1].long() * width,
                                    minlength=height * width).reshape(height, width)
    
    t_sequence = events[:, 2] - events[:, 2][0]
    t_sequence = torch.tensor(t_sequence)
    t_values = (t_sequence * 255).clamp(0, 255).byte()


    # Tmax
    Tmax_out, _ = scatter_max(t_values, index, dim=-1, dim_size=height * width)
    Tmax = Tmax_out.reshape(height, width)

    # Tmin
    Tmin_out, _ = scatter_min(t_values, index, dim=-1, dim_size=height * width)
    Tmin = Tmin_out.reshape(height, width)

    tanh = nn.Tanh()
    AEP = tanh(Histogram_neg - Histogram_pos)
    num_events = Histogram_pos + Histogram_neg
    ED = num_events / num_events.max()
    # result = torch.stack([Tmax, Tmin, AEP, ED], dim=2)
    # 将Tmax, Tmin, AEP, ED拼接成一个tensor, shape为[4, height, width]
    result = torch.stack([Tmax, Tmin, AEP, ED], dim=0)

    event_surface = result.float()
    # print(f"pos: {pos.shape}, neg: {neg.shape}, events: {events.shape}\
    #       \nResult: {result.shape}, Histogram_pos: {Histogram_pos.shape}, Histogram_neg: {Histogram_neg.shape}")

    return event_surface.numpy()

# ==========================================================

def _generate_voxel_grid_with_weight(events, shape, nr_temporal_bins, weight_map, separate_pol=True):
    height, width = shape
    voxel_grid_positive = np.zeros((nr_temporal_bins, height, width), np.float32).ravel()
    voxel_grid_negative = np.zeros((nr_temporal_bins, height, width), np.float32).ravel()

    last_stamp = events[-1, 2]
    first_stamp = events[0, 2]
    deltaT = last_stamp - first_stamp
    if deltaT == 0: deltaT = 1.0

    xs = events[:, 0].astype(np.int32)
    ys = events[:, 1].astype(np.int32)
    ts = (nr_temporal_bins - 1) * (events[:, 2] - first_stamp) / deltaT
    pols = events[:, 3] 
    pols[pols == 0] = -1

    tis = ts.astype(np.int32)
    dts = ts - tis

    # Apply weight
    w = weight_map[ys, xs]  # 获取对应像素的权重
    vals_left = np.abs(pols) * (1.0 - dts) * w
    vals_right = np.abs(pols) * dts * w

    pos_events_indices = pols == 1
    valid_pos = (xs < width) & (xs >= 0) & (ys < height) & (ys >= 0) & (ts >= 0) & (ts < nr_temporal_bins)

    # Positive
    valid_indices_pos = np.logical_and(tis < nr_temporal_bins, pos_events_indices)
    valid_indices_pos = np.logical_and(valid_indices_pos, valid_pos)
    np.add.at(voxel_grid_positive, xs[valid_indices_pos] + ys[valid_indices_pos] * width +
              tis[valid_indices_pos] * width * height, vals_left[valid_indices_pos])
    valid_indices_pos = np.logical_and((tis + 1) < nr_temporal_bins, pos_events_indices)
    valid_indices_pos = np.logical_and(valid_indices_pos, valid_pos)
    np.add.at(voxel_grid_positive, xs[valid_indices_pos] + ys[valid_indices_pos] * width +
              (tis[valid_indices_pos] + 1) * width * height, vals_right[valid_indices_pos])

    # Negative
    valid_indices_neg = np.logical_and(tis < nr_temporal_bins, ~pos_events_indices)
    valid_indices_neg = np.logical_and(valid_indices_neg, valid_pos)
    np.add.at(voxel_grid_negative, xs[valid_indices_neg] + ys[valid_indices_neg] * width +
              tis[valid_indices_neg] * width * height, vals_left[valid_indices_neg])
    valid_indices_neg = np.logical_and((tis + 1) < nr_temporal_bins, ~pos_events_indices)
    valid_indices_neg = np.logical_and(valid_indices_neg, valid_pos)
    np.add.at(voxel_grid_negative, xs[valid_indices_neg] + ys[valid_indices_neg] * width +
              (tis[valid_indices_neg] + 1) * width * height, vals_right[valid_indices_neg])

    voxel_grid_positive = np.reshape(voxel_grid_positive, (nr_temporal_bins, height, width))
    voxel_grid_negative = np.reshape(voxel_grid_negative, (nr_temporal_bins, height, width))

    if separate_pol:
        return np.concatenate([voxel_grid_positive, voxel_grid_negative], axis=0)
    else:
        return voxel_grid_positive - voxel_grid_negative

def normalize_weight(x, min_val=0.5, max_val=1.5):
    x_min = x.min()
    x_max = x.max()
    norm = (x - x_min) / (x_max - x_min + 1e-5)
    return min_val + norm * (max_val - min_val)

def generate_voxel_grid_edge_weighted(events, shape, nr_temporal_bins, img, separate_pol=True):
    height, width = shape
    sobel_x = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32).view(1,1,3,3)
    sobel_y = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=torch.float32).view(1,1,3,3)
    edge_x = F.conv2d(img, sobel_x, padding=1)
    edge_y = F.conv2d(img, sobel_y, padding=1)
    edge = torch.sqrt(edge_x ** 2 + edge_y ** 2)
    w = normalize_weight(edge.squeeze())

    return _generate_voxel_grid_with_weight(events, shape, nr_temporal_bins, w.cpu().numpy(), separate_pol)

