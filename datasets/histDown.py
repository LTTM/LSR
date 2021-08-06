import numpy as np
from math import ceil, floor


# define weights to be used, one for each source dataset
gta_weight = np.array([3.6613e-01, 9.0714e-02, 1.8823e-01, 2.0762e-02, 7.1692e-03, 1.1866e-02,
                1.5088e-03, 9.3590e-04, 8.5027e-02, 2.4222e-02, 1.5210e-01, 3.9858e-03,
                3.5832e-04, 2.8775e-02, 1.2881e-02, 4.1957e-03, 7.1898e-04, 3.6897e-04,
                6.2678e-05])
synthia_weight = np.array([0.1916, 0.1993, 0.3021, 0.0028, 0.0028, 0.0108, 0.0004, 0.0011, 0.1071,
                0.0712, 0.0437, 0.0049, 0.0420, 0.0159, 0.0021, 0.0023])
city_13_weight = np.array([0.3849, 0.0634, 0.2398, 0.0022, 0.0058, 0.1670, 0.0427, 0.0129, 0.0014,
                0.0719, 0.0027, 0.0011, 0.0042])


def histDown(im, shape, thresh=0.5, num_classes=19, void_class=-1, out_void=None, conf_map = None, conf_th=.5, use_weights=False):
    assert len(shape) == 2, "Output shape must be (w,h)"
    assert len(im.shape) == 2 or (len(im.shape) == 3 and np.any(im.shape == 1)), "Input array must be 2D"

    if np.any(im.shape == 1):
        dim = np.where(im.shape == 1)
        im = np.squeeze(im, dim)

    dw = im.shape[1]/shape[0]
    assert dw >= 1, "Cannot upsample image along width"
    dh = im.shape[0]/shape[1]
    assert dh >= 1, "Cannot upsample image along height"

    if dw == dh == 1:
        return im

    # duplicate the input, not to override it
    x = im.copy()

    if out_void is None:
        out_void = void_class

    idxs = np.indices(shape[::-1])

    # compute kernel size
    kh, kw = int(np.ceil(dh/2)), int(np.ceil(dw/2))
    # extract image dimensions
    th, tw = x.shape[0]-1, x.shape[1]-1
    # map old indices in new indices
    ch, cw = np.floor(dh*idxs[0,...]).astype(int), np.floor(dw*idxs[1,...]).astype(int)

    # pad the image and change void class id in num_classes
    pad = ((kh,kh),(kw,kw))
    padded = np.pad(x, pad, mode='constant', constant_values=num_classes)
    padded[padded==void_class] = num_classes

    # get the limits for each window, i.e. [h0:h1, w0:w1]
    h0 = ch
    h1 = ch+2*kh+1
    w0 = cw
    w1 = cw+2*kw+1

    # get the indices corresponding to each window
    _kh = np.ones_like(h0)*(2*kh+1)
    _kw = np.ones_like(w0)*(2*kw+1)
    hs = np.repeat(linspace2d(np.stack((h0,h1,_kh))), 2*kh+1, axis=0)
    ws = np.repeat(linspace2d(np.stack((w0,w1,_kw))), 2*kw+1, axis=0)

    # extract the flattened windows (i.e. 50x50 image with stride 2 becomes a 4x25x25 tensor)
    strided = padded[hs,ws,...]
    # apply winDown to each of them
    down = np.apply_along_axis(winDown,
                               0,
                               strided,
                               thresh=thresh,
                               num_classes=num_classes,
                               void_class=void_class,
                               out_void=out_void,
                               use_weights=use_weights).astype(int)

    # if a confidence map is provided,
    # use it to mask the input image
    if conf_map is not None:
        mask = conf_map<conf_th
        down[mask] = void_class

    return down

def lispaceTuple(t):
    ll, ul, steps = t
    return np.linspace(ll, ul, steps, dtype=int)

def linspace2d(x):
    return np.apply_along_axis(lispaceTuple, 0, x)

# computes the downsampling step for a given label window
def winDown(win, thresh, num_classes, void_class, out_void, use_weights=False):
    c = np.bincount(win, minlength=num_classes+1)[:-1]
    if use_weights:
        if num_classes==19:
            c = c/gta_weight
        elif num_classes==16:
            c = c/synthia_weight
        else:
            c = c/city_13_weight
    if c.sum() > 0:
        id1, id0 = np.argsort(c)[-2:]
        if c[id0] < c[id1]:
            id0, id1 = id1, id0
        if c[id1] > 0:
            r = c[id1]/c[id0]
            if r < thresh:
                return id0
            else:
                return out_void
        else:
            return id0
    else:
        return out_void
