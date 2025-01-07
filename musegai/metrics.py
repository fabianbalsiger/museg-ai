""" compute segmentation metrics """
import csv
import numpy as np

"""
- multi-label/global DSC?
"""

class Process:
    def __init__(self, ref, pred, *, labels=None, ignore=None, options={}, skip_zero=True):
        self.ref = np.asarray(ref).astype(int)
        self.pred = np.asarray(pred).astype(int)
        self.spacing = np.array(getattr(ref, 'spacing', (1,) * self.ref.ndim))
        self.ignore = ignore
        self.labels = dict(labels) or {i: f'label {i}' for i in np.unique(ref)}
        if skip_zero:
            self.labels.pop(0, None)
        self.options = options

    def __call__(self, metrics, *, subset=None, disp=True):
        labels = {i: self.labels[i] for i in subset or self.labels}
        rows = []
        for i in labels:
            if disp:
                print(f"Label: {labels[i]} ({i}):")
            ref_mask = self.ref == i
            pred_mask = self.pred == i
            bin_metrics = BinaryMetrics(ref_mask, pred_mask, ignore=self.ignore, spacing=self.spacing, options=self.options)
            row = {'label': labels[i]}
            for metric in metrics:
                row[metric] = bin_metrics(metric)
                print(f"\tmetric: {metric}={row[metric]}")
            rows.append(row)
        return rows
                

class BinaryMetrics:
    def __init__(self, ref, pred, *, spacing=None, ignore=None, options={}):
        self.ref = np.asarray(ref) > 0
        self.pred = np.asarray(pred) > 0
        self.ignore = ignore
        self.options = options

        self.is_ref = self.ref.sum() > 0
        self.is_pred = self.pred.sum() > 0
        self.keep = True if self.ignore is None else ~(np.asarray(self.ignore) > 0)
        self.spacing = np.array((1,) * self.ref.ndim if spacing is None else spacing)

        self.metrics = [method for method in dir(BinaryMetrics) if not method.startswith('__')]

    def __call__(self, fun):
        if not fun in self.metrics:
            raise ValueError(f'Unknown metric: {fun}')
        return getattr(self, fun)()
    

    # overlap
    
    def iou(self):
        """ intersection over union"""
        return (self.ref & self.pred & self.keep).sum() / (self.keep & (self.ref | self.pred)).sum()
        
    def dsc(self):
        iou = self.iou()
        return 2 * iou / (1 + iou)
    
    
    # boundaries

    def b_iou(self):
        """ boundary intersection over union"""
        d = self.options['b_iou.d']
        bd_ref = boundary(self.ref, rd=d, ignore=self.ignore)
        bd_pred = boundary(self.pred, rd=d, ignore=self.ignore)
        return (bd_ref & bd_pred).sum() / (bd_ref | bd_pred).sum()


    def nds(self):
        """ normalized surface distance """
        tau = self.options['nds.tau']
        d = tau / self.spacing
        bd_ref = boundary(self.ref, ignore=self.ignore)
        br_ref = border_region(self.ref, d=d, ignore=self.ignore)
        bd_pred = boundary(self.pred, ignore=self.ignore)
        br_pred = border_region(self.pred, d=d, ignore=self.ignore)

        numerator = (bd_ref & br_pred).sum() + (bd_pred & br_ref).sum()
        denominator = bd_ref.sum() + bd_pred.sum()
        return numerator / denominator
    
    def hdx(self, pc=None):
        """ xth-percentile hausdorff distance """
        if not (self.is_ref and self.is_pred):
            return None
        pc = pc or self.options['xhd.pc']
        coords = (np.indices(self.ref.shape).T * self.spacing).T
        bd_ref = coords[:, boundary(self.ref, ignore=self.ignore)]
        bd_pred = coords[:, boundary(self.pred, ignore=self.ignore)]
        dist = np.zeros((bd_ref.shape[1], bd_pred.shape[1]))
        for i in range(len(coords)):
            dist += (bd_ref[i, :, np.newaxis] - bd_pred[i])**2
        dist_ref = np.min(dist, axis=1)
        dist_pred = np.min(dist, axis=0)
        return max(np.percentile(dist_ref, pc)**0.5, np.percentile(dist_pred, pc)**0.5)
    
    def hd95(self):
        """ 95th-percentile hausdorff distance """
        return self.hdx(pc=95)
        



#
# functions

def boundary(mask, *, rd=1, ignore=None):
    if ignore is not None:
        mask = mask & ~ignore
    return mask != binary_erosion(mask, rd=rd, ignore=ignore)

def border_region(mask, *, d=0, ignore=None):
    bdmap = boundary(mask, ignore=ignore)
    d = np.ones(bdmap.ndim, dtype=int) * d
    if ~np.any(d > 0):
        return bdmap
    return binary_dilation(bdmap, rd=d, ignore=ignore)

def binary_dilation(mask, *, rd=1, ignore=None):
    ndim = mask.ndim
    ignore = True if ignore is None else ~(np.asarray(ignore) > 0)
    # kernel
    rd = np.round(np.ones(ndim) * rd).astype(int)
    ker = (np.sum(np.abs(np.indices(tuple(2 * rd + 1)).T - rd), axis=-1) <= rd.max()).T
    diff = np.array(mask.shape) - np.array(ker.shape)
    ker = np.pad(ker, [(-(-d//2), d//2) for d in diff])
    # fft convolution
    fft_mask = np.fft.fftn(1.0 * mask)
    fft_ker = np.fft.fftn(1.0 * np.fft.fftshift(ker))
    filtered = (np.fft.ifftn(fft_mask * fft_ker).real * ignore) > 1e-8
    return filtered

def binary_erosion(mask, *, rd=1, ignore=None):
    mask = ~mask if ignore is None else ~mask & ~ignore
    dil = binary_dilation(mask, rd=rd, ignore=ignore)
    return ~dil if ignore is None else ~dil & ~ignore
    