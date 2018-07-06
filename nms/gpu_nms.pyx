
# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# Modified By Zijie Xie
# --------------------------------------------------------


import numpy as np
cimport numpy as np
from libcpp cimport bool
assert sizeof(int) == sizeof(np.int32_t)

cdef extern from "gpu_nms.hpp":
    void _nms(np.int32_t*, int*, np.float32_t*, int, int, float, bool, int)

def gpu_nms(np.ndarray[np.float32_t, ndim=2] dets, np.float thresh, type='Union',
            np.int32_t device_id=0):
    cdef int boxes_num = dets.shape[0]
    cdef int boxes_dim = dets.shape[1]
    cdef int num_out
    cdef np.ndarray[np.int32_t, ndim=1] \
        keep = np.zeros(boxes_num, dtype=np.int32)
    cdef np.ndarray[np.float32_t, ndim=1] \
        scores = dets[:, 4]
    cdef np.ndarray[np.int_t, ndim=1] \
        order = scores.argsort()[::-1]
    cdef np.ndarray[np.float32_t, ndim=2] \
        sorted_dets = dets[order, :]
    cdef bool mode_s=False
    if type == 'Min':
        mode_s = True
    else:
        mode_s = False
    _nms(&keep[0], &num_out, &sorted_dets[0, 0], boxes_num, boxes_dim, thresh, mode_s, device_id)
    keep = keep[:num_out]
    return list(order[keep])


