# MTCNN Verison of NMS
the comparison of nms in speed

method 1:
thresh=0.7, time wastes:0.0287
thresh=0.8, time wastes:0.1057
thresh=0.9, time wastes:0.4204

method 2:
thresh=0.7, time wastes:0.0272
thresh=0.8, time wastes:0.1038
thresh=0.9, time wastes:0.4184

method 3:
thresh=0.7, time wastes:0.0298
thresh=0.8, time wastes:0.1217
thresh=0.9, time wastes:0.4718

method 4:
thresh=0.7, time wastes:0.0120
thresh=0.8, time wastes:0.0063
thresh=0.9, time wastes:0.0071

Reference:
py-faster-rcnn: https://github.com/rbgirshick/py-faster-rcnn/tree/master/lib/nms
