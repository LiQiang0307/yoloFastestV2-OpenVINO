'''
Descripttion: OpenVINO 测试视频文件
version: 
Author: LiQiang
Date: 2021-11-21 15:08:51
LastEditTime: 2021-11-21 15:29:13
'''
from numpy.core.fromnumeric import shape
from numpy.lib.type_check import imag
from openvino.inference_engine import IECore
import cv2
import numpy as np
import os
import cv2
import time
import argparse

import torch
from torch import tensor
from torch.nn.functional import pairwise_distance
import utils.utils
cfg = utils.utils.load_datafile('G:\深度学习实训\Yolo-FastestV2\data\coco.data')
"""
CPU
GNA
GPU
"""
ie = IECore()
for device in ie.available_devices:
    print(device)
with open("G:\深度学习实训\Yolo-FastestV2\data\coco.names")as f:
    labels = [line.strip() for line in f.readlines()]


model_xml = "G:\深度学习实训\Yolo-FastestV2\openVINO\yolo-fastestv2.xml"
model_bin = "G:\深度学习实训\Yolo-FastestV2\openVINO\yolo-fastestv2.bin"

net = ie.read_network(model=model_xml, weights=model_bin)
input_blob = next(iter(net.inputs))
out_blob = next(iter(net.outputs))

n, c, h, w = net.inputs[input_blob].shape
print("模型输入的尺寸:", n, c, h, w)
device = 'cpu'
exec_net = ie.load_network(network=net, device_name="CPU")


cap = cv2.VideoCapture("vinoTest.mp4")  # 视频测试
# cap = cv2.VideoCapture(0)  # 本地摄像头

while True:
    ret, ori_img = cap.read()
    if ret is not True:
        break
    res_img = cv2.resize(
        ori_img, (352, 352), interpolation=cv2.INTER_LINEAR)
    img = res_img.reshape(1, 352, 352, 3)
    img = torch.from_numpy(img.transpose(0, 3, 1, 2))
    img = img.to(device).float() / 255.0
    inf_start = time.time()
    res = exec_net.infer(inputs={input_blob: img})
    inf_end = time.time()-inf_start
    # print(res.keys())

    out = (torch.from_numpy(res['777']), torch.from_numpy(
        res['778']), torch.from_numpy(res['779']), torch.from_numpy(res['780']), torch.from_numpy(res['781']), torch.from_numpy(res['782']))

    # 特征图后处理
    output = utils.utils.handel_preds(out, cfg, device)
    # print(output)
    output_boxes = utils.utils.non_max_suppression(
        output, conf_thres=0.3, iou_thres=0.1)
    # print(output_boxes)
    # 加载label names
    LABEL_NAMES = []
    with open(cfg["names"], 'r') as f:
        for line in f.readlines():
            LABEL_NAMES.append(line.strip())

    h, w, _ = ori_img.shape
    scale_h, scale_w = h / cfg["height"], w / cfg["width"]

    # 绘制预测框
    for box in output_boxes[0]:
        box = box.tolist()

        obj_score = box[4]
        category = LABEL_NAMES[int(box[5])]

        x1, y1 = int(box[0] * scale_w), int(box[1] * scale_h)
        x2, y2 = int(box[2] * scale_w), int(box[3] * scale_h)
        # print(x1)
        cv2.rectangle(ori_img, (x1, y1), (x2, y2), (255, 255, 0), 2)
        cv2.putText(ori_img, '%.2f' % obj_score,
                    (x1, y1 - 5), 0, 0.7, (0, 255, 0), 2)
        cv2.putText(ori_img, category, (x1, y1 - 25), 0, 0.7, (0, 255, 0), 2)
        # fps
        cv2.putText(ori_img, "infer time(ms):%.3f" %
                    (inf_end*1000), (50, 50), 0, 0.7, (255, 0, 255), 2)
    cv2.imshow('result', ori_img)
    c = cv2.waitKey(1)
    if c == 27:
        break
    # cv2.destroyAllWindows()
    # cv2.imwrite("test_result.png", ori_img)
