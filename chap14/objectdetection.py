
# # IoU计算。
# def cal_iou_xyxy(box1,box2):
#     x1min, y1min, x1max, y1max = box1[0], box1[1], box1[2], box1[3]
#     x2min, y2min, x2max, y2max = box2[0], box2[1], box2[2], box2[3]
#     #计算两个框的面积
#     s1 = (y1max - y1min + 1.) * (x1max - x1min + 1.)
#     s2 = (y2max - y2min + 1.) * (x2max - x2min + 1.)
#
#     #计算相交部分的坐标
#     xmin = max(x1min,x2min)
#     ymin = max(y1min,y2min)
#     xmax = min(x1max,x2max)
#     ymax = min(y1max,y2max)
#
#     inter_h = max(ymax - ymin + 1, 0)
#     inter_w = max(xmax - xmin + 1, 0)
#
#     intersection = inter_h * inter_w
#     union = s1 + s2 - intersection
#
#     #计算iou
#     iou = intersection / union
#     return iou
#
# box1 = [100,100,200,200]
# box2 = [120,120,220,220]
# iou = cal_iou_xyxy(box1,box2)
# print(iou)


# NMS计算
import cv2
import numpy as np
import matplotlib.pyplot as plt


# iou = np.array([0.57184716, 0.76505679])
# order = np.array([2, 1, 4, 5, 6, 8])
# inds = np.where(iou <= 0.8)[0]  # 将重叠度大于给定阈值的边框剔除掉，仅保留剩下的边框，返回相应的下标 [1,2]
# print('inds:', inds)  # 得到分数小于阈值的框索引 [0 1]
# order = order[inds + 1]  # 从剩余的候选框中继续筛选[1 2]
# print('order:', order)  # [1 4 ]

# # NMS计算实例。
# import numpy as np
# boxes = np.array([[100, 100, 210, 210, 0.72],
#                   [250, 250, 420, 420, 0.8],
#                   [220, 220, 320, 330, 0.92],
#                   [100, 100, 210, 210, 0.72],
#                   [230, 240, 325, 330, 0.81],
#                   [220, 230, 315, 340, 0.9]])
#
#
# def py_cpu_nms(dets, thresh):
#
#     x1 = dets[:, 0]
#     y1 = dets[:, 1]
#     x2 = dets[:, 2]
#     y2 = dets[:, 3]
#
#     areas = (y2 - y1 + 1) * (x2 - x1 + 1)
#     scores = dets[:, 4]
#     keep = []
#     index = scores.argsort()[::-1]
#
#     while index.size > 0:
#         print("sorted index of boxes according to scores", index)
#
#         i = index[0]
#         keep.append(i)
#
#         x11 = np.maximum(x1[i], x1[index[1:]])
#         y11 = np.maximum(y1[i], y1[index[1:]])
#         x22 = np.minimum(x2[i], x2[index[1:]])
#         y22 = np.minimum(y2[i], y2[index[1:]])
#
#         print("x1 values by original order:", x1)
#         print("x1 value by scores:", x1[index[:]])
#         print("x11 value means  replacing the less value compared"\
#               " with the value by the largest score :" , x11)
#
#         w = np.maximum(0, x22 - x11 + 1)
#         h = np.maximum(0, y22 - y11 + 1)
#         overlaps = w * h
#
#         ious = overlaps / (areas[i] + areas[index[1:]] - overlaps)
#
#         idx = np.where(ious <= thresh)[0]
#
#         index = index[idx + 1]
#
#     return keep
#
#
# import matplotlib.pyplot as plt
#
#
# def plot_bbox(dets, c='k', title_name="title"):
#     x1 = dets[:, 0]
#     y1 = dets[:, 1]
#     x2 = dets[:, 2]
#     y2 = dets[:, 3]
#
#     plt.plot([x1, x2], [y1, y1], c)
#     plt.plot([x1, x1], [y1, y2], c)
#     plt.plot([x1, x2], [y2, y2], c)
#     plt.plot([x2, x2], [y1, y2], c)
#     plt.title(title_name)
#
# if __name__ == '__main__':
#     plot_bbox(boxes, 'k', title_name="before NMS")
#     plt.show()
#
#     keep = py_cpu_nms(boxes, thresh=0.7)
#
#     plot_bbox(boxes[keep], 'r', title_name="after NMS")
#     plt.show()


# #fasterrcnn_resnet50网络目标检测。
# ## 导入相关模块
# import numpy as np
# import torchvision
# import torch
# import torchvision.transforms as transforms
# from PIL import Image, ImageDraw, ImageFont
# import matplotlib.pyplot as plt
#
# model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
# model.eval()
#
#
# ## 准备需要检测的图像
# image = Image.open("./MSCOCO2017/val2017/000000005477.jpg")
# transform_d = transforms.Compose([transforms.ToTensor()])
# image_t = transform_d(image)
# pred = model([image_t])
# print(pred)
#
#
# COCO_INSTANCE_CATEGORY_NAMES = [
#     '__BACKGROUND__', 'person', 'bicycle', 'car', 'motorcycle',
#     'airplane', 'bus', 'train', 'trunk', 'boat', 'traffic light',
#     'fire hydrant', 'N/A', 'stop sign', 'parking meter', 'bench',
#     'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant',
#     'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A',
#     'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
#     'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
#     'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
#     'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
#     'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
#     'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A',
#     'dining table', 'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop',
#     'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
#     'toaster', 'toaster', 'sink', 'refrigerator', 'N/A', 'book', 'clock',
#     'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
# ]
#
# ## 检测出目标的类别和得分
# pred_class = [COCO_INSTANCE_CATEGORY_NAMES[ii] for ii in list(pred[0]['labels'].numpy())]
# pred_score = list(pred[0]['scores'].detach().numpy())
#
# ## 检测出目标的边界框
# pred_boxes = [[ii[0], ii[1], ii[2], ii[3]] for ii in list(pred[0]['boxes'].detach().numpy())]
#
# ## 只保留识别的概率大约 0.5 的结果。
# pred_index = [pred_score.index(x) for x in pred_score if x > 0.5]
#
# ## 设置图像显示的字体
# fontsize = np.int16(image.size[1] / 20)
#
# ## 可视化对象
# draw = ImageDraw.Draw(image)
# for index in pred_index:
#     box = pred_boxes[index]
#     draw.rectangle(box, outline="blue")
#     texts = pred_class[index]+":"+str(np.round(pred_score[index], 2))
#     draw.text((box[0], box[1]), texts, fill="blue")
#
# ## 显示图像
# plt.imshow(image)
# plt.show()


# 随机选择文件夹中3张图片使用fasterrcnn_resnet50网络目标检测。
import numpy as np
import torchvision
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import os


COCO_INSTANCE_CATEGORY_NAMES = [
    '__BACKGROUND__', 'person', 'bicycle', 'car', 'motorcycle',
    'airplane', 'bus', 'train', 'trunk', 'boat', 'traffic light',
    'fire hydrant', 'N/A', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant',
    'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A',
    'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
    'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A',
    'dining table', 'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop',
    'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
    'toaster', 'toaster', 'sink', 'refrigerator', 'N/A', 'book', 'clock',
    'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# 加载预训练的模型
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# 准备需要检测的图像文件夹
image_folder_path = "./MSCOCO2017/val2017/"
number = 3
transform_d = transforms.Compose([transforms.ToTensor()])

# 随机选择10张图片
dirs = os.listdir(image_folder_path)
idx = np.random.randint(0, len(dirs), number)

for i in idx:
    image_path = os.path.join(image_folder_path, dirs[i])
    image = Image.open(image_path)

    image_t = transform_d(image)
    pred = model([image_t])
    print(pred)

    # 检测出目标的类别和得分
    pred_class = [COCO_INSTANCE_CATEGORY_NAMES[ii] for ii in list(pred[0]['labels'].numpy())]
    pred_score = list(pred[0]['scores'].detach().numpy())

    # 检测出目标的边界框
    pred_boxes = [[ii[0], ii[1], ii[2], ii[3]] for ii in list(pred[0]['boxes'].detach().numpy())]

    # 只保留识别的概率大约 0.5 的结果。
    pred_index = [pred_score.index(x) for x in pred_score if x > 0.5]

    # 设置图像显示的字体
    fontsize = np.int16(image.size[1] / 4)

    # 可视化对象
    draw = ImageDraw.Draw(image)
    for index in pred_index:
        box = pred_boxes[index]
        draw.rectangle(box, outline="blue")
        texts = pred_class[index]+":"+str(np.round(pred_score[index], 2))
        draw.text((box[0], box[1]), texts, fill="blue")

    # 显示图像
    plt.imshow(image)
    plt.show()

