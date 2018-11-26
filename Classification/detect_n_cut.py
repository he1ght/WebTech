from __future__ import division

import argparse
import os

import numpy as np
import torch
from PIL import Image
from models import *
from torch.autograd import Variable
from torch.utils.data import DataLoader
from utils.datasets import *
from utils.utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--image_folder', type=str, default='data/samples', help='path to dataset')
parser.add_argument('--config_path', type=str, default='config/yolov3.cfg', help='path to model config file')
parser.add_argument('--weights_path', type=str, default='weights/yolov3.weights', help='path to weights file')
parser.add_argument('--class_path', type=str, default='data/coco.names', help='path to class label file')
parser.add_argument('--conf_thres', type=float, default=0.8, help='object confidence threshold')
parser.add_argument('--nms_thres', type=float, default=0.4, help='iou thresshold for non-maximum suppression')
parser.add_argument('--batch_size', type=int, default=1, help='size of the batches')
parser.add_argument('--n_cpu', type=int, default=0, help='number of cpu threads to use during batch generation')
parser.add_argument('--img_size', type=int, default=416, help='size of each image dimension')
parser.add_argument('--use_cuda', type=bool, default=True, help='whether to use cuda if available')
opt = parser.parse_args()
print(opt)

cuda = torch.cuda.is_available() and opt.use_cuda

os.makedirs('output', exist_ok=True)

# Set up model
model = Darknet(opt.config_path, img_size=opt.img_size)
model.load_weights(opt.weights_path)

if cuda:
    model.cuda()

model.eval()  # Set in evaluation mode

dataloader = DataLoader(ImageFolder(opt.image_folder, img_size=opt.img_size),
                        batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu)

classes = load_classes(opt.class_path)  # Extracts class labels from file

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

imgs = []  # Stores image paths
img_detections = []  # Stores detections for each image index

for batch_i, (img_paths, input_imgs) in enumerate(dataloader):
    # Configure input
    input_imgs = Variable(input_imgs.type(Tensor))
    # Get detections
    with torch.no_grad():
        detections = model(input_imgs)
        detections = non_max_suppression(detections, 80, opt.conf_thres, opt.nms_thres)
    # Save image and detections
    imgs.extend(img_paths)
    img_detections.extend(detections)

print('\nSaving images:')
for img_i, (path, detections) in enumerate(zip(imgs, img_detections)):
    print("(%d) Image: '%s'" % (img_i, path))

    img = Image.open(path)
    img_np = np.array(img)

    pad_x = max(img_np.shape[0] - img_np.shape[1], 0) * (opt.img_size / max(img_np.shape))
    pad_y = max(img_np.shape[1] - img_np.shape[0], 0) * (opt.img_size / max(img_np.shape))

    unpad_h = opt.img_size - pad_y
    unpad_w = opt.img_size - pad_x

    if detections is not None:
        unique_labels = detections[:, -1].cpu().unique()
        n_cls_preds = len(unique_labels)
        cnt = 0
        for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
            print('\t+ Label: %s, Conf: %.5f' % (classes[int(cls_pred)], cls_conf.item()))

            # Rescale coordinates to original dimensions
            box_h = ((y2 - y1) / unpad_h) * img_np.shape[0]
            box_w = ((x2 - x1) / unpad_w) * img_np.shape[1]
            y1 = ((y1 - pad_y // 2) / unpad_h) * img_np.shape[0]
            x1 = ((x1 - pad_x // 2) / unpad_w) * img_np.shape[1]

            c_area = (x1.item(), y1.item(), x1.item() + box_w.item(), y1.item() + box_h.item())
            cropped_img = img.crop(c_area)
            cropped_img.save('output/%d.%d_%s.png' % (img_i, cnt, classes[int(cls_pred)]))
            cnt += 1
