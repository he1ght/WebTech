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

def detect_object(image_folder="data/samples", output="output", use_gpu=True, config_path="config/yolov3.cfg",
                  weights_path="weights/yolov3.weights", class_path="data/coco.names", conf_thres=0.8, nms_thres=0.4,
                  batch_size=1, n_cpu=0, img_size=416):
    cuda = torch.cuda.is_available() and use_gpu

    os.makedirs(output, exist_ok=True)

    # Set up model
    model = Darknet(config_path, img_size=img_size)
    model.load_weights(weights_path)
    preds = list()

    if cuda:
        model.cuda()

    model.eval()  # Set in evaluation mode

    dataloader = DataLoader(ImageFolder(image_folder, img_size=img_size),
                            batch_size=batch_size, shuffle=False, num_workers=n_cpu)

    classes = load_classes(class_path)  # Extracts class labels from file

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    imgs = []  # Stores image paths
    img_detections = []  # Stores detections for each image index

    for batch_i, (img_paths, input_imgs) in enumerate(dataloader):
        # Configure input
        input_imgs = Variable(input_imgs.type(Tensor))
        # Get detections
        with torch.no_grad():
            detections = model(input_imgs)
            detections = non_max_suppression(detections, 80, conf_thres, nms_thres)
        # Save image and detections
        imgs.extend(img_paths)
        img_detections.extend(detections)
        break

    for img_i, (path, detections) in enumerate(zip(imgs, img_detections)):
        print("(%d) Image: '%s'" % (img_i, path))

        img = Image.open(path)
        img_np = np.array(img)

        pad_x = max(img_np.shape[0] - img_np.shape[1], 0) * (img_size / max(img_np.shape))
        pad_y = max(img_np.shape[1] - img_np.shape[0], 0) * (img_size / max(img_np.shape))

        unpad_h = img_size - pad_y
        unpad_w = img_size - pad_x

        if detections is not None:
            unique_labels = detections[:, -1].cpu().unique()
            n_cls_preds = len(unique_labels)
            cnt = 0
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                print('\t+ Label: %s, Conf: %.5f' % (classes[int(cls_pred)], cls_conf.item()))
                img_info = {"index":cnt, "label":classes[int(cls_pred)]}
                # Rescale coordinates to original dimensions
                box_h = ((y2 - y1) / unpad_h) * img_np.shape[0]
                box_w = ((x2 - x1) / unpad_w) * img_np.shape[1]
                y1 = ((y1 - pad_y // 2) / unpad_h) * img_np.shape[0]
                x1 = ((x1 - pad_x // 2) / unpad_w) * img_np.shape[1]

                c_area = (x1.item(), y1.item(), x1.item() + box_w.item(), y1.item() + box_h.item())
                cropped_img = img.crop(c_area)
                try:
                    cropped_img.save('%s/%d.jpg' % (output, img_info['index']))
                except OSError:
                    cropped_img = cropped_img.convert("RGB")
                    cropped_img.save('%s/%d.jpg' % (output, img_info['index']))

                cnt += 1
                preds.append(img_info)
        break
    return preds


if __name__ == "__main__":
    detect_object(image_folder="data/samples")
