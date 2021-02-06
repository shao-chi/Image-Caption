import sys
import os
sys.path.append("./data/yolov5/")

import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np
from PIL import Image

from data.yolov5.models.experimental import attempt_load
from data.yolov5.utils.datasets import LoadStreams, LoadImages
from data.yolov5.utils.general import check_img_size, non_max_suppression, \
                                    apply_classifier, scale_coords, xyxy2xywh, \
                                    strip_optimizer, set_logging, increment_path
from data.yolov5.utils.plots import plot_one_box
from data.yolov5.utils.torch_utils import select_device, load_classifier, time_synchronized


def get_boxes(weights, image_path, num_obj, transforms, image_size, save_img=False):
    # weights = 'yolov5x.pt'
    imgsz = 640 # img_size
    conf_thres = 0.000005
    iou_thres = 0.6 #0.45
    device = 'cpu'
    classes = None
    agnostic_nms = None
    augment = None
    exist_ok = None
    save_conf = True
    
    # Initialize
    set_logging()
    device = select_device(device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16

    # Set Dataloader
    vid_path, vid_writer = None, None
    # save_img = False
    dataset = LoadImages(image_path, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once

    all_box_list = []
    for path, img, im0s, _ in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        # t1 = time_synchronized()
        pred = model(img, augment=augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)

        if save_img:
            _, image_name = os.path.split(image_path)
            image_name = image_name.split('.')[0]
            save_dir = Path(increment_path(Path('demo') / image_name, exist_ok=True))  # increment run
            (save_dir).mkdir(parents=True, exist_ok=True)  # make dir


        # Process detections
        for i, det in enumerate(pred):  # detections per image
            p, s, im0 = Path(path), '', im0s

            if save_img:
                txt_path = str(save_dir / ('labels_' + p.stem))
                save_path = str(save_dir / ('yolo_' + p.name))

            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Write results
                box_list = []
                # for *xyxy, conf, cls_ in reversed(det):
                if len(det) < num_obj:
                    num_obj = len(det)

                i_obj = 0
                positions = []
                xyxy_list = []
                for *xyxy, conf, cls_ in det[:num_obj]:
                    # xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  
                    # normalized xywh
                    
                    try:
                        obj = im0s[int(xyxy[1]):int(xyxy[3]),
                                   int(xyxy[0]):int(xyxy[2])]
                        obj = cv2.resize(obj, (image_size, image_size),
                                         interpolation=cv2.INTER_CUBIC)
                        xyxy_list.append(xyxy)

                    except:
                        continue

                    obj = transforms(Image.fromarray(cv2.cvtColor(obj, cv2.COLOR_BGR2RGB))).unsqueeze(0)

                    if i_obj == 0:
                        img_tensor = obj

                    else:
                        img_tensor = torch.cat([img_tensor, obj])

                    xyxy_ = [xyxy[0]/im0s.shape[1], xyxy[1]/im0s.shape[0], \
                             xyxy[2]/im0s.shape[1], xyxy[3]/im0s.shape[0]]
                    zeros = [0] * 80
                    zeros[int(cls_)] = conf
                    positions.append(xyxy_ + zeros)

                    i_obj += 1
                    if i_obj == num_obj // 2:
                        break

                    # Add bbox to image
                    if save_img:
                        # normalized xywh
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn) \
                                .view(-1) \
                                .tolist()

                        line = (names[int(cls_)], conf, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%s ' + ('%g ' * (len(line) - 1)).rstrip()) \
                                        % line + '\n')

                        label = '%s %.2f' % (names[int(cls_)], conf)
                        plot_one_box(xyxy, im0,
                                     label=label,
                                     color=colors[int(cls_)],
                                     line_thickness=1)
                        cv2.imwrite(save_path, im0)

                    # box_list.append([np.array(xyxy), conf, int(cls_)])

        # all_box_list.append(box_list)

    return img_tensor, positions, xyxy_list # all_box_list
