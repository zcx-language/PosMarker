import os
import pickle
import cv2
import torch
import numpy as np

from utils.timer import Timer
from utils.nms_wrapper import nms


def img_to_crops(img, crop_size, lap_size):
    crops = []
    offsets = []
    height, width = img.shape[:2]
    for y in range(0, height, crop_size - lap_size):
        for x in range(0, width, crop_size - lap_size):
            # 最后一个为了避免缩放， 反向切割
            if x + crop_size > width:
                x = width - crop_size
            if y + crop_size > height:
                y = height - crop_size
            crops.append(
                torch.from_numpy(img[y: y + crop_size, x: x + crop_size]).permute(2, 0, 1).unsqueeze(0))
            offsets.append((x, y))
    return offsets, crops


def test_net(save_folder, net, detector, cuda, testset, transform, max_per_image=300, thresh=0.005):

    # 创建目录
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    # 测试集信息
    num_images = len(testset)
    num_classes = 2

    # shape [2, 425, 15], 最外层为两个list， 每个list长度为425， 内层list每个元素为长度为15的list， 最内层元素为[50, 5]的ndarray
    all_boxes = [[[[] for _ in range(15)] for _ in range(num_images)]
                 for _ in range(num_classes)]

    _t = {'im_detect': Timer(), 'misc': Timer()}
    det_file = os.path.join(save_folder, 'detections.pkl')

    all_time = 0

    # 对每张图
    for i in range(num_images):
        # 转成NCWH格式，送到gpu
        img = testset.pull_image(i)
        # resize
        img = cv2.resize(np.array(img), (1920, 1080), interpolation=cv2.INTER_LINEAR).astype(np.float32)
        offsets, crops = img_to_crops(img, 512, 100)

        # 对每张crop
        for idx, crop in enumerate(crops):

            if cuda:
                crop = crop.to(torch.device("cuda"))

            _t['im_detect'].tic()
            # 单张图的预测结果，out shape: [[1, 32756, 4], [32756, 2]]
            out = net(x=crop, test=True)  # forward pass

            # decode预测结果，[1, 32756, 4], [1, 32756, 2]
            boxes, scores = detector.forward(out, priors)
            detect_time = _t['im_detect'].toc()
            all_time += detect_time

            # squeeze 检测结果
            boxes = boxes[0]
            scores = scores[0]

            boxes = boxes.cpu().numpy()
            scores = scores.cpu().numpy()
            # scale each detection back up to the image
            scale = torch.Tensor([512, 512, 512, 512]).cpu().numpy()
            boxes *= scale

            _t['misc'].tic()

            for j in range(1, num_classes):
                for k in range(15):
                    inds = np.where(scores[:, j] > thresh)[0]
                    if len(inds) == 0:
                        all_boxes[j][i][k] = np.empty([0, 5], dtype=np.float32)
                        continue
                    c_bboxes = boxes[inds]
                    c_scores = scores[inds, j]
                    c_dets = np.hstack((c_bboxes, c_scores[:, np.newaxis])).astype(
                        np.float32, copy=False)
                    keep = nms(c_dets, 0.45, force_cpu=False)
                    keep = keep[:50]
                    c_dets = c_dets[keep, :]
                    all_boxes[j][i][k] = c_dets
                    all_boxes[j][i][k] = all_boxes[j][i][k] + np.array([offsets[idx][0], offsets[idx][1],
                                                                        offsets[idx][0], offsets[idx][1]],
                                                                        dtype=np.float)
                if max_per_image > 0:
                    image_scores = np.hstack([all_boxes[j][i][k][:, -1] for j in range(1, num_classes)])
                    if len(image_scores) > max_per_image:
                        image_thresh = np.sort(image_scores)[-max_per_image]
                        for j in range(1, num_classes):
                            keep = np.where(all_boxes[j][i][k][:, -1] >= image_thresh)[0]
                            all_boxes[j][i][k] = all_boxes[j][i][k][keep, :]

            # 将最内层的list合并为一个ndarray
            for i in range(all_boxes.shape[0]):
                for j in range(all_boxes.shape[1]):
                    all_boxes[i][j] = np.vstack(all_boxes[i][j])

            keep = nms(all_boxes, 0.45, force_cpu=False)
            keep = keep[:50]
            all_boxes = all_boxes[keep, :]

            if max_per_image > 0:
                image_scores = np.hstack([all_boxes[j][i][:, -1] for j in range(1, num_classes)])
                if len(image_scores) > max_per_image:
                    image_thresh = np.sort(image_scores)[-max_per_image]
                    for j in range(1, num_classes):
                        keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                        all_boxes[j][i] = all_boxes[j][i][keep, :]

            nms_time = _t['misc'].toc()
            all_time += nms_time

            if i % 20 == 0:
                print('im_detect: {:d}/{:d} {:.3f}s {:.3f}s'
                      .format(i + 1, num_images, detect_time, nms_time))
                _t['im_detect'].clear()
                _t['misc'].clear()

    avg_time = all_time / num_images
    if avg_time:
        FPS = 1.0 / avg_time
        print('average detection time: {:.3f}s FPS: {:.3f}'.format(avg_time, FPS))

    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

    print('Evaluating detections')

    APs, mAP = testset.evaluate_detections(all_boxes, save_folder)
    return APs, mAP


def img_to_crops(img, crop_size, lap_size):
    crops = []
    offsets = []
    height, width = img.shape[:2]
    for y in range(0, height, crop_size - lap_size):
        for x in range(0, width, crop_size - lap_size):
            # 最后一个为了避免缩放， 反向切割
            if x + crop_size > width:
                x = width - crop_size
            if y + crop_size > height:
                y = height - crop_size
            crops.append(torch.from_numpy(img[y: y + crop_size, x: x + crop_size]).permute(2, 0, 1).unsqueeze(0))
            offsets.append((x, y))
    return offsets, crops


# for crop_box in crop_boxes:
#     xmin, ymin, xmax, ymax = crop_box
#     b = random.randint(0, 255)
#     g = random.randint(0, 255)
#     r = random.randint(0, 255)
#     img = cv2.rectangle(img, (xmin, ymin), (xmax-1, ymax-1), (b, g, r), 2)
#
# cv2.imshow("", img)
# cv2.waitKey(0)

img = cv2.imread("/data_ssd2/hzh/paperworks/dataset/MarkedDATA_AUG_REC/VOC2007/JPEGImages/train000006.jpg")
offsets, crops = img_to_crops(img, 512, 100)
print(offsets, crops.shape)