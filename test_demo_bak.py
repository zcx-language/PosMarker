import os
import numpy as np
import cv2

import torch
from models.SSD_vgg_bn import build_net
from layers.functions import Detect, PriorBox
from data import MARK_512, BaseTransform
from data.mark import AnnotationTransform, VOCDetection, detection_collate
from utils.timer import Timer
from utils.nms_wrapper import nms
import pickle
import argparse

parser = argparse.ArgumentParser(
    description='test mAP with checkpoints')
parser.add_argument('-e', '--epoch', default='40',
                    help='num of epochs of checkpoint file')
args = parser.parse_args()


def test_net(save_folder, net, detector, cuda, testset, transform, max_per_image=300, thresh=0.005):
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    # dump predictions and assoc. ground truth to text file for now
    num_images = len(testset)
    num_classes = 2

    # shape [2, 425], 每个类别每张图的bbox
    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(num_classes)]

    _t = {'im_detect': Timer(), 'misc': Timer()}
    det_file = os.path.join(save_folder, 'detections.pkl')

    all_time = 0
    for i in range(num_images):
        img = testset.pull_image(i)
        with torch.no_grad():
            # 对x进行缩放，适配网络输入尺寸
            x = transform(img).unsqueeze(0)
        if cuda:
            x = x.to(torch.device("cuda"))

        _t['im_detect'].tic()
        # out shape: [[1, 32756, 4], [32756, 2]]
        out = net(x=x, test=True)  # forward pass

        # [1, 32756, 4], [1, 32756, 2]
        boxes, scores = detector.forward(out, priors)
        detect_time = _t['im_detect'].toc()
        all_time += detect_time
        boxes = boxes[0]
        scores = scores[0]

        boxes = boxes.cpu().numpy()
        scores = scores.cpu().numpy()
        # scale each detection back up to the image
        scale = torch.Tensor([img.shape[1], img.shape[0],
                              img.shape[1], img.shape[0]]).cpu().numpy()
        boxes *= scale

        _t['misc'].tic()

        #
        # print('box shape: ', boxes.shape)

        for j in range(1, num_classes):
            inds = np.where(scores[:, j] > thresh)[0]
            if len(inds) == 0:
                all_boxes[j][i] = np.empty([0, 5], dtype=np.float32)
                continue
            c_bboxes = boxes[inds]
            c_scores = scores[inds, j]
            c_dets = np.hstack((c_bboxes, c_scores[:, np.newaxis])).astype(
                np.float32, copy=False)
            keep = nms(c_dets, 0.45, force_cpu=False)
            keep = keep[:50]
            c_dets = c_dets[keep, :]
            all_boxes[j][i] = c_dets
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


import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

cfg = MARK_512

mark_root = '/data_ssd2/hzh/paperworks/dataset/original/photos'

# args.epoch = '16'
net = build_net((512, 512), 2)
# net.load_state_dict(torch.load('weights/SSD_vgg_bn_MARK_512/20210419_fix_gen/SSD_vgg_bn_MARK_epoches_' + args.epoch + '.pth'))
net.load_state_dict(torch.load('weights/SSD_vgg_bn_MARK_512/20210421_fix_gen/SSD_vgg_bn_MARK_epoches_' + args.epoch + '.pth'))
net.to('cuda')
net.eval()
top_k = 300
detector = Detect(2, 0, cfg)

priorbox = PriorBox(cfg)
with torch.no_grad():
    priors = priorbox.forward()

testset = VOCDetection(mark_root, [('2007', 'test')], None, AnnotationTransform())
APs, mAP = test_net('weights/SSD_vgg_bn_MARK_512/20210421_fix_gen/ss_predict', net, detector, True, testset,
                        BaseTransform((net.size, net.size), (104, 117, 123)),
                        top_k, thresh=0.05)
APs = [str(num) for num in APs]
print(mAP)

