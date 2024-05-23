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
parser.add_argument('-e', '--epoch', default='30',
                    help='num of epochs of checkpoint file')
args = parser.parse_args()


labelmap = ('__background__', 'mark')





class ObjectDetector:
    def __init__(self, net, detection, transform, num_classes=21, cuda=False, max_per_image=300, thresh=0.5):
        self.net = net
        self.detection = detection
        self.transform = transform
        self.max_per_image = 300
        self.num_classes = num_classes
        self.max_per_image = max_per_image
        self.cuda = cuda
        self.thresh = thresh

    def predict(self, img):
        scale = torch.Tensor([img.shape[1], img.shape[0],
                              img.shape[1], img.shape[0]]).cpu().numpy()
        _t = {'im_detect': Timer(), 'misc': Timer()}
        assert img.shape[2] == 3
        x = Variable(self.transform(img).unsqueeze(0), volatile=True)
        if self.cuda:
            x = x.cuda()
        _t['im_detect'].tic()
        f, out = net(x, test=True)  # forward pass
        pyramid_fea.append(f)
        boxes, scores = self.detection.forward(out, priors)
        detect_time = _t['im_detect'].toc()
        boxes = boxes[0]
        scores = scores[0]

        boxes = boxes.cpu().numpy()
        scores = scores.cpu().numpy()
        # scale each detection back up to the image
        boxes *= scale
        _t['misc'].tic()
        all_boxes = [[] for _ in range(num_classes)]

        for j in range(1, num_classes):
            inds = np.where(scores[:, j] > self.thresh)[0]
            if len(inds) == 0:
                all_boxes[j] = np.zeros([0, 5], dtype=np.float32)
                continue
            c_bboxes = boxes[inds]
            c_scores = scores[inds, j]
            print(scores[:, j])
            c_dets = np.hstack((c_bboxes, c_scores[:, np.newaxis])).astype(
                np.float32, copy=False)
            # keep = nms(c_bboxes,c_scores)

            keep = py_cpu_nms(c_dets, 0.45)
            keep = keep[:50]
            c_dets = c_dets[keep, :]
            all_boxes[j] = c_dets
        if self.max_per_image > 0:
            image_scores = np.hstack([all_boxes[j][:, -1] for j in range(1, num_classes)])
            if len(image_scores) > self.max_per_image:
                image_thresh = np.sort(image_scores)[-self.max_per_image]
                for j in range(1, num_classes):
                    keep = np.where(all_boxes[j][:, -1] >= image_thresh)[0]
                    all_boxes[j] = all_boxes[j][keep, :]

        nms_time = _t['misc'].toc()
        print('net time: ', detect_time)
        print('post time: ', nms_time)
        return all_boxes


def test_net(save_folder, net, detector, cuda, testset, transform, max_per_image=300, thresh=0.005):
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    # dump predictions and assoc. ground truth to text file for now
    num_classes = 2
    num_images = 1

    # shape [2, 425], 每个类别每张图的bbox
    all_boxes = [[] for _ in range(num_classes)]

    _t = {'im_detect': Timer(), 'misc': Timer()}
    det_file = os.path.join(save_folder, 'detections.pkl')

    all_time = 0
    if True: #only one image input
        i = 0
        img = cv2.imread('4.jpg')
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
        print('box shape: ', boxes.shape)

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

        detect_bboxes = all_boxes

        nms_time = _t['misc'].toc()

        if i % 20 == 0:
            print('im_detect: {:d}/{:d} {:.3f}s {:.3f}s'
                  .format(i + 1, num_images, detect_time, nms_time))
            _t['im_detect'].clear()
            _t['misc'].clear()

            for class_id, class_collection in enumerate(detect_bboxes):
                if len(class_collection) > 0:
                    for i in range(class_collection.shape[0]):
                        if class_collection[i, -1] > 0.5:
                            pt = class_collection[i]
                            cv2.rectangle(image, (int(pt[0]), int(pt[1])), (int(pt[2]),
                                                                            int(pt[3])), COLORS[class_id % 3], 2)
                            cv2.putText(image, labelmap[class_id], (int(pt[0]), int(pt[1])), FONT,
                                        0.5, (255, 255, 255), 2)
            cv2.imwrite('4_detection' + '.jpg', image)





os.environ['CUDA_VISIBLE_DEVICES'] = '1'
cfg = MARK_512
mark_root = '/data_ssd2/hzh/paperworks/dataset/augment/ScreenshotsDATA'

args.epoch = '5'
net = build_net((512, 512), 2)
net.load_state_dict(torch.load('weights/SSD_vgg_bn_MARK_512/20210419_fix_gen/SSD_vgg_bn_MARK_epoches_' + args.epoch + '.pth'))
net.to('cuda')
net.eval()
top_k = 300
detector = Detect(2, 0, cfg)

priorbox = PriorBox(cfg)
with torch.no_grad():
    priors = priorbox.forward()

# testset = VOCDetection(mark_root, [('2007', 'test')], None, AnnotationTransform())
# test_net('weights/SSD_vgg_bn_MARK_512/20210419_fix_gen/ss_predict', net, detector, True, testset,
#                         BaseTransform((net.size, net.size), (104, 117, 123)),
#                         top_k, thresh=0.01)

transform = BaseTransform((net.size, net.size), (104, 117, 123))
object_detector = ObjectDetector(net, detector, transform)

img = cv2.imread('4.jpg')
detect_bboxes = object_detector.predict(img)

for class_id, class_collection in enumerate(detect_bboxes):
    if len(class_collection) > 0:
        for i in range(class_collection.shape[0]):
            if class_collection[i, -1] > 0.5:  # thresh = 0.6
                pt = class_collection[i]
                cv2.rectangle(image, (int(pt[0]), int(pt[1])), (int(pt[2]),
                                                                int(pt[3])), COLORS[class_id % 3], 2)
                cv2.putText(image, labelmap[class_id] + '%.2f' % (class_collection[i, -1]), (int(pt[0]), int(pt[1])),
                            FONT,
                            0.5, (30, 30, 30), 2)
cv2.imwrite('4_detection' + '.jpg', image)


