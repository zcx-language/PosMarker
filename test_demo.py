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
            img_temp = img.copy()[y: y + crop_size, x: x + crop_size]
            img_temp = cv2.resize(np.array(img_temp), (512, 512), interpolation=cv2.INTER_LINEAR).astype(np.float32)
            crops.append(torch.from_numpy(img_temp).permute(2, 0, 1).unsqueeze(0))
            offsets.append((x, y))
    return offsets, crops


# def test_net_yolt2(save_folder, net, detector, cuda, testset, transform, max_per_image=300, thresh=0.005):
#
#     # 创建目录
#     if not os.path.exists(save_folder):
#         os.mkdir(save_folder)
#
#     # 测试集信息
#     num_images = len(testset)
#     num_classes = 2
#
#     # shape [2, 425, 15], 最外层为两个list， 每个list长度为425， 内层list每个元素为长度为15的list， 最内层元素为[50, 5]的ndarray
#     all_boxes = [[[[] for _ in range(15)] for _ in range(num_images)]
#                  for _ in range(num_classes)]
#
#     _t = {'im_detect': Timer(), 'misc': Timer()}
#     det_file = os.path.join(save_folder, 'detections.pkl')
#
#     all_time = 0
#
#     #######################################
#     #               对每张图i
#     #######################################
#     for i in range(num_images):
#         # 转成NCWH格式，送到gpu
#         img = testset.pull_image(i)
#         # resize
#         img = cv2.resize(np.array(img), (1920, 1080), interpolation=cv2.INTER_LINEAR).astype(np.float32)
#         offsets, crops = img_to_crops(img, 512, 100)
#
#         #######################################
#         #               对每张crop idx
#         #######################################
#         for idx, crop in enumerate(crops):
#
#             if cuda:
#                 crop = crop.to(torch.device("cuda"))
#
#             _t['im_detect'].tic()
#             # 单张图的预测结果，out shape: [[1, 32756, 4], [32756, 2]]
#             out = net(x=crop, test=True)  # forward pass
#
#             # decode预测结果，[1, 32756, 4], [1, 32756, 2]
#             boxes, scores = detector.forward(out, priors)
#             detect_time = _t['im_detect'].toc()
#             all_time += detect_time
#
#             # squeeze 检测结果
#             boxes = boxes[0]
#             scores = scores[0]
#
#             boxes = boxes.cpu().numpy()
#             scores = scores.cpu().numpy()
#             # scale each detection back up to the image
#             scale = torch.Tensor([512, 512, 512, 512]).cpu().numpy()
#             boxes *= scale
#
#             _t['misc'].tic()
#
#             #######################################
#             #               对每个类别j
#             #######################################
#             for j in range(1, num_classes):
#                 inds = np.where(scores[:, j] > thresh)[0]
#                 if len(inds) == 0:
#                     all_boxes[j][i][idx] = np.empty([0, 5], dtype=np.float32)
#                     continue
#                 c_bboxes = boxes[inds] + np.array([offsets[idx][0], offsets[idx][1],
#                                                                     offsets[idx][0], offsets[idx][1]],
#                                                                     dtype=np.float)
#                 c_scores = scores[inds, j]
#                 c_dets = np.hstack((c_bboxes, c_scores[:, np.newaxis])).astype(
#                     np.float32, copy=False)
#                 keep = nms(c_dets, 0.45, force_cpu=False)
#                 keep = keep[:50]
#                 c_dets = c_dets[keep, :]
#                 all_boxes[j][i][idx] = c_dets
#                 all_boxes[j][i][idx] = all_boxes[j][i][idx]
#             if max_per_image > 0:
#                 image_scores = np.hstack([all_boxes[j][i][idx][:, -1] for j in range(1, num_classes)])
#                 if len(image_scores) > max_per_image:
#                     image_thresh = np.sort(image_scores)[-max_per_image]
#                     for j in range(1, num_classes):
#                         keep = np.where(all_boxes[j][i][idx][:, -1] >= image_thresh)[0]
#                         all_boxes[j][i][idx] = all_boxes[j][i][idx][keep, :]
#
#         # 将最内层的list合并为一个ndarray
#         all_boxes[j][i] = np.vstack(all_boxes[j][i])
#
#         keep = nms(all_boxes, 0.45, force_cpu=False)
#         keep = keep[:50]
#         all_boxes = all_boxes[keep, :]
#
#         if max_per_image > 0:
#             image_scores = np.hstack([all_boxes[j][i][:, -1] for j in range(1, num_classes)])
#             if len(image_scores) > max_per_image:
#                 image_thresh = np.sort(image_scores)[-max_per_image]
#                 for j in range(1, num_classes):
#                     keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
#                     all_boxes[j][i] = all_boxes[j][i][keep, :]
#
#         nms_time = _t['misc'].toc()
#         all_time += nms_time
#
#         if i % 20 == 0:
#             print('im_detect: {:d}/{:d} {:.3f}s {:.3f}s'
#                   .format(i + 1, num_images, detect_time, nms_time))
#             _t['im_detect'].clear()
#             _t['misc'].clear()
#
#     avg_time = all_time / num_images
#     if avg_time:
#         FPS = 1.0 / avg_time
#         print('average detection time: {:.3f}s FPS: {:.3f}'.format(avg_time, FPS))
#
#     with open(det_file, 'wb') as f:
#         pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)
#
#     print('Evaluating detections')
#
#     APs, mAP = testset.evaluate_detections(all_boxes, save_folder)
#     return APs, mAP


def test_net_yolt3(save_folder, net, detector, cuda, testset, transform, max_per_image=300, thresh=0.005):

    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    # dump predictions and assoc. ground truth to text file for now
    num_images = len(testset)
    num_classes = 2

    # shape [2, 425], 每个类别每张图的bbox
    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(num_classes)]

    _t = {'infer_time': Timer(), 'postproc_time': Timer(), 'nms_time': Timer()}
    det_file = os.path.join(save_folder, 'detections.pkl')

    all_time = 0
    for i in range(num_images):
        img = testset.pull_image(i)
        img = cv2.resize(np.array(img), (1920, 1080), interpolation=cv2.INTER_LINEAR).astype(np.float32)
        # cv2.imshow('original', img.astype(np.uint8))
        # cv2.waitKey(0)

        # 转为crops
        crop_size = 512
        offsets, crops = img_to_crops(img.copy(), crop_size, 100)
        #print(len(crops))
        # with torch.no_grad():
        #     # 对x进行缩放，适配网络输入尺寸
        #     origin_img = transform(img).unsqueeze(0)
        if cuda:
            # x = crops.to(torch.device("cuda"))
            crops = [crop.to(torch.device("cuda")) for crop in crops]
            # origin_img = origin_img.to(torch.device("cuda"))
            # crops.append(origin_img)



        _t['infer_time'].tic()
        # out shape: [[1, 32756, 4], [32756, 2]]
        # out = net(x=x, test=True)  # forward pass
        boxes_list = []
        scores_list = []
        with torch.no_grad():
            for crop in crops:
                out = net(x=crop, test=True)
                # [1, 32756, 4], [1, 32756, 2]
                boxes, scores = detector.forward(out, priors)

                boxes_list.append(boxes)
                scores_list.append(scores)

        detect_time = _t['infer_time'].toc()
        all_time += detect_time
        # boxes = boxes[0]
        # scores = scores[0]

        _t['postproc_time'].tic()

        for idx in range(len(boxes_list)):

            boxes = boxes_list[idx][0].cpu().numpy()
            scores = scores_list[idx][0].cpu().numpy()

            # scale each detection back up to the image
            scale = torch.Tensor([crop_size, crop_size, crop_size, crop_size]).cpu().numpy()
            boxes *= scale
            boxes += np.array([offsets[idx][0], offsets[idx][1], offsets[idx][0], offsets[idx][1]], dtype=np.float)
            # boxes[:, 0::2] *= (1920 / 1000.)
            # boxes[:, 1::2] *= (1080 / 1000.)


            # nms
            _t['nms_time'].tic()
            for j in range(1, num_classes): #迭代只会运行一次
                inds = np.where(scores[:, j] > thresh)[0]

                c_bboxes = boxes[inds]
                c_scores = scores[inds, j]
                c_background_scores = (scores[inds, 0])[:, np.newaxis]

                #
                c_dets = np.hstack((c_bboxes, c_scores[:, np.newaxis])).astype(
                    np.float32, copy=False)

                keep = nms(c_dets, 0.45, force_cpu=False)
                keep = keep[:50]
                c_dets = c_dets[keep, :]
                c_background_scores = c_background_scores[keep, :]

                #TODO: 把c_dets = np.hstack((c_bboxes, c_scores[:, np.newaxis])) 展开
                boxes = c_dets[:, :4]
                scores = np.hstack((c_background_scores, (c_dets[:, 4])[:, np.newaxis]))


            boxes_list[idx] = boxes
            scores_list[idx] = scores


        # 所有bbox合到一起
        boxes = np.vstack(boxes_list)
        scores = np.vstack(scores_list)

        # print("BOX:", boxes)
        # print("scores:", scores)
        # print(max(scores[:, 1]))

        # print('second nms box shape:', boxes.shape)


        # nms
        _t['nms_time'].tic()
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

            if True:

                img_ = img.copy().astype(np.uint8)

                detect_bboxes = all_boxes[j][i]
                for class_id, class_collection in enumerate(detect_bboxes):
                    if len(class_collection) > 0:
                        if class_collection[-1] > 0.2:
                            pt = class_collection

                            img_ = cv2.rectangle(img_, (int(pt[0]), int(pt[1])), (int(pt[2]),
                                                                            int(pt[3])), (0,0,255), 1)
                                # cv2.putText(image, 'mark', (int(pt[0]), int(pt[1])), FONT,
                                #             0.3, (255, 255, 255), 1)
                cv2.imshow('result', img_.astype(np.uint8))
                cv2.waitKey(0)


        post_time = _t['postproc_time'].toc()
        nms_time = _t['nms_time'].toc()
        all_time += post_time

        if i % 20 == 0:
            print('im_detect: {:d}/{:d}, infer time: {:.3f}s, postproc time: {:.3f}s, nms time: {:.3f}s'
                  .format(i + 1, num_images, detect_time, post_time, nms_time))
            _t['infer_time'].clear()
            _t['postproc_time'].clear()
            _t['nms_time'].clear()

    avg_time = all_time / num_images
    if avg_time:
        FPS = 1.0 / avg_time
        print('average detection time: {:.3f}s FPS: {:.3f}'.format(avg_time, FPS))

    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

    print('Evaluating detections')

    APs, mAP = testset.evaluate_detections(all_boxes, save_folder, iou_all=True)
    return APs, mAP



def test_net_yolt(save_folder, net, detector, cuda, testset, transform, max_per_image=300, thresh=0.005):

    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    # dump predictions and assoc. ground truth to text file for now
    num_images = len(testset)
    num_classes = 2

    # shape [2, 425], 每个类别每张图的bbox
    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(num_classes)]

    _t = {'infer_time': Timer(), 'postproc_time': Timer(), 'nms_time': Timer()}
    det_file = os.path.join(save_folder, 'detections.pkl')

    all_time = 0
    for i in range(num_images):
        img = testset.pull_image(i)
        img = cv2.resize(np.array(img), (1920, 1080), interpolation=cv2.INTER_LINEAR).astype(np.float32)

        # 转为crops
        offsets, crops = img_to_crops(img, 512, 100)
        if cuda:
            # x = crops.to(torch.device("cuda"))
            crops = [crop.to(torch.device("cuda")) for crop in crops]

        _t['infer_time'].tic()
        # out shape: [[1, 32756, 4], [32756, 2]]
        # out = net(x=x, test=True)  # forward pass
        boxes_list = []
        scores_list = []
        with torch.no_grad():
            for crop in crops:
                out = net(x=crop, test=True)
                # [1, 32756, 4], [1, 32756, 2]
                boxes, scores = detector.forward(out, priors)
                boxes_list.append(boxes)
                scores_list.append(scores)

        detect_time = _t['infer_time'].toc()
        all_time += detect_time
        # boxes = boxes[0]
        # scores = scores[0]

        _t['postproc_time'].tic()

        for idx in range(len(boxes_list)):

            boxes = boxes_list[idx][0].cpu().numpy()
            scores = scores_list[idx][0].cpu().numpy()

            # scale each detection back up to the image
            scale = torch.Tensor([512, 512, 512, 512]).cpu().numpy()
            boxes *= scale
            boxes += np.array([offsets[idx][0], offsets[idx][1], offsets[idx][0], offsets[idx][1]], dtype=np.float)

            boxes_list[idx] = boxes
            scores_list[idx] = scores

        # 所有bbox合到一起
        boxes = np.vstack(boxes_list)
        scores = np.vstack(scores_list)

        print('box shape: ', boxes.shape)


        # nms
        _t['nms_time'].tic()
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

        post_time = _t['postproc_time'].toc()
        nms_time = _t['nms_time'].toc()
        all_time += post_time

        if i % 20 == 0:
            print('im_detect: {:d}/{:d}, infer time: {:.3f}s, postproc time: {:.3f}s, nms time: {:.3f}s'
                  .format(i + 1, num_images, detect_time, post_time, nms_time))
            _t['infer_time'].clear()
            _t['postproc_time'].clear()
            _t['nms_time'].clear()

    avg_time = all_time / num_images
    if avg_time:
        FPS = 1.0 / avg_time
        print('average detection time: {:.3f}s FPS: {:.3f}'.format(avg_time, FPS))

    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

    print('Evaluating detections')

    APs, mAP = testset.evaluate_detections(all_boxes, save_folder)
    return APs, mAP


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
        img_copy = img.copy() #cv2.resize(img, (512, 512))
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

            detect_bboxes = all_boxes[j][i]

            if True:
                img_ = img_copy.copy()

                for class_id, class_collection in enumerate(detect_bboxes):
                    if len(class_collection) > 0:
                        if class_collection[-1] > 0.2:
                            pt = class_collection

                            img_ = cv2.rectangle(img_.astype(np.uint8), (int(pt[0]), int(pt[1])), (int(pt[2]),
                                                                            int(pt[3])), (0, 0, 255), 1)
                            # cv2.putText(image, 'mark', (int(pt[0]), int(pt[1])), FONT,
                            #             0.3, (255, 255, 255), 1)
                cv2.imshow('result', img_.astype(np.uint8))
                cv2.waitKey(0)

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

    APs, mAP = testset.evaluate_detections(all_boxes, save_folder, iou_all=True)
    return APs, mAP


os.environ['CUDA_VISIBLE_DEVICES'] = '1'
cfg = MARK_512
# mark_root = '/data_ssd2/hzh/paperworks/dataset/augment/ScreenshotsDATA'
mark_root = '/data_ssd2/hzh/paperworks/dataset/augment/BASELINE'
# mark_root = '/data_ssd2/hzh/paperworks/dataset/augment/WEAK'
# mark_root = '/data_ssd2/hzh/paperworks/dataset/augment/BASELINE/VOC2007/Photos_ip8/voc_test'
# mark_root = '/data_ssd2/hzh/paperworks/dataset/augment/WEAK/VOC2007/Photos/marks/video/voc_test'
# mark_root = '/data_ssd2/hzh/paperworks/dataset/augment/STRONG/VOC2007/Photos/marks/video/voc_test'

args.epoch = '15'
net = build_net((512, 512), 2)
# net.load_state_dict(torch.load('weights/SSD_vgg_bn_MARK_512/20210419_fix_gen/SSD_vgg_bn_MARK_epoches_' + args.epoch + '.pth'))
# net.load_state_dict(torch.load('/data_ssd2/hzh/PytorchSSD1215/weights/SSD_vgg_bn_MARK_512/20210424_baseline/SSD_vgg_bn_MARK_epoches_' + args.epoch + '.pth'))
# net.load_state_dict(torch.load('/data_ssd2/hzh/PytorchSSD1215/weights/SSD_vgg_bn_MARK_512/20210503_baseline_validate/SSD_vgg_bn_MARK_epoches_' + args.epoch + '.pth'))
# net.load_state_dict(torch.load("/data_ssd2/hzh/PytorchSSD1215/weights/SSD_vgg_bn_MARK_512/20210503_baseline_validate/SSD_vgg_bn_MARK_epoches_30.pth"))
net.load_state_dict(torch.load("/data_ssd2/hzh/PytorchSSD1215/weights/SSD_vgg_bn_MARK_512/20210513_baseline_aug_new_crop/SSD_vgg_bn_MARK_epoches_38.pth"))
net.to('cuda')
net.eval()
top_k = 300
detector = Detect(2, 0, cfg)

priorbox = PriorBox(cfg)
with torch.no_grad():
    priors = priorbox.forward()

testset = VOCDetection(mark_root, [('2007', 'test')], None, AnnotationTransform())
APs, mAP = test_net_yolt3('/data_ssd2/hzh/PytorchSSD1215/weights/SSD_vgg_bn_MARK_512/20210512_baseline_wo_crop/ss_predict', net, detector, True, testset,
                        BaseTransform((net.size, net.size), (104, 117, 123)),
                        top_k, thresh=0.01)
APs = [str(num) for num in APs]
print(mAP)

