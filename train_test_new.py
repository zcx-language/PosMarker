from __future__ import print_function

import argparse
import pickle
import time

import cv2
import random
import numpy as np
import os
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
from tensorboardX import SummaryWriter

from data import VOCroot, COCOroot, MARKroot_base, VOC_300, VOC_512, MARK_512, COCO_300, COCO_512, COCO_mobile_300, \
    COCODetection, BaseTransform, preproc, preproc_new, preproc_old

from layers.functions import Detect, PriorBox
from layers.modules import MultiBoxLoss
from utils.nms_wrapper import nms
from utils.timer import Timer
from utils.tools import tensor_to_np


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
            # img_temp = cv2.resize(np.array(img_temp), (512, 512), interpolation=cv2.INTER_LINEAR).astype(np.float32)
            crops.append(torch.from_numpy(img_temp).permute(2, 0, 1).unsqueeze(0))
            offsets.append((x, y))
    return offsets, crops


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(
    description='Receptive Field Block Net Training')
parser.add_argument('-v', '--version', default='SSD_vgg_bn',
                    help='RFB_vgg ,RFB_E_vgg RFB_mobile SSD_vgg version.')
parser.add_argument('-s', '--size', default='512',
                    help='300 or 512 input size.')
parser.add_argument('-d', '--dataset', default='MARK',
                    help='VOC or COCO or MARK dataset')
parser.add_argument('--basenet', default='weights/vgg16_reducedfc.pth', help='pretrained base model')
parser.add_argument('--jaccard_threshold', default=0.5,
                    type=float, help='Min Jaccard index for matching')
parser.add_argument('-b', '--batch_size', default=4,
                    type=int, help='Batch size for training')
parser.add_argument('--num_workers', default=4,
                    type=int, help='Number of workers used in dataloading')
parser.add_argument('--cuda', action="store_true", help='Use cuda to train model')
parser.add_argument('--ngpu', default=1, type=int, help='gpus')
parser.add_argument('--lr', '--learning-rate',
                    default=4e-3, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')

parser.add_argument('--resume_net', action="store_true", help='resume net for retraining')
parser.add_argument('--resume_epoch', default=0,
                    type=int, help='resume iter for retraining')

parser.add_argument('-max', '--max_epoch', default=40,
                    type=int, help='max epoch for retraining')
parser.add_argument('--weight_decay', default=5e-4,
                    type=float, help='Weight decay for SGD')
parser.add_argument('-we', '--warm_epoch', default=3,
                    type=int, help='max epoch for retraining')
parser.add_argument('--gamma', default=0.1,
                    type=float, help='Gamma update for SGD')
parser.add_argument('--log_iters', default=True,
                    type=bool, help='Print the loss at each iteration')
parser.add_argument('--save_folder', default='weights/',
                    help='Location to save checkpoint models')
parser.add_argument('--date', default=None)
parser.add_argument('--save_frequency', default=2)
parser.add_argument('--retest', action="store_true",
                    help='test cache results')
parser.add_argument('--test_frequency', default=2)
args = parser.parse_args()

cur_time = time.strftime('_%Y-%m-%d-%H-%M', time.localtime(time.time()))
if not args.date:
    args.date = time.strftime('%Y%m%d', time.localtime(time.time()))
writer = SummaryWriter(os.path.join('runs', args.version+ '_' + args.dataset + '_' + str(args.size)+cur_time))
save_folder = os.path.join(args.save_folder, args.version + '_' + args.dataset + '_' + str(args.size), args.date + '_weak')

if not os.path.exists(save_folder):
    os.makedirs(save_folder)
test_save_dir = os.path.join(save_folder, 'ss_predict')
if not os.path.exists(test_save_dir):
    os.makedirs(test_save_dir)

log_file_path = save_folder + '/train' + cur_time + '.log'
print('saving weights and intermediate result in: %s' % save_folder)


if args.dataset == 'VOC':
    train_sets = [('2007', 'trainval'), ('2012', 'trainval')]
    cfg = (VOC_512, VOC_300)[args.size == '300']
    from data.voc0712 import AnnotationTransform, VOCDetection, detection_collate
# add for mark dataset
elif args.dataset == 'MARK':
    train_sets = [('2007', 'trainval')]
    cfg = MARK_512
    from data.mark import AnnotationTransform, VOCDetection, detection_collate
else:
    train_sets = [('2017', 'train')]
    cfg = (COCO_300, COCO_512)[args.size == '512']

if args.version == 'RFB_vgg':
    from models.RFB_Net_vgg import build_net
elif args.version == 'RFB_E_vgg':
    from models.RFB_Net_E_vgg import build_net
elif args.version == 'RFB_mobile':
    from models.RFB_Net_mobile import build_net

    cfg = COCO_mobile_300
elif args.version == 'SSD_vgg':
    from models.SSD_vgg import build_net
elif args.version == 'SSD_vgg_bn':
    from models.SSD_vgg_bn import build_net
elif args.version == 'FSSD_vgg':
    from models.FSSD_vgg import build_net
elif args.version == 'FRFBSSD_vgg':
    from models.FRFBSSD_vgg import build_net
else:
    print('Unkown version!')
rgb_std = (1, 1, 1)

# 修改img_dim 支持输入图片尺寸不为正方形的情况
# img_dim = (300, 512)[args.size == '512']
# img_dim = None
if args.size == '300':
    img_dim = (300, 300)
elif args.size == '512':
    img_dim = (512, 512)
elif args.size == '1024':
    img_dim = (1024, 576)

if 'vgg' in args.version:
    rgb_means = (104, 117, 123)
elif 'mobile' in args.version:
    rgb_means = (103.94, 116.78, 123.68)

p = (0.6, 0.2)[args.version == 'RFB_mobile']
num_classes = (21, 81)[args.dataset == 'COCO']
num_classes = 2
batch_size = args.batch_size
weight_decay = 0.0005
gamma = 0.1
momentum = 0.9

# 构造检测器
net = build_net(img_dim, num_classes)
print(net)


if not args.resume_net:
    net.init_model(args.basenet)
else:
    # load resume network
    # if args.dataset == 'MARK':
    #     resume_net_path = args.resume_net
    # else:
    resume_net_path = os.path.join(save_folder, args.version + '_' + args.dataset + '_epoches_' + \
                                       str(args.resume_epoch) + '.pth')

    # resume_net_path = "/data_ssd2/hzh/PytorchSSD1215/weights/SSD_vgg_bn_MARK_512/20210424_baseline/SSD_vgg_bn_MARK_epoches_15.pth"

    print('Loading resume network', resume_net_path)
    state_dict = torch.load(resume_net_path)
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        head = k[:7]
        if head == 'module.':
            name = k[7:]  # remove `module.`
        else:
            name = k
        new_state_dict[name] = v
    net.load_state_dict(new_state_dict)

if args.ngpu > 1:
    net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))

if args.cuda:
    net.to("cuda")
    cudnn.benchmark = True

detector = Detect(num_classes, 0, cfg)
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=args.momentum, weight_decay=args.weight_decay)
# optimizer = optim.RMSprop(net.parameters(), lr=args.lr,alpha = 0.9, eps=1e-08,
#                      momentum=args.momentum, weight_decay=args.weight_decay)

criterion = MultiBoxLoss(num_classes, 0.5, True, 0, True, 3, 0.5, False)
priorbox = PriorBox(cfg)
with torch.no_grad():
    priors = priorbox.forward()
# dataset
print('Loading Dataset...')
if args.dataset == 'VOC':
    testset = VOCDetection(
        VOCroot, [('2007', 'test')], None, AnnotationTransform())
    train_dataset = VOCDetection(VOCroot, train_sets, preproc(
        img_dim, rgb_means, rgb_std, p), AnnotationTransform())
# add for mark dataset
elif args.dataset == 'MARK':
    testset = VOCDetection(
        MARKroot_base, [('2007', 'test')], None, AnnotationTransform())

    # TODO: debug为什么会出现Nan？
    # train_dataset = VOCDetection(MARKroot, train_sets, preproc(
    #     img_dim, rgb_means, rgb_std, p, True), AnnotationTransform())

    train_dataset = VOCDetection(MARKroot_base, train_sets, preproc_new(
        img_dim, rgb_means, rgb_std), AnnotationTransform())

elif args.dataset == 'COCO':
    testset = COCODetection(
        COCOroot, [('2017', 'val')], None)
    train_dataset = COCODetection(COCOroot, train_sets, preproc(
        img_dim, rgb_means, rgb_std, p))
else:
    print('Only VOC and COCO are supported now!')
    exit()


# 将一个batch的数据可视化
def tensor_to_np(tensor):

    def tensor_to_np_single(tensor):
        img = tensor.cpu().numpy().transpose((1, 2, 0)).astype(np.uint8)
        # img += np.array(rgb_means[::-1]).astype(np.uint8)
        return img

    img_list = []
    for i in range(tensor.shape[0]):
        img_list.append(tensor_to_np_single(tensor[i]))
    return img_list


def train():

    net.train()
    # loss counters
    loc_loss = 0  # epoch
    conf_loss = 0
    epoch = 0
    if args.resume_net:
        epoch = 0 + args.resume_epoch
    epoch_size = len(train_dataset) // args.batch_size
    max_iter = args.max_epoch * epoch_size

    # stepvalues_VOC = (150 * epoch_size, 200 * epoch_size, 250 * epoch_size)
    stepvalues_VOC = (20 * epoch_size, 30 * epoch_size, 35 * epoch_size)
    stepvalues_COCO = (90 * epoch_size, 120 * epoch_size, 140 * epoch_size)
    stepvalues = (stepvalues_VOC, stepvalues_COCO)[args.dataset == 'COCO']
    print('Training', args.version, 'on', train_dataset.name)
    step_index = 0

    if args.resume_epoch > 0:
        start_iter = args.resume_epoch * epoch_size
    else:
        start_iter = 0

    log_file = open(log_file_path, 'w')

    stepvalues_list = list(stepvalues)
    stepvalues_list = [i//epoch_size for i in stepvalues_list]

    log_file.write(str(args) + '\n' + "lr step: " + str(stepvalues_list) + '\n')
    batch_iterator = None
    mean_loss_c = 0
    mean_loss_l = 0

    resume_flag = True if args.resume_net else False
    # resume_flag = False

    last_loss = 0
    for iteration in range(start_iter, max_iter + 10):
        if iteration % epoch_size == 0:
            # create batch iterator
            batch_iterator = iter(data.DataLoader(train_dataset, batch_size,
                                                  shuffle=True, num_workers=args.num_workers,
                                                  collate_fn=detection_collate))
            loc_loss = 0
            conf_loss = 0
            if epoch % args.save_frequency == 0 and epoch > 0:
                torch.save(net.state_dict(), os.path.join(save_folder, args.version + '_' + args.dataset + '_epoches_' +
                                                          repr(epoch) + '.pth'))
            if epoch % args.test_frequency == 0 and epoch > 0:
                net.eval()
                top_k = (300, 200)[args.dataset == 'COCO']
                if args.dataset in ['VOC', 'MARK'] and not resume_flag:

                    APs, mAP = test_net_yolt3(test_save_dir, net, detector, args.cuda, testset,
                                        BaseTransform((net.size, net.size), rgb_means, rgb_std, (2, 0, 1)),
                                        top_k, thresh=0.01)
                    APs = [str(num) for num in APs]
                    writer.add_scalar('mAP', mAP, epoch)
                    mAP = str(mAP)
                    log_file.write(str(iteration) + ' APs:\n' + '\n'.join(APs))
                    log_file.write('mAP:\n' + mAP + '\n')
                # else:
                #     test_net(test_save_dir, net, detector, args.cuda, testset,
                #              BaseTransform(net.module.size, rgb_means, rgb_std, (2, 0, 1)),
                #              top_k, thresh=0.01)
                resume_flag = False
                net.train()
            epoch += 1

        load_t0 = time.time()
        if iteration in stepvalues:
            step_index = stepvalues.index(iteration) + 1
        lr = adjust_learning_rate(optimizer, args.gamma, epoch, step_index, iteration, epoch_size)

        # load train data
        # images的shape是[batchsize, 3, 512, 512]
        # targets是一个长度为batchsize的list， list中每个元素的shape是[num_gts_of_this_img, 5]
        images, targets = next(batch_iterator)

        # TODO: 修改代码：这里的images是原图(1920*1080)，根据targets裁剪出rois，送入Encoder得到嵌入强度的数据
        if False:
            with torch.no_grad():
                img_list = tensor_to_np(images)
            for i, img in enumerate(img_list):
                target = targets[i].detach().clone().numpy()
                img = img.copy()
                for anno in target:
                    img = cv2.rectangle(img, (int(anno[0]*512), int(anno[1]*512)), (int(anno[2]*512), int(anno[3]*512)), (0, 0, 255), 1)
                cv2.imshow("", img)
                if cv2.waitKey(0) == 27:
                    break


        if args.cuda:
            images = images.to("cuda")
            with torch.no_grad():
                targets = [anno.to("cuda") for anno in targets]
        else:
            images = images.to("cpu")
            with torch.no_grad():
                targets = [anno.to("cpu") for anno in targets]
        # forward
        out = net(images)
        # backprop
        optimizer.zero_grad()
        # arm branch loss
        loss_l, loss_c = criterion(out, priors, targets)
        # odm branch loss

        mean_loss_c = mean_loss_c + loss_c.item()
        mean_loss_l = mean_loss_l + loss_l.item()

        loss = loss_l + loss_c

        # vis
        skip_flag = False
        base_path = '/data_ssd2/hzh/PytorchSSD1215/vis'
        if last_loss and iteration > 10 and float(loss) > 10*last_loss:
            skip_flag = True
            img_list = tensor_to_np(images)
            for i, img in enumerate(img_list):
                img_name = str(iteration) + '_' + str(i) + '_' + str(int(float(loss) // last_loss)) + '.jpg'
                cv2.imwrite(os.path.join(base_path, img_name), img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

        last_loss = float(loss)

        if not skip_flag:
            loss.backward()

        optimizer.step()
        load_t1 = time.time()
        if iteration and iteration % 10 == 0:

            writer.add_scalar('loc_loss', mean_loss_l / 10, iteration)
            writer.add_scalar('conf_loss', mean_loss_c / 10, iteration)
            writer.add_scalar('loss', loss, iteration)
            writer.add_scalar('lr', lr, epoch)

            c_t = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
            print(c_t + ' || Epoch: ' + repr(epoch) + ' || epochiter: ' + repr(iteration % epoch_size) + '/' + repr(epoch_size)
                  + ' || Totel iter: ' +
                  repr(iteration) + ' || L: %.4f C: %.4f || ' % (
                      mean_loss_l / 10, mean_loss_c / 10) +
                  'Batch time: %.4f sec || ' % (load_t1 - load_t0) + 'LR: %.8f' % (lr))
            log_file.write(
                c_t + ' || Epoch: ' + repr(epoch) + ' || epochiter: ' + repr(iteration % epoch_size) + '/' + repr(epoch_size)
                  + ' || Totel iter: ' +
                  repr(iteration) + ' || L: %.4f C: %.4f || ' % (
                      mean_loss_l / 10, mean_loss_c / 10) +
                  'Batch time: %.4f sec || ' % (load_t1 - load_t0) + 'LR: %.8f' % (lr) + '\n')

            mean_loss_c = 0
            mean_loss_l = 0

    log_file.close()
    torch.save(net.state_dict(), os.path.join(save_folder,
                                              'Final_' + args.version + '_' + args.dataset + '.pth'))


def adjust_learning_rate(optimizer, gamma, epoch, step_index, iteration, epoch_size):
    """Sets the learning rate
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    if epoch < args.warm_epoch:
        lr = 1e-6 + (args.lr - 1e-6) * iteration / (epoch_size * (args.warm_epoch - 1))
    else:
        lr = args.lr * (gamma ** (step_index))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def test_net(save_folder, net, detector, cuda, testset, transform, max_per_image=300, thresh=0.005):
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    # dump predictions and assoc. ground truth to text file for now
    num_images = len(testset)
    num_classes = (21, 2)[args.dataset == 'MARK']
    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(num_classes)]

    _t = {'im_detect': Timer(), 'misc': Timer()}
    det_file = os.path.join(save_folder, 'detections.pkl')

    if args.retest:
        f = open(det_file, 'rb')
        all_boxes = pickle.load(f)
        print('Evaluating detections')
        testset.evaluate_detections(all_boxes, save_folder, iou_all=True)
        return

    all_time = 0
    for i in range(num_images):
        img = testset.pull_image(i)
        with torch.no_grad():
            x = transform(img).unsqueeze(0)
        if cuda:
            x = x.to(torch.device("cuda"))

        _t['im_detect'].tic()
        out = net(x=x, test=True)  # forward pass
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

        for j in range(1, num_classes):
            inds = np.where(scores[:, j] > thresh)[0]
            if len(inds) == 0:
                all_boxes[j][i] = np.empty([0, 5], dtype=np.float32)
                continue
            c_bboxes = boxes[inds]
            c_scores = scores[inds, j]
            c_dets = np.hstack((c_bboxes, c_scores[:, np.newaxis])).astype(
                np.float32, copy=False)
            if args.dataset == 'VOC':
                cpu = False
            else:
                cpu = False

            keep = nms(c_dets, 0.45, force_cpu=cpu)
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
    if args.dataset in ['VOC', 'MARK']:
        APs, mAP = testset.evaluate_detections(all_boxes, save_folder, iou_all=True)
        return APs, mAP
    else:
        testset.evaluate_detections(all_boxes, save_folder)


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

            if False:

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


if __name__ == '__main__':
    train()
