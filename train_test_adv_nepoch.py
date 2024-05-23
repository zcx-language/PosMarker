from __future__ import print_function

import os
import time
import argparse
import pickle

import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
from tensorboardX import SummaryWriter

from data import MARK_512, MARKroot_adv, BaseTransform, preproc_adv
from layers.functions import Detect, PriorBox
from layers.modules import MultiBoxLoss
from models.DotsGenerator_yuv import DotsGenerator50

from utils.tools import *
from utils.nms_wrapper import nms
from utils.timer import Timer

# debug
# torch.autograd.set_detect_anomaly(True)

parser = argparse.ArgumentParser(
    description='Mark Detection Training')

parser.add_argument('-v', '--version', default='SSD_vgg_bn',
                    help='SSD_vgg_bn and SSD_vgg version.')

parser.add_argument('-s', '--size', default='512',
                    help='300 or 512 input size.')

parser.add_argument('-d', '--dataset', default='MARK',
                    help='MARK dataset')

parser.add_argument('--basenet', default='weights/vgg16_reducedfc.pth', help='pretrained base model')

parser.add_argument('--jaccard_threshold', default=0.5,
                    type=float, help='Min Jaccard index for matching')

parser.add_argument('-b', '--batch_size', default=8,
                    type=int, help='Batch size for training')

parser.add_argument('--num_workers', default=4,
                    type=int, help='Number of workers used in dataloading')

parser.add_argument('--cuda', action="store_true", help='Use cuda to train model')

parser.add_argument('--ngpu', default=1, type=int, help='gpus')

parser.add_argument('--lr_d', '--learning-rate-det',
                    default=1e-7, type=float, help='initial learning rate of detector')

parser.add_argument('--lr_g', '--learning-rate-gen',
                    default=4e-3, type=float, help='initial learning rate of generator')

parser.add_argument('--momentum', default=0.9, type=float, help='momentum')

parser.add_argument('--resume_net', action="store_true", help='resume net for retraining')

parser.add_argument('--resume_epoch', default=0,
                    type=int, help='resume iter for retraining')

parser.add_argument('-max', '--max_epoch', default=50,
                    type=int, help='max epoch for retraining')

parser.add_argument('--weight_decay', default=5e-4,
                    type=float, help='Weight decay for SGD')

parser.add_argument('-we', '--warm_epoch', default=2,
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

parser.add_argument('--note', default="",
                    help='note on this training')

args = parser.parse_args()

# 当前时间
cur_time = time.strftime('_%Y-%m-%d-%H-%M', time.localtime(time.time()))

# 如果不指明date，选择当前日期
if not args.date:
    args.date = time.strftime('%Y%m%d', time.localtime(time.time()))

# tensorboard writer
writer = SummaryWriter(os.path.join('runs', args.version + '_' + args.dataset + '_'
                                    + str(args.size)+cur_time) + ' ' + args.note)

# 保存目录
save_folder = os.path.join(args.save_folder, args.version + '_' + args.dataset + '_' + str(args.size), args.date + '_' + args.note)

if not os.path.exists(save_folder):
    os.makedirs(save_folder)

# 测试文件保存目录
test_save_dir = os.path.join(save_folder, 'ss_predict')
if not os.path.exists(test_save_dir):
    os.makedirs(test_save_dir)

log_file_path = save_folder + '/train' + cur_time + '.log'
print('saving weights and intermediate result in: %s' % save_folder)

if args.dataset == 'MARK':
    train_sets = [('2007', 'trainval')]
    cfg = MARK_512
    from data.mark import AnnotationTransform, VOCDetection, detection_collate
else:
    raise NotImplementedError("Other dataset not supported!")


if args.version == 'SSD_vgg':
    from models.SSD_vgg import build_net
elif args.version == 'SSD_vgg_bn':
    from models.SSD_vgg_bn import build_net
else:
    raise NotImplementedError("Other architecture not supported!")

rgb_std = (1, 1, 1)
rgb_means = (104, 117, 123)  # only for VGG?
p = 0.6
num_classes = 2

# TODO 修改img_dim 支持输入图片尺寸不为正方形的情况
if args.size == '300':
    img_dim = (300, 300)
elif args.size == '512':
    img_dim = (512, 512)
elif args.size == '1024':
    img_dim = (1024, 576)


# 构造检测器
net = build_net(img_dim, num_classes)
print(net)

# 构造编码器
dg = DotsGenerator50()
dg.load_state_dict(torch.load('/data_ssd2/hzh/paperworks/gen/gen_epoch6.pth'))

if not args.resume_net:
    net.init_model(args.basenet)
else:
    resume_net_path = os.path.join(save_folder, args.version + '_' + args.dataset + '_epoches_' + \
                                       str(args.resume_epoch) + '.pth')
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
    dg.to("cuda")
    cudnn.benchmark = True

detector = Detect(num_classes, 0, cfg)

# 优化器
optimizer_D = optim.SGD(net.parameters(), lr=args.lr_d,
                        momentum=args.momentum, weight_decay=args.weight_decay)
optimizer_G = optim.SGD(dg.parameters(), lr=args.lr_g,
                        momentum=args.momentum, weight_decay=args.weight_decay)

criterion = MultiBoxLoss(num_classes, 0.5, True, 0, True, 3, 0.5, False)
priorbox = PriorBox(cfg)


with torch.no_grad():
    priors = priorbox.forward()


# dataset
print('Loading Dataset...')
# add for mark dataset
if args.dataset == 'MARK':
    testset = VOCDetection(
        MARKroot_adv, [('2007', 'test')], None, AnnotationTransform())

    # TODO: debug为什么会出现Nan？
    train_dataset = VOCDetection(MARKroot_adv, train_sets, preproc_adv(
        img_dim, rgb_means, rgb_std), AnnotationTransform())


# 用来测试生成器的图片和标注
# test_image = cv2.imread('/data_ssd2/hzh/paperworks/dataset/original/screenshots_jpg/00036.jpg')
# test_image = torch.from_numpy(test_image).permute(2, 0, 1).float().cuda()
# test_targets = torch.tensor([[300, 400, 350, 450, 1], [500, 900, 550, 950, 1],
#                              [800, 600, 850, 650, 1], [1000, 500, 1050, 550, 1]])


def train():

    net.train()

    # 判断从第几个epoch开始
    epoch = 0
    if args.resume_net:
        epoch = 0 + args.resume_epoch

    # 每个epoch迭代次数和训练总迭代数
    epoch_size = len(train_dataset) // args.batch_size
    max_iter = args.max_epoch * epoch_size

    # lr变化区间
    stepvalues = (30 * epoch_size, 40 * epoch_size, 70 * epoch_size)
    stepvalues_list = [i // epoch_size for i in list(stepvalues)]
    print('Training', args.version, 'on', train_dataset.name)

    step_index = 0

    if args.resume_epoch > 0:
        start_iter = args.resume_epoch * epoch_size
    else:
        start_iter = 0

    # 训练log
    log_file = open(log_file_path, 'w')
    log_file.write(str(args) + '\n' + "lr step: " + str(stepvalues_list) + '\n')
    batch_iterator = None

    # 用来在命令行显示
    mean_loss_c = 0
    mean_loss_l = 0
    mean_loss_v = 0

    resume_flag = True if args.resume_net else False

    last_loss_d = 0
    for iteration in range(start_iter, max_iter + 10):
        if iteration % epoch_size == 0:

            # torch.save(dg.state_dict(), os.path.join(save_folder, 'DotGenerator_' + args.dataset + '_epoches_' +
            #                                          repr(epoch) + '.pth'))

            # 每个epoch保存生成器生成的图片
            dg.eval()
            with torch.no_grad():
                gt_crops = dg.get_crops(test_image, test_targets)
                out = dg(gt_crops)
                image_ = dg.rewrite_image(out, test_image, test_targets)
                image_ = image_.permute(1, 2, 0).detach().cpu().numpy()
                cv2.imwrite(os.path.join(save_folder, 'epoch' + str(epoch) + '.jpg'), image_,
                            [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            dg.train()

            # create batch iterator
            batch_iterator = iter(data.DataLoader(train_dataset, args.batch_size,
                                                  shuffle=True, num_workers=args.num_workers,
                                                  collate_fn=detection_collate))
            if epoch % args.save_frequency == 0 and epoch > 0:
                torch.save(net.state_dict(), os.path.join(save_folder, args.version + '_' + args.dataset + '_epoches_' +
                                                          repr(epoch) + '.pth'))
                torch.save(dg.state_dict(), os.path.join(save_folder, 'DotGenerator_' + args.dataset + '_epoches_' +
                                                         repr(epoch) + '.pth'))
            if epoch % args.test_frequency == 0 and epoch > 0:

                # 同时重新标记测试集

                dg.eval()
                test_anno_path = os.path.join(MARKroot_adv, 'VOC2007/Annotations')
                test_img_path = os.path.join(MARKroot_adv, 'VOC2007/JPEGImages_bak')
                test_mark_path = os.path.join(MARKroot_adv, 'VOC2007/JPEGImages')
                test_txt_path = os.path.join(MARKroot_adv, 'VOC2007/ImageSets/Main/test.txt')
                with open(test_txt_path, 'r') as test_file:
                    for line in test_file.readlines():
                        line = line.strip()
                        if line:
                            img_full_name = os.path.join(test_img_path, line + '.jpg')

                            test_image_ = cv2.imread(img_full_name)
                            test_image_ = torch.from_numpy(test_image_).permute(2, 0, 1).float().cuda()

                            test_targets_ = getAnnotBoxLoc_XML(os.path.join(test_anno_path, line + '.xml'))
                            test_targets_ = torch.from_numpy(np.array(test_targets_)).cuda()

                            with torch.no_grad():
                                gt_crops = dg.get_crops(test_image_, test_targets_)
                                out = dg(gt_crops)
                                image_ = dg.rewrite_image(out, test_image_, test_targets_)
                                image_ = image_.permute(1, 2, 0).detach().cpu().numpy()
                                cv2.imwrite(os.path.join(test_mark_path, line + '.jpg'), image_,
                                            [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                print("rewrite test dataset done.")
                dg.train()

                net.eval()
                top_k = (300, 200)[args.dataset == 'COCO']
                if args.dataset in ['VOC', 'MARK'] and not resume_flag:

                    APs, mAP = test_net(test_save_dir, net, detector, args.cuda, testset,
                                        BaseTransform((net.size, net.size), rgb_means, rgb_std, (2, 0, 1)),
                                        top_k, thresh=0.01)
                    APs = [str(num) for num in APs]
                    writer.add_scalar('mAP', mAP, epoch)
                    mAP = str(mAP)
                    log_file.write(str(iteration) + ' APs:\n' + '\n'.join(APs))
                    log_file.write('mAP:\n' + mAP + '\n')

                resume_flag = False
                net.train()
            epoch += 1

        load_t0 = time.time()
        if iteration in stepvalues:
            step_index = stepvalues.index(iteration) + 1

        # 调整optimizer的lr
        # FIXME 只调整了一个optimizer的lr
        lr_g = adjust_learning_rate_D(optimizer_G, args.gamma, epoch, step_index, iteration, epoch_size)
        lr_d = adjust_learning_rate_G(optimizer_D, args.gamma, epoch, step_index, iteration, epoch_size)

        # ------------------------------------------------
        #                 Train Generator
        # ------------------------------------------------

        optimizer_G.zero_grad()
        net.eval()

        # load train data
        # images的shape是[batchsize, 3, 1080, 1920] float32
        # targets是一个长度为batchsize的list， list中每个元素的shape是[num_gts_of_this_img, 5] float32
        images, targets = next(batch_iterator)
        if args.cuda:
            images = images.to('cuda')
            with torch.no_grad():
                targets = [target.to('cuda') for target in targets]

        # TODO: 调用DotsGenerators，得到打上标记点的图片
        image_list = []
        target_list = []
        loss_v = 0
        for idx in range(args.batch_size):
            image = images[idx]  # [3, 1080, 1920]
            image_o = image.clone()
            target = targets[idx]  # [x, 5]
            # image = dg(image, target)
            gt_crops = dg.get_crops(image, target)  # [num_gts, 3, 40, 40]

            out = dg(gt_crops)
            image_r = dg.rewrite_image(out, image, target)

            # image_2 = image/255
            # image_o = torch.div(image_o, torch.tensor(255.))
            # image = torch.div(image, torch.tensor(255.))

            # loss_v += visible_criterion(image_o, image, dot_list_format)
            # image_r = image / 1.
            # TODO 为啥在loss_v之前修改image会报错，在loss_v之后不会？
            # loss_v = loss_v + nn.MSELoss()(image_o, image) / gt_crops.shape[0]
            # loss_v = loss_v + nn.MSELoss()(image_o, image)
            # loss_v = loss_v + CropMSELoss(image_o, image, dot_list_format)
            # loss_v = loss_v + visible_criterion(image_o, image_r, target)
            loss_v = loss_v + nn.MSELoss()(image_o, image_r)

            # print('before before backward')
            # for n, p in net.named_parameters():
            #     print(n, p._version)
            ###############################################################
            # 需要修改
            # 1.对image和targets进行增广
            # 1.1.水平翻转
            # boxes = target[:, :4]  # 和target共享内存
            # labels = target[:, -1]
            #
            # # 转为numpy
            # image = image.permute(1, 2, 0).detach().cpu().numpy()
            # # image = image.cpu().numpy()
            # boxes = boxes.cpu().numpy()
            #
            # image, boxes = _mirror(image, boxes)
            #
            # # 1.2.旋转
            # image, boxes = _rotate(image, boxes)
            #
            # # 1.3.透视
            # image, boxes = _perspect(image, boxes)
            #
            # # 1.4.噪声
            # image, boxes = _noise(image, boxes)
            #
            # # 1.5. TODO add crop op
            # # 1.6.转成pytorch格式
            # image = torch.from_numpy(image.transpose(2, 0, 1)).float()
            # target = torch.cat((torch.from_numpy(boxes), labels), 1)
            ###############################################################
            image_list.append(image_r.unsqueeze(0))

            # normalize for SSD
            height, width = image_r.shape[1:]
            bboxes = target[:, :4].clone()
            bboxes[:, 0::2] /= width
            bboxes[:, 1::2] /= height

            target_list.append(torch.cat((bboxes, target[:, -1].unsqueeze(1)), dim=1))

        # 一个batch的loss
        loss_v = loss_v / args.batch_size

        # 2.concat images和targets
        # TODO: 检查是否有错？
        # images = torch.cat(image_list, 0)
        # targets = target_list

        # 2.1.concat images -> [bs, 3, 512, 512]
        images = torch.cat(image_list, 0)
        # 2.2.concat targets -> List
        targets = target_list

        # resize
        images = F.interpolate(images, img_dim, mode='bilinear')

        # forward
        out = net(images)
        # backprop

        # 一个batch的位置和置信度loss
        loss_l, loss_c = criterion(out, priors, targets)
        mean_loss_v += loss_v

        # print('before backward')
        # for n, p in net.named_parameters():
        #     print(n, p._version)

        # 反传
        loss_g = loss_l + loss_c + loss_v
        loss_g.backward()
        optimizer_G.step()
        net.train()

        # ------------------------------------------------
        #                 Train Detector
        # ------------------------------------------------
        # TODO: 调用DotsGenerators，得到打上标记点的图片

        optimizer_D.zero_grad()
        dg.eval()

        # forward
        out = net(images.detach())
        # backprop

        loss_l, loss_c = criterion(out, priors, targets)
        loss_d = loss_l + loss_c

        mean_loss_c = mean_loss_c + loss_c.item()
        mean_loss_l = mean_loss_l + loss_l.item()
        # mean_loss_all += (mean_loss_c + mean_loss_l)

        # vis
        skip_flag = False
        base_path = '/data_ssd2/hzh/PytorchSSD1215/vis'
        if last_loss_d and iteration > 10 and float(loss_d) > 10 * last_loss_d:
            skip_flag = True
            img_list = tensor_to_np(images)
            for i, img in enumerate(img_list):
                img_name = str(iteration) + '_' + str(i) + '_' + str(int(float(loss_d) // last_loss_d)) + '.jpg'
                cv2.imwrite(os.path.join(base_path, img_name), img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

        last_loss_d = float(loss_d)

        if not skip_flag:
            loss_d.backward()

        optimizer_D.step()
        dg.train()

        load_t1 = time.time()
        if iteration and iteration % 10 == 0:

            # 用来显示的loss，迭代10次一次
            writer.add_scalar('loc_loss', mean_loss_l / 10, iteration)
            writer.add_scalar('conf_loss', mean_loss_c / 10, iteration)
            writer.add_scalar('visbile_loss', mean_loss_v / 10, iteration)

            # 实时的loss
            writer.add_scalar('loss', loss_d + loss_v, iteration)
            writer.add_scalar('lr_g', lr_g, epoch)
            writer.add_scalar('lr_d', lr_d, epoch)

            c_t = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
            print(c_t + ' || Epoch: ' + repr(epoch) + ' || epochiter: ' + repr(iteration % epoch_size) + '/' + repr(epoch_size)
                  + ' || Totel iter: ' +
                  repr(iteration) + ' || L: %-8.4f C: %-8.4f V: %-8.4f || ' % (
                      # mean_loss_l / 10, mean_loss_c / 10, mean_loss_v / 10) +
                      mean_loss_l / 10, mean_loss_c / 10, mean_loss_v / 10) +
                  'Batch time: %.4f sec || ' % (load_t1 - load_t0) + 'G LR: %.8f' % lr_g + '  D LR: %.8f' % lr_d)
            log_file.write(
                c_t + ' || Epoch: ' + repr(epoch) + ' || epochiter: ' + repr(iteration % epoch_size) + '/' + repr(epoch_size)
                  + ' || Totel iter: ' +
                  repr(iteration) + ' || L: %-8.4f C: %-8.4f V: %-8.4f || ' % (
                      mean_loss_l / 10, mean_loss_c / 10, mean_loss_v / 10) +
                  'Batch time: %.4f sec || ' % (load_t1 - load_t0) + 'G LR: %.8f' % lr_g + '  D LR: %.8f' % lr_d + '\n')

            mean_loss_c = 0
            mean_loss_l = 0
            mean_loss_v = 0

    log_file.close()
    torch.save(net.state_dict(), os.path.join(save_folder,
                                              'Final_' + args.version + '_' + args.dataset + '.pth'))


def adjust_learning_rate_D(optimizer, gamma, epoch, step_index, iteration, epoch_size):
    """Sets the learning rate
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    if epoch < args.warm_epoch:
        lr = 1e-6 + (args.lr_d - 1e-6) * iteration / (epoch_size * (args.warm_epoch - 1))
        lr = 1e-8 + (args.lr_d - 1e-8) * iteration / (epoch_size * (args.warm_epoch - 1))
    else:
        lr = args.lr_d * (gamma ** (step_index))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def adjust_learning_rate_G(optimizer, gamma, epoch, step_index, iteration, epoch_size):
    """Sets the learning rate
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    if epoch < args.warm_epoch:
        lr = 1e-6 + (args.lr_g - 1e-6) * iteration / (epoch_size * (args.warm_epoch - 1))
    else:
        lr = args.lr_g * (gamma ** (step_index))
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
        testset.evaluate_detections(all_boxes, save_folder)
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
        APs, mAP = testset.evaluate_detections(all_boxes, save_folder)
        return APs, mAP
    else:
        testset.evaluate_detections(all_boxes, save_folder)


if __name__ == '__main__':
    train()
