from __future__ import print_function

import argparse
import pickle
import time

import numpy as np
import os
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
from tensorboardX import SummaryWriter

from data import VOCroot, MARKroot, VOC_300, VOC_512, VOC_512_MARK, AnnotationTransform, VOCDetection, detection_collate, BaseTransform, preproc
from layers.functions import Detect, PriorBox
from layers.modules import MultiBoxLoss
from utils.nms_wrapper import nms
from utils.timer import Timer


def int_list(args):
    return [int(x) for x in args.split(',')]

parser = argparse.ArgumentParser(
    description='Receptive Field Block Net Training')
parser.add_argument('-v', '--version', default='SSD_shuffle',
                    help='RFB_vgg ,RFB_E_vgg RFB_mobile SSD_vgg version.')
parser.add_argument('-s', '--size', type=int, default=512,
                    help='300 or 512 input size.')
parser.add_argument('-d', '--dataset', default='MARK',
                    help='VOC or COCO dataset')
parser.add_argument(
    '--basenet', default='weights/ShuffleNetV2P.Medium.pth.tar', help='pretrained base model')
parser.add_argument('--jaccard_threshold', default=0.5,
                    type=float, help='Min Jaccard index for matching')
parser.add_argument('-b', '--batch_size', default=16,
                    type=int, help='Batch size for training')
parser.add_argument('--num_workers', default=3,
                    type=int, help='Number of workers used in dataloading')
parser.add_argument('--cuda', action="store_true", help='Use cuda to train model')
parser.add_argument('--ngpu', default=1, type=int, help='gpus')
parser.add_argument('--lr', '--learning-rate',
                    default=1e-2, type=float, help='initial learning rate')
parser.add_argument('--step', type=int_list, help='Step learning rate adjust ,example:"--step 200,400,600"')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')

parser.add_argument('--resume_net', default=False, help='resume net for retraining')
parser.add_argument('--resume_epoch', default=0,
                    type=int, help='resume iter for retraining')

parser.add_argument('-max', '--max_epoch', default=60,
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
parser.add_argument('--date', default='20201118')
parser.add_argument('--save_frequency', default=5)
parser.add_argument('--retest', action="store_true",
                    help='test cache results')
parser.add_argument('--test_frequency', default=5)
args = parser.parse_args()

cur_time = time.strftime('_%Y-%m-%d-%H-%M', time.localtime(time.time()))
writer = SummaryWriter(os.path.join('runs', args.version+ '_' + args.dataset + '_' + str(args.size)+cur_time))
save_folder = os.path.join(args.save_folder, args.version + '_' + args.dataset + '_' + str(args.size), args.date)
if not os.path.exists(save_folder):
    os.makedirs(save_folder)
test_save_dir = os.path.join(save_folder, 'ss_predict')
if not os.path.exists(test_save_dir):
    os.makedirs(test_save_dir)

log_file_path = save_folder + '/train' + cur_time + '.log'
print('saving weights and intermediate result in: %s' % save_folder)


if args.dataset == 'VOC':
    train_sets = [('2007', 'trainval'), ('2012', 'trainval')]
    cfg = (VOC_300, VOC_512)[args.size == '512']
elif args.dataset == 'MARK':
    train_sets = [('2007', 'trainval')]
    cfg = VOC_512_MARK

if args.version == 'SSD_vgg':
    from models.SSD_vgg import build_net
elif args.version == 'SSD_shuffle':
    from models.SSD_shuffle import build_net
else:
    print('Unkown version!')

rgb_std = (1, 1, 1)
img_dim = (300, 512)[args.size == 512]
rgb_means = (104, 117, 123)

p = (0.6, 0.2)[args.version == 'RFB_mobile']
num_classes = (2, 21)[args.dataset == 'VOC']
batch_size = args.batch_size
weight_decay = 0.0005
gamma = 0.1
momentum = 0.9

net = build_net(img_dim, num_classes)
print(net)
if not args.resume_net:
    net.init_model(args.basenet)
else:
    # load resume network
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
elif args.dataset == 'MARK':
    testset = VOCDetection(
        MARKroot, [('2007', 'test')], None, AnnotationTransform())
    train_dataset = VOCDetection(MARKroot, train_sets, preproc(
        img_dim, rgb_means, rgb_std, p))
else:
    print('Only VOC and COCO are supported now!')
    exit()


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

    if args.step:
        stepvalues = tuple([i * epoch_size for i in args.step])
    else:
        stepvalues = (40 * epoch_size, 55 * epoch_size)
    print('learning rate step', [i // epoch_size for i in stepvalues])
    print('Training', args.version, 'on', train_dataset.name)
    step_index = 0

    if args.resume_net:
        start_iter = args.resume_epoch * epoch_size
    else:
        start_iter = 0

    log_file = open(log_file_path, 'w')
    batch_iterator = None
    mean_loss_c = 0
    mean_loss_l = 0
    for iteration in range(start_iter, max_iter + 10):
        if (iteration % epoch_size == 0):
            # create batch iterator
            batch_iterator = iter(data.DataLoader(train_dataset, batch_size,
                                                  shuffle=True, num_workers=args.num_workers,
                                                  collate_fn=detection_collate))
            if epoch % args.save_frequency == 0 and epoch > 0:
                torch.save(net.state_dict(), os.path.join(save_folder, args.version + '_' + args.dataset + '_epoches_' +
                                                          repr(epoch) + '.pth'))
            if epoch % args.test_frequency == 0 and epoch > 0:
                net.eval()
                top_k = 400

                APs, mAP = test_net(test_save_dir, net, detector, args.cuda, testset,
                                    BaseTransform(args.size, rgb_means, rgb_std, (2, 0, 1)),
                                    top_k, thresh=0.01)
                APs = [str(num) for num in APs]
                writer.add_scalar('mAP', mAP, iteration)
                mAP = str(mAP)
                log_file.write(str(iteration) + ' APs:\n' + '\n'.join(APs))
                log_file.write('mAP:\n' + mAP + '\n')
                net.train()
            epoch += 1

        load_t0 = time.time()

        for i in range(len(stepvalues)):
            if i == 0 and iteration < stepvalues[i]:
                step_index = 0
            elif stepvalues[i-1] <= iteration < stepvalues[i]:
                step_index = i
            elif i == len(stepvalues)-1 and iteration >= stepvalues[i]:
                step_index = i+1

        if iteration in stepvalues:
            step_index = stepvalues.index(iteration) + 1
            print('learning rate step', [i // epoch_size for i in stepvalues])

        lr = adjust_learning_rate(optimizer, args.gamma, epoch, step_index, iteration, epoch_size)

        # load train data
        images, targets = next(batch_iterator)

        # print(np.sum([torch.sum(anno[:,-1] == 2) for anno in targets]))

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
        loss.backward()
        optimizer.step()
        load_t1 = time.time()
        if iteration % 10 == 0:

            writer.add_scalar('loc_loss', mean_loss_l / 10, iteration)
            writer.add_scalar('conf_loss', mean_loss_c / 10, iteration)
            writer.add_scalar('loss', loss, iteration)

            print('Epoch:' + repr(epoch) + ' || epochiter: ' + repr(iteration % epoch_size) + '/' + repr(epoch_size)
                  + '|| Totel iter ' +
                  repr(iteration) + ' || L: %.4f C: %.4f||' % (
                      mean_loss_l / 10, mean_loss_c / 10) +
                  'Batch time: %.4f sec. ||' % (load_t1 - load_t0) + 'LR: %.8f' % (lr))
            log_file.write(
                'Epoch:' + repr(epoch) + ' || epochiter: ' + repr(iteration % epoch_size) + '/' + repr(epoch_size)
                + '|| Totel iter ' +
                repr(iteration) + ' || L: %.4f C: %.4f||' % (
                    mean_loss_l / 10, mean_loss_c / 10) +
                'Batch time: %.4f sec. ||' % (load_t1 - load_t0) + 'LR: %.8f' % (lr) + '\n')

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
        lr = 1e-6 + (args.lr - 1e-6) * iteration / (epoch_size * args.warm_epoch)
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
    num_classes = (2, 21)[args.dataset == 'VOC']
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
    APs, mAP = testset.evaluate_detections(all_boxes, save_folder)
    return APs, mAP


if __name__ == '__main__':
    train()
    # net.eval()
    # test_net(test_save_dir, net, detector, args.cuda, testset,
    #          BaseTransform(args.size, rgb_means, rgb_std, (2, 0, 1)),
    #          400, thresh=0.01)
