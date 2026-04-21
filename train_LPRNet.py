# -*- coding: utf-8 -*-
# /usr/bin/env/python3

'''
Pytorch implementation for LPRNet.
Author: aiboy.wei@outlook.com .
'''

from data.load_data import CHARS, CHARS_DICT, LPRDataLoader
from model.LPRNet import build_lprnet
# import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import *
from torch import optim
import torch.nn as nn
import numpy as np
import argparse
import logging
import torch
import time
import os
import math
import sys


logger = logging.getLogger("LPRNet")
PROVINCE_CHARS = set(CHARS[:31])
DEFAULT_PROVINCE_CHAR = CHARS[0]


def setup_logging():
    if logger.handlers:
        return

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

def sparse_tuple_for_ctc(T_length, lengths):
    input_lengths = []
    target_lengths = []

    for ch in lengths:
        input_lengths.append(T_length)
        target_lengths.append(ch)

    return tuple(input_lengths), tuple(target_lengths)

def adjust_learning_rate(optimizer, cur_epoch, base_lr, lr_schedule, max_epoch, strategy='cosine', min_lr_ratio=0.01, warmup_epochs=0):
    """
    Sets the learning rate.
    strategy='step': multi-step decay
    strategy='cosine': cosine annealing with optional warmup and min lr floor
    """
    strategy = str(strategy).lower()

    if strategy == 'step':
        # Decay by 0.1 at each milestone and keep decayed lr in late epochs.
        decay_count = sum(1 for e in lr_schedule if cur_epoch >= e)
        lr = base_lr * (0.1 ** decay_count)
    else:
        min_lr = max(0.0, base_lr * min_lr_ratio)
        if warmup_epochs > 0 and cur_epoch <= warmup_epochs:
            lr = base_lr * float(cur_epoch) / float(max(1, warmup_epochs))
        else:
            total = max(1, max_epoch - warmup_epochs)
            progress = float(cur_epoch - warmup_epochs) / float(total)
            progress = min(max(progress, 0.0), 1.0)
            lr = min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


def decode_plate(indices):
    return ''.join(CHARS[int(index)] for index in indices)


def normalize_plate_text(plate_text):
    chinese_chars = [c for c in plate_text if c in PROVINCE_CHARS]
    non_chinese_chars = [c for c in plate_text if c not in PROVINCE_CHARS]
    province_char = chinese_chars[0] if chinese_chars else DEFAULT_PROVINCE_CHAR
    return province_char + ''.join(non_chinese_chars)


def decode_plate_for_log(indices):
    plate_text = decode_plate(indices)
    return normalize_plate_text(plate_text)


def resolve_num_workers(num_workers):
    workers = max(0, int(num_workers))
    if os.name == 'nt' and workers > 0:
        logger.warning("Windows detected: force num_workers=0 to avoid Ctrl+C hang (requested=%d)", workers)
        return 0
    return workers

def get_parser():
    parser = argparse.ArgumentParser(description='parameters to train net')
    parser.add_argument('--max_epoch', default=15, type=int, help='epoch to train the network')
    parser.add_argument('--img_size', default=[94, 24], help='the image size')
    parser.add_argument('--train_img_dirs', default="./data/test", help='the train images path')
    parser.add_argument('--test_img_dirs', default="./data/test", help='the test images path')
    parser.add_argument('--dropout_rate', default=0.5, type=float, help='dropout rate.')
    parser.add_argument('--learning_rate', default=0.1, type=float, help='base value of learning rate.')
    parser.add_argument('--lpr_max_len', default=8, type=int, help='license plate number max length.')
    parser.add_argument('--train_batch_size', default=128, type=int, help='training batch size.')
    parser.add_argument('--test_batch_size', default=120, type=int, help='testing batch size.')
    parser.add_argument('--phase_train', default=True, type=bool, help='train or test phase flag.')
    parser.add_argument('--num_workers', default=0, type=int, help='Number of workers used in dataloading. On Windows, values >0 are forced to 0 for stable Ctrl+C exit.')
    parser.add_argument('--cuda', default=True, type=bool, help='Use cuda to train model')
    parser.add_argument('--resume_epoch', default=0, type=int, help='resume iter for retraining')
    parser.add_argument('--save_interval', default=2000, type=int, help='interval for save model state dict')
    parser.add_argument('--test_interval', default=2000, type=int, help='interval for evaluate')
    parser.add_argument('--log_interval', default=20, type=int, help='interval for training log output by epoch')
    parser.add_argument('--eval_log_interval', default=50, type=int, help='interval for correct evaluation sample log output')
    parser.add_argument('--epoch_iter', default=0, type=int, help='manual iterations per epoch, 0 means auto by dataset size')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight_decay', default=2e-5, type=float, help='Weight decay for SGD')
    parser.add_argument('--lr_schedule', default=[4, 8, 12, 14, 16], type=int, nargs='+', help='schedule for learning rate.')
    parser.add_argument('--lr_strategy', default='cosine', choices=['step', 'cosine'], help='learning rate strategy')
    parser.add_argument('--warmup_epochs', default=3, type=int, help='warmup epochs for cosine strategy')
    parser.add_argument('--min_lr_ratio', default=0.01, type=float, help='minimum lr ratio relative to base lr for cosine strategy')
    parser.add_argument('--augment', default=True, type=bool, help='enable training data augmentation')
    parser.add_argument('--aug_prob', default=0.7, type=float, help='probability to apply augmentation per sample')
    parser.add_argument('--color_jitter', default=0.2, type=float, help='color jitter strength')
    parser.add_argument('--noise_std', default=6.0, type=float, help='std of gaussian noise in augmentation')
    parser.add_argument('--save_folder', default='./weights/', help='Location to save checkpoint models')
    # parser.add_argument('--pretrained_model', default='./weights/Final_LPRNet_model.pth', help='pretrained base model')
    parser.add_argument('--pretrained_model', default='', help='pretrained base model')

    args = parser.parse_args()

    return args

def collate_fn(batch):
    imgs = []
    labels = []
    lengths = []
    for _, sample in enumerate(batch):
        img, label, length = sample
        imgs.append(torch.from_numpy(img))
        labels.extend(label)
        lengths.append(length)
    labels = np.asarray(labels).flatten().astype(np.int64)

    return (torch.stack(imgs, 0), torch.from_numpy(labels), lengths)

def train():
    args = get_parser()
    setup_logging()

    logger.info("Starting training script")
    logger.info("Arguments: %s", args)

    args.num_workers = resolve_num_workers(args.num_workers)
    logger.info("Effective dataloader workers: %d", args.num_workers)

    T_length = 18 # args.lpr_max_len
    epoch = 0 + args.resume_epoch
    loss_val = 0

    if not os.path.exists(args.save_folder):
        os.mkdir(args.save_folder)
        logger.info("Created save folder: %s", args.save_folder)
    else:
        logger.info("Using save folder: %s", args.save_folder)

    lprnet = build_lprnet(lpr_max_len=args.lpr_max_len, phase=args.phase_train, class_num=len(CHARS), dropout_rate=args.dropout_rate)
    device = torch.device("cuda:0" if args.cuda else "cpu")
    lprnet.to(device)
    logger.info("Successful to build network on device: %s", device)
    logger.info("Model class count: %d, max plate length: %s, dropout: %s", len(CHARS), args.lpr_max_len, args.dropout_rate)

    # load pretrained model
    if args.pretrained_model:
        lprnet.load_state_dict(torch.load(args.pretrained_model))
        logger.info("Loaded pretrained model successfully: %s", args.pretrained_model)
    else:
        def xavier(param):
            nn.init.xavier_uniform(param)

        def weights_init(m):
            for key in m.state_dict():
                if key.split('.')[-1] == 'weight':
                    if 'conv' in key:
                        nn.init.kaiming_normal_(m.state_dict()[key], mode='fan_out')
                    if 'bn' in key:
                        m.state_dict()[key][...] = xavier(1)
                elif key.split('.')[-1] == 'bias':
                    m.state_dict()[key][...] = 0.01

        lprnet.backbone.apply(weights_init)
        lprnet.container.apply(weights_init)
        logger.info("Initialized network weights successfully")

    # define optimizer
    # optimizer = optim.SGD(lprnet.parameters(), lr=args.learning_rate,
    #                       momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer = optim.RMSprop(lprnet.parameters(), lr=args.learning_rate, alpha = 0.9, eps=1e-08,
                         momentum=args.momentum, weight_decay=args.weight_decay)
    train_img_dirs = os.path.expanduser(args.train_img_dirs)
    test_img_dirs = os.path.expanduser(args.test_img_dirs)
    train_dataset = LPRDataLoader(
        train_img_dirs.split(','),
        args.img_size,
        args.lpr_max_len,
        augment=args.augment,
        aug_prob=args.aug_prob,
        color_jitter=args.color_jitter,
        noise_std=args.noise_std,
    )
    test_dataset = LPRDataLoader(test_img_dirs.split(','), args.img_size, args.lpr_max_len, augment=False)
    logger.info("Resolved train directories: %s", train_img_dirs)
    logger.info("Resolved test directories: %s", test_img_dirs)

    if len(train_dataset) == 0:
        raise ValueError("No training images were found. Check --train_img_dirs: {}".format(train_img_dirs))
    if len(test_dataset) == 0:
        raise ValueError("No test images were found. Check --test_img_dirs: {}".format(test_img_dirs))

    logger.info("Training samples: %d", len(train_dataset))
    logger.info("Test samples: %d", len(test_dataset))

    auto_epoch_size = max(1, math.ceil(len(train_dataset) / args.train_batch_size))
    epoch_size = args.epoch_iter if args.epoch_iter > 0 else auto_epoch_size
    max_iter = args.max_epoch * epoch_size
    logger.info("Train batch size: %d, test batch size: %d", args.train_batch_size, args.test_batch_size)
    logger.info("Auto iterations per epoch: %d", auto_epoch_size)
    logger.info("Effective iterations per epoch: %d, total iterations: %d", epoch_size, max_iter)
    logger.info("Training log interval: %d, evaluation log interval: %d", args.log_interval, args.eval_log_interval)
    logger.info(
        "LR strategy: %s | warmup_epochs: %d | min_lr_ratio: %.6f | milestones: %s",
        args.lr_strategy,
        args.warmup_epochs,
        args.min_lr_ratio,
        args.lr_schedule,
    )
    logger.info(
        "Augmentation: %s | aug_prob: %.2f | color_jitter: %.2f | noise_std: %.2f",
        args.augment,
        args.aug_prob,
        args.color_jitter,
        args.noise_std,
    )

    ctc_loss = nn.CTCLoss(blank=len(CHARS)-1, reduction='mean') # reduction: 'none' | 'mean' | 'sum'

    if args.resume_epoch > 0:
        start_iter = args.resume_epoch * epoch_size
    else:
        start_iter = 0

    logger.info("Resume epoch: %d, start iteration: %d", args.resume_epoch, start_iter)

    interrupted = False
    epoch_loss_steps = 0
    epoch_time_sum = 0.0
    current_lr = args.learning_rate
    try:
        for iteration in range(start_iter, max_iter):
            if iteration % epoch_size == 0:
                # create batch iterator
                batch_iterator = iter(DataLoader(train_dataset, args.train_batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn))
                loss_val = 0
                epoch_loss_steps = 0
                epoch_time_sum = 0.0
                epoch += 1
                logger.info("Starting epoch %d/%d", epoch, args.max_epoch)

            if iteration !=0 and iteration % args.save_interval == 0:
                checkpoint_path = args.save_folder + 'LPRNet_' + '_iteration_' + repr(iteration) + '.pth'
                torch.save(lprnet.state_dict(), checkpoint_path)
                logger.info("Saved checkpoint at iteration %d: %s", iteration, checkpoint_path)

            if (iteration + 1) % args.test_interval == 0:
                logger.info("Starting evaluation at iteration %d", iteration + 1)
                Greedy_Decode_Eval(lprnet, test_dataset, args)
                # lprnet.train() # should be switch to train mode

            start_time = time.time()
            # load train data
            try:
                images, labels, lengths = next(batch_iterator)
            except StopIteration:
                batch_iterator = iter(DataLoader(train_dataset, args.train_batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn))
                images, labels, lengths = next(batch_iterator)
            # labels = np.array([el.numpy() for el in labels]).T
            # print(labels)
            # get ctc parameters
            input_lengths, target_lengths = sparse_tuple_for_ctc(T_length, lengths)
            # update lr
            lr = adjust_learning_rate(
                optimizer,
                epoch,
                args.learning_rate,
                args.lr_schedule,
                args.max_epoch,
                strategy=args.lr_strategy,
                min_lr_ratio=args.min_lr_ratio,
                warmup_epochs=args.warmup_epochs,
            )
            current_lr = lr

            if args.cuda:
                images = Variable(images, requires_grad=False).cuda()
                labels = Variable(labels, requires_grad=False).cuda()
            else:
                images = Variable(images, requires_grad=False)
                labels = Variable(labels, requires_grad=False)

            # forward
            logits = lprnet(images)
            log_probs = logits.permute(2, 0, 1) # for ctc loss: T x N x C
            # print(labels.shape)
            log_probs = log_probs.log_softmax(2).requires_grad_()
            # log_probs = log_probs.detach().requires_grad_()
            # print(log_probs.shape)
            # backprop
            optimizer.zero_grad()
            loss = ctc_loss(log_probs, labels, input_lengths=input_lengths, target_lengths=target_lengths)
            if loss.item() == np.inf:
                logger.warning("Encountered infinite loss at iteration %d, skipping update", iteration)
                continue
            loss.backward()
            optimizer.step()
            loss_val += loss.item()
            epoch_loss_steps += 1
            end_time = time.time()
            batch_time = end_time - start_time
            epoch_time_sum += batch_time
            if (iteration + 1) % epoch_size == 0 and (epoch % max(1, args.log_interval) == 0):
                avg_loss = loss_val / max(1, epoch_loss_steps)
                avg_batch_time = epoch_time_sum / max(1, epoch_loss_steps)
                logger.info(
                    "Epoch %d | avg_loss %.4f | avg_batch %.4fs | lr %.8f",
                    epoch,
                    avg_loss,
                    avg_batch_time,
                    current_lr,
                )
    except KeyboardInterrupt:
        interrupted = True
        logger.warning("KeyboardInterrupt received, stopping training gracefully...")

    if interrupted:
        interrupt_model_path = args.save_folder + 'Interrupted_LPRNet_model.pth'
        torch.save(lprnet.state_dict(), interrupt_model_path)
        logger.info("Saved interrupted model to %s", interrupt_model_path)
        return
    # final test
    logger.info("Training complete, running final evaluation")
    Greedy_Decode_Eval(lprnet, test_dataset, args)

    # save final parameters
    final_model_path = args.save_folder + 'Final_LPRNet_model.pth'
    torch.save(lprnet.state_dict(), final_model_path)
    logger.info("Saved final model to %s", final_model_path)

def Greedy_Decode_Eval(Net, datasets, args):
    was_training = Net.training
    Net.eval()

    if len(datasets) == 0:
        raise ValueError("The test dataset is empty.")

    epoch_size = max(1, math.ceil(len(datasets) / args.test_batch_size))
    batch_iterator = iter(DataLoader(datasets, args.test_batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn))
    logger.info("Evaluating on %d samples using %d batches", len(datasets), epoch_size)

    Tp = 0
    Tn_1 = 0
    Tn_2 = 0
    t1 = time.time()
    sample_index = 0
    correct_sample_index = 0
    with torch.no_grad():
        for i in range(epoch_size):
            # load train data
            images, labels, lengths = next(batch_iterator)
            start = 0
            targets = []
            for length in lengths:
                label = labels[start:start+length]
                targets.append(label)
                start += length
            targets = np.array([el.numpy() for el in targets])

            if args.cuda:
                images = Variable(images.cuda())
            else:
                images = Variable(images)

            # forward
            prebs = Net(images)
            # greedy decode
            prebs = prebs.cpu().detach().numpy()
            preb_labels = list()
            for i in range(prebs.shape[0]):
                preb = prebs[i, :, :]
                preb_label = list()
                for j in range(preb.shape[1]):
                    preb_label.append(np.argmax(preb[:, j], axis=0))
                no_repeat_blank_label = list()
                pre_c = preb_label[0]
                if pre_c != len(CHARS) - 1:
                    no_repeat_blank_label.append(pre_c)
                for c in preb_label: # dropout repeate label and blank label
                    if (pre_c == c) or (c == len(CHARS) - 1):
                        if c == len(CHARS) - 1:
                            pre_c = c
                        continue
                    no_repeat_blank_label.append(c)
                    pre_c = c
                preb_labels.append(no_repeat_blank_label)
            for i, label in enumerate(preb_labels):
                sample_index += 1
                target_text = decode_plate(targets[i])
                predict_text = decode_plate(label)
                is_match = target_text == predict_text
                if is_match:
                    correct_sample_index += 1
                if is_match and correct_sample_index % max(1, args.eval_log_interval) == 0:
                    logger.info(
                        "Eval sample %d | target=%s | predict=%s | match=%s",
                        sample_index,
                        decode_plate_for_log(targets[i]),
                        decode_plate_for_log(label),
                        is_match,
                    )
                if len(label) != len(targets[i]):
                    Tn_1 += 1
                    continue
                if (np.asarray(targets[i]) == np.asarray(label)).all():
                    Tp += 1
                else:
                    Tn_2 += 1

    if was_training:
        Net.train()

    Acc = Tp * 1.0 / (Tp + Tn_1 + Tn_2)
    logger.info("Test Accuracy: %.6f [%d:%d:%d:%d]", Acc, Tp, Tn_1, Tn_2, (Tp + Tn_1 + Tn_2))
    t2 = time.time()
    logger.info("Test Speed: %.6fs per sample over %d samples", (t2 - t1) / len(datasets), len(datasets))


if __name__ == "__main__":
    try:
        train()
    except KeyboardInterrupt:
        logger.warning("Program interrupted by user.")
        sys.exit(130)
