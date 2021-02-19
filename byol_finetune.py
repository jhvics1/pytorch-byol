import os
import random
import argparse
import multiprocessing
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from torchvision import models, transforms
import torchvision.datasets as datasets
from utils import Bar, config, mkdir_p, AverageMeter, accuracy
from datetime import datetime
from tensorboardX import SummaryWriter


def train(model, criterion, opt, softmax, bar, epoch, loader, losses, top1, top5, writer):
    # for training
    model.train()
    for batch_idx, (inputs, labels) in enumerate(loader):
        outputs = model(inputs.cuda())
        outputs = softmax(outputs)
        loss = criterion(outputs, labels.cuda())

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, labels.cuda().data, topk=(1, 5))
        losses.update(loss.data.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        opt.zero_grad()
        loss.backward()
        opt.step()
        # plot progress
        bar.suffix = 'Epoch {epoch} - Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f} | ({batch}/{size})'.format(
            epoch=epoch,
            batch=batch_idx + 1,
            size=len(loader),
            total=bar.elapsed_td,
            eta=bar.eta_td,
            loss=loss.item(),
            top1=top1.avg,
            top5=top5.avg,
        )
        n_iter = epoch * len(loader) + batch_idx + 1
        writer.add_scalar('Train/loss', loss.data.item(), n_iter)
        writer.add_scalar('Train/top1', prec1.data.item(), n_iter)
        writer.add_scalar('Train/top5', prec5.data.item(), n_iter)
        bar.next()
    writer.add_scalar('Avg.loss', losses.avg, epoch)
    writer.add_scalar('Avg.top1', top1.avg, epoch)
    writer.add_scalar('Avg.top5', top5.avg, epoch)
    bar.finish()


def test(model, criterion, softmax, bar, epoch, loader, losses, top1, top5, writer):
    model.eval()
    for batch_idx, (inputs, labels) in enumerate(loader):
        outputs = model(inputs.cuda())
        outputs = softmax(outputs)
        loss = criterion(outputs, labels.cuda())

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, labels.cuda().data, topk=(1, 5))
        losses.update(loss.data.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        # plot progress
        bar.suffix = 'Epoch {epoch} - Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f} | ({batch}/{size})'.format(
            epoch=epoch,
            batch=batch_idx + 1,
            size=len(loader),
            total=bar.elapsed_td,
            eta=bar.eta_td,
            loss=loss.item(),
            top1=top1.avg,
            top5=top5.avg,
        )
        n_iter = epoch * len(loader) + batch_idx + 1
        writer.add_scalar('Test/loss', loss.data.item(), n_iter)
        writer.add_scalar('Test/top1', prec1.data.item(), n_iter)
        writer.add_scalar('Test/top5', prec5.data.item(), n_iter)
        bar.next()
    writer.add_scalar('Avg.loss', losses.avg, epoch)
    writer.add_scalar('Avg.top1', top1.avg, epoch)
    writer.add_scalar('Avg.top5', top5.avg, epoch)
    bar.finish()


def main():
    global parser, args, args
    # arguments
    parser = argparse.ArgumentParser(description='byol-lightning-test')
    # Architecture & hyper-parameter
    parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet',
                        help='model architecture: | [resnet, ...] (default: resnet18)')
    parser.add_argument('--depth', type=int, default=18, help='Model depth.')
    parser.add_argument('-c', '--checkpoint', default='../checkpoints', type=str, metavar='PATH',
                        help='path to save checkpoint (default: checkpoint)')
    parser.add_argument('--epoch', type=int, default=100, help='Epoch')
    parser.add_argument('--batch-size', type=int, default=32, help='Epoch')
    parser.add_argument('--lr', '--learning-rate', default=1, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--num-classes', type=int, default=100, help='Epoch')
    parser.add_argument('--from-scratch', action='store_true', default=False,
                        help='use pre-trained model')
    parser.add_argument('--tune-all', action='store_true', default=False,
                        help='use pre-trained model')

    # Device options
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    parser.add_argument('--gpu-id', default='0', type=str,
                        help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--model-path', '--mp', type=str,
                        help='byol trained model path')
    # Paths
    parser.add_argument('-d', '--dataset', default='neu', type=str)
    parser.add_argument('--image_folder', type=str, required=True,
                        help='path to your folder of images for self-supervised learning')
    parser.add_argument('--board-path', '--bp', default='../board', type=str,
                        help='tensorboardx path')
    parser.add_argument('--board-tag', '--tg', default='fine-tuned', type=str,
                        help='tensorboardx writer tag')
    args = parser.parse_args()

    # Use CUDA
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    use_cuda = torch.cuda.is_available()
    # Torch Seed
    # Random seed
    if args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
    # Random Lib Seed
    random.seed(args.manualSeed)
    # Numpy Seed
    np.random.seed(args.manualSeed)
    if use_cuda:
        torch.cuda.manual_seed_all(args.manualSeed)

    # constants
    args.image_size = 256
    args.workers = multiprocessing.cpu_count()

    args.task_time = datetime.now().isoformat()
    output_name = "{}{:d}-bs{:d}-lr{:.5f}-{}".format(args.arch,
                                                     args.depth,
                                                     args.batch_size,
                                                     args.lr,
                                                     args.board_tag)
    args.checkpoint = os.path.join(args.checkpoint, args.dataset, output_name, args.task_time)
    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)
    config.save_config(args, os.path.join(args.checkpoint, "config.txt"))

    writer_train = SummaryWriter(
        log_dir=os.path.join(args.board_path, args.dataset, output_name, args.task_time, "train"))
    writer_test = SummaryWriter(
        log_dir=os.path.join(args.board_path, args.dataset, output_name, args.task_time, "test"))

    if args.arch is "resnet":
        if args.depth == 18:
            model = models.resnet18(pretrained=False).cuda()
        elif args.depth == 34:
            model = models.resnet34(pretrained=False).cuda()
        elif args.depth == 50:
            model = models.resnet50(pretrained=False).cuda()
        elif args.depth == 101:
            model = models.resnet101(pretrained=False).cuda()
        else:
            assert ("Not supported Depth")

    if not args.from_scratch:
        checkpoint = torch.load(args.model_path)
        model.load_state_dict(checkpoint)
    print("\t==> Fine tune full layers? : {}".format(str(args.tune_all)))
    # Simple manual fine tuning logic
    # if full == False, only last layer will be fine tuned~!!
    if not args.tune_all:
        params = model.parameters()
        for param in params:
            param.requires_grad = False
    model.num_classes = args.num_classes
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, args.num_classes)

    model = torch.nn.DataParallel(model).cuda()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss().cuda()
    softmax = nn.Softmax(1).cuda()

    # Data loading code
    traindir = os.path.join(args.image_folder, 'train')
    testdir = os.path.join(args.image_folder, 'test')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    trainloader = torch.utils.data.DataLoader(
        datasets.ImageFolder(traindir, transforms.Compose([
            transforms.Resize(args.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(0.4, 0.4, 0.4),
            transforms.ToTensor(),
            # normalize,
        ])),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)
    testloader = torch.utils.data.DataLoader(
        datasets.ImageFolder(testdir, transforms.Compose([
            transforms.Resize(args.image_size),
            transforms.ToTensor(),
            # normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    losses_train = AverageMeter()
    top1_train = AverageMeter()
    top5_train = AverageMeter()
    losses_test = AverageMeter()
    top1_test = AverageMeter()
    top5_test = AverageMeter()

    for epoch in range(args.epoch):
        bar_train = Bar('Processing', max=len(trainloader))
        bar_test = Bar('Processing', max=len(testloader))
        train(model, criterion, opt, softmax, bar_train, epoch, trainloader, losses_train, top1_train, top5_train,
              writer_train)
        test(model, criterion, softmax, bar_test, epoch, testloader, losses_test, top1_test, top5_test,
             writer_test)
    # save your improved network
    torch.save(model.state_dict(), os.path.join(args.checkpoint, 'byol-finetune.pt'))


if __name__ == '__main__':
    main()
