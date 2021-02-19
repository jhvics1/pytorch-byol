import os
import random
import argparse
import multiprocessing
import numpy as np
import torch
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from PIL import Image
from utils import Bar, config, mkdir_p, AverageMeter
from datetime import datetime
from tensorboardX import SummaryWriter

from byol_pytorch import BYOL

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
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')

# Device options
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
# Paths
parser.add_argument('-d', '--dataset', default='neu', type=str)
parser.add_argument('--image_folder', type=str, required=True,
                    help='path to your folder of images for self-supervised learning')
parser.add_argument('--board-path', '--bp', default='../board', type=str,
                    help='tensorboardx path')
parser.add_argument('--board-tag', '--tg', default='byol', type=str,
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
NUM_GPUS = 1
IMAGE_EXTS = ['.jpg', '.png', '.jpeg', '.bmp']
NUM_WORKERS = multiprocessing.cpu_count()

# task_time = datetime.now().isoformat()
# args.checkpoint = os.path.join(args.checkpoint, args.dataset, "{}-{}{:d}-bs{:d}-lr{:.5f}-{}".format(args.arch,
#                                                                                                     args.depth,
#                                                                                                     args.batch_size,
#                                                                                                     args.lr,
#                                                                                                     args.board_tag),
#                                task_time)
# if not os.path.isdir(args.checkpoint):
#     mkdir_p(args.checkpoint)
# config.save_config(args, os.path.join(args.checkpoint, "config.txt"))
#
# writer_train = SummaryWriter(
#     log_dir=os.path.join(args.board_path, args.dataset, "{}-{}{:d}-bs{:d}-lr{:.5f}-{}".format(args.arch,
#                                                                                               args.depth,
#                                                                                               args.batch_size,
#                                                                                               args.lr,
#                                                                                               args.board_tag),
#                          task_time, "train"))
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
    log_dir=os.path.join(args.board_path, args.dataset, output_name, args.task_time))

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


def expand_greyscale(t):
    return t.expand(3, -1, -1)


class ImagesDataset(Dataset):
    def __init__(self, folder, image_size):
        super().__init__()
        self.folder = folder
        self.paths = []

        for path in Path(f'{folder}').glob('**/*'):
            _, ext = os.path.splitext(path)
            if ext.lower() in IMAGE_EXTS:
                self.paths.append(path)

        print(f'{len(self.paths)} images found')

        self.transform = transforms.Compose([
            transforms.Resize(args.image_size),
            transforms.RandomSizedCrop((args.image_size, args.image_size)),
            transforms.ColorJitter(0.4, 0.4, 0.4),
            transforms.ToTensor(),
            # normalize
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        img = img.convert('RGB')
        return self.transform(img)


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

learner = BYOL(
    model,
    image_size=args.image_size,
    hidden_layer='avgpool',
    projection_size=256,
    projection_hidden_size=4096,
    moving_average_decay=0.99,
    use_momentum=False  # turn off momentum in the target encoder
)

opt = torch.optim.Adam(learner.parameters(), lr=args.lr)
ds = ImagesDataset(args.image_folder, args.image_size)
trainloader = DataLoader(ds, batch_size=args.batch_size, num_workers=NUM_WORKERS, shuffle=True)

losses = AverageMeter()

for epoch in range(args.epoch):
    bar = Bar('Processing', max=len(trainloader))
    for batch_idx, inputs in enumerate(trainloader):
        loss = learner(inputs.cuda())
        losses.update(loss.data.item(), inputs.size(0))

        opt.zero_grad()
        loss.backward()
        opt.step()
        # plot progress
        bar.suffix = 'Epoch {epoch} - ({batch}/{size}) | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} '.format(
            epoch=epoch,
            batch=batch_idx + 1,
            size=len(trainloader),
            total=bar.elapsed_td,
            eta=bar.eta_td,
            loss=loss.item(),
        )
        n_iter = epoch * len(trainloader) + batch_idx + 1
        writer_train.add_scalar('Train/loss', loss.data.item(), n_iter)
        bar.next()

    writer_train.add_scalar('Avg.loss', losses.avg, epoch)
    bar.finish()
# save your improved network
torch.save(model.state_dict(), os.path.join(args.checkpoint, 'byol.pt'))
