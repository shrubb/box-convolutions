import argparse
import os
import random
import shutil
import time

import torch
import torch.nn as nn
import torch.optim
import torch.utils.data

import tensorboardX

from models.ERFNet import ERFNet, BoxERFNet
from models.ENet import ENet, BoxENet, BoxOnlyENet

model_names = ['ENet', 'BoxENet', 'BoxOnlyENet', 'ERFNet', 'BoxERFNet']

parser = argparse.ArgumentParser(
    description='Simple Cityscapes semantic segmentation training')

parser.add_argument('data',
                    help='path to dataset')
parser.add_argument('--arch', '-a', default='BoxENet', choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) + ' (default: BoxENet)')
parser.add_argument('-j', '--workers', default=4, type=int,
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=1300, type=int,
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=12, type=int,
                    help='mini-batch size (default: 12)')

# optimizer
parser.add_argument('--optimizer', default='Adam', type=str,
                    help='optimizer type (default: Adam)')
parser.add_argument('--lr', '--learning-rate', default=None, type=float, # 1e-3
                    help='initial learning rate')
parser.add_argument('--lr-decay', default=0.45, type=float,
                    help='See --decay-steps')
parser.add_argument('--decay-steps', default='', type=str, # "600,780,880,960,1040,1120,1200,1260"
                    help='Comma-separated epoch numbers at which to multiply LR by --lr-decay')
parser.add_argument('--momentum', default=None, type=float, # 0.9
                    help='momentum')
parser.add_argument('--nesterov', default=None, type=bool, # True
                    help='use Nesterov momentum?')
parser.add_argument('--weight-decay', '--wd', default=None, type=float, # 2e-4
                    help='weight decay')

parser.add_argument('--run-name', default=time.ctime(time.time())[4:-8], type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--resume', action='store_true',
                    help='resume from latest checkpoint')
parser.add_argument('--start-epoch', default=0, type=int,
                    help='Manual epoch number (useful on restarts)')

parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='Do not train, only evaluate model on the validation set')

best_classIoU = 0


def main():
    global args, best_classIoU
    args = parser.parse_args()
    
    random.seed(666)
    torch.manual_seed(666)
  
    from datasets import Cityscapes
    
    train_dataset = Cityscapes(
        args.data, split='train', size=(1024, 512), augmented=True)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)
    train_loader.iter = iter(train_loader)

    val_dataset = Cityscapes(
        args.data, split='val', size=(1024, 512), augmented=False)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # create model and append `log_softmax` to it, to split this part of Criterion between GPUs
    print('Architecture:', args.arch)
    class ModelWithLogSoftmax(globals()[args.arch]):
        def forward(self, x):
            heatmap_raw = super().forward(x)
            return torch.nn.functional.log_softmax(heatmap_raw, dim=1)

    if 'Box' in args.arch:
        model = ModelWithLogSoftmax(
            n_classes=19, max_input_h=512, max_input_w=1024)
    else:
        model = ModelWithLogSoftmax(n_classes=19)
    model.cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.NLLLoss(weight=train_dataset.class_weights).cuda()

    optimizer_kwargs = {hyperparam: getattr(args, hyperparam) \
        for hyperparam in ('lr', 'weight_decay', 'momentum', 'nesterov') \
        if getattr(args, hyperparam) is not None}
    optimizer = torch.optim.__dict__[args.optimizer](model.parameters(), **optimizer_kwargs)

    # optionally resume from a checkpoint
    if args.resume:
        checkpoint_path = os.path.join('runs', args.run_name, 'model.pth')
        if os.path.isfile(checkpoint_path):
            print("=> loading checkpoint '{}'".format(checkpoint_path))
            checkpoint = torch.load(checkpoint_path)
            args.start_epoch = checkpoint['epoch']
            best_classIoU = checkpoint['best_classIoU']
 
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(checkpoint_path, checkpoint['epoch']))
        else:
            raise FileNotFoundError("=> no checkpoint found at '{}'".format(checkpoint_path))

    torch.backends.cudnn.benchmark = True
    model = torch.nn.DataParallel(model).cuda()

    board_writer = tensorboardX.SummaryWriter(os.path.join('runs', args.run_name))

    if args.evaluate:
        torch.backends.cudnn.deterministic = True
        val_loader.iter = iter(val_loader)
        validate(val_loader, model, criterion)
        return

    # warm up to save GPU memory
    #sample_input = torch.zeros(
    #    (12, 3, 512, 1024), device='cuda', dtype=torch.float32, requires_grad=True)
    #(model(sample_input).sum() * 0).backward()

    for epoch in range(args.start_epoch, args.epochs):
        print('Epoch', epoch+1)

        val_loader.iter = iter(val_loader)
        # train for one epoch
        train_metrics = train(train_loader, model, criterion, optimizer, epoch, board_writer)
        
        train_loader.iter = iter(train_loader)
        # evaluate on validation set
        val_metrics = validate(val_loader, model, criterion)

        # record metrics to tensorboard
        for tra,val,name in zip(train_metrics, val_metrics, ('Class IoU', 'Category IoU', 'Loss')):
            board_writer.add_scalars(name, {'Train': tra, 'Test': val}, epoch+1)
        
        # remember best score and save checkpoint
        classIoU = val_metrics[0]
        is_best = classIoU > best_classIoU
        best_classIoU = max(classIoU, best_classIoU)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.module.state_dict(),
            'best_classIoU': best_classIoU,
            'optimizer' : optimizer.state_dict(),
        }, is_best, os.path.join('runs', args.run_name, 'model.pth'))


def train(train_loader, model, criterion, optimizer, epoch, board_writer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()

    n_classes = criterion.weight.numel()-1
    confusion_matrix = torch.zeros((n_classes, n_classes), device='cuda', dtype=torch.long)

    # switch to train mode
    model.train()

    total_batches = len(train_loader)

    end = time.time()
    for i, (input, target) in enumerate(train_loader.iter):
        if i > total_batches: break

        # measure data loading time
        data_time.update(time.time() - end)
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        # compute output
        output = model(input)
        loss = criterion(output, target)
        # compute gradient and do SGD step
        global_iteration = epoch * total_batches + i
        adjust_learning_rate(optimizer, epoch, i, total_batches)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)

        with torch.no_grad():
            if i % 100 == 0: 
                """import numpy as np
                image = input[0].cpu().numpy().transpose((1,2,0)).copy()
                image -= image.min()
                image /= image.max()
                image *= 255.
                image = image.astype(np.uint8)
                output_map = output[0].max(0)[1].cpu().numpy()
                import imgaug
                segmap_display = imgaug.SegmentationMapOnImage(output_map, image.shape[:2], 20).draw_on_image(image)
                segmap_display = segmap_display.transpose((2,0,1)).copy()

                board_writer.add_image('Example segmentation', segmap_display, global_iteration)"""

                # update confusion matrix to compute IoU
                output = output.max(1)[1].view(-1).to(confusion_matrix)
                target = target.view(-1).to(output)

                confusion_matrix_update = \
                    torch.bincount(target*(n_classes+1) + output, minlength=(n_classes+1)**2)
                confusion_matrix += \
                    confusion_matrix_update.view(n_classes+1, n_classes+1)[1:,1:].to(confusion_matrix)

                loss_meter.update(loss.item(), input.size(0))

                board_writer.add_scalar('Learning rate',
                    next(iter(optimizer.param_groups))['lr'], global_iteration)
                board_writer.add_scalars('Time',
                    {'Total batch time': batch_time.val,
                     'Data loading time': data_time.val}, global_iteration)
                board_writer.add_scalar('Online batch loss', loss_meter.val, global_iteration)

        end = time.time()

    classIoU, categIoU = compute_IoU(confusion_matrix.cpu())
    return classIoU, categIoU, loss_meter.avg


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    loss_meter = AverageMeter()

    # switch to evaluate mode
    model.eval()
    
    n_classes = criterion.weight.numel()-1
    confusion_matrix = torch.zeros((n_classes, n_classes), device='cuda', dtype=torch.long)

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader.iter):
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # record loss
            loss_meter.update(loss.item(), input.size(0))

            # update confusion matrix to compute IoU
            output = output.max(1)[1].view(-1)
            target = target.view(-1).to(output)

            confusion_matrix_update = \
                torch.bincount(target*(n_classes+1) + output, minlength=(n_classes+1)**2)
            confusion_matrix += \
                confusion_matrix_update.view(n_classes+1, n_classes+1)[1:,1:].to(confusion_matrix)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    classIoU, categoryIoU = compute_IoU(confusion_matrix.cpu())

    print('Class IoU:', classIoU)
    print('Category IoU:', categoryIoU)
    return classIoU, categoryIoU, loss_meter.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth'):
    assert filename.endswith('.pth')
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename[:-4] + '_best.pth')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, epoch_iteration=None, iters_per_epoch=None):
    step_epochs = torch.tensor(list(map(int, args.decay_steps.split(','))))
    lr = args.lr * (args.lr_decay ** (epoch > step_epochs).sum().item())

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def compute_IoU(confusion_matrix):
    n_classes = confusion_matrix.shape[0]

    with torch.no_grad():
        class_categories = [
            [0, 1],                   # flat
            [2, 3, 4],                # construction
            [5, 6, 7],                # object
            [8, 9],                   # nature
            [10],                     # sky
            [11, 12],                 # human
            [13, 14, 15, 16, 17, 18], # vehicle
        ]

        classIoU = torch.empty(n_classes, dtype=torch.float32)
        for class_idx in range(n_classes):
            TP = confusion_matrix[class_idx, class_idx].item()
            FN = confusion_matrix[class_idx, :].sum().item() - TP
            FP = confusion_matrix[:, class_idx].sum().item() - TP

            classIoU[class_idx] = TP / max(TP + FP + FN, 1)

        categoryIoU = torch.empty(len(class_categories), dtype=torch.float32)
        for category_idx, category in enumerate(class_categories):
            TP = 0
            for class_idx in category:
                TP += confusion_matrix[class_idx, category].sum().item()

            FN, FP = -TP, -TP
            for class_idx in category:
                FN += confusion_matrix[class_idx, :].sum().item()
                FP += confusion_matrix[:, class_idx].sum().item()

            categoryIoU[category_idx] = TP / max(TP + FP + FN, 1)

        return classIoU.mean(), categoryIoU.mean()


if __name__ == '__main__':
    main()
