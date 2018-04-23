import argparse
import os
import time
import shutil
import torch
import torchvision
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim

from CenterLoss import CenterLoss


from torch.nn.utils import clip_grad_norm

from dataset import TSNDataSet
from seqvlad_models import SeqVLAD_with_centerloss
from transforms import *
from opts import parser

from collections import OrderedDict

best_prec1 = 0


def main():
    global args, best_prec1
    args = parser.parse_args()

    if args.dataset == 'ucf101':
        num_class = 101
    elif args.dataset == 'hmdb51':
        num_class = 51
    elif args.dataset == 'kinetics':
        num_class = 400
    else:
        raise ValueError('Unknown dataset '+args.dataset)




    model = SeqVLAD_with_centerloss(num_class, args.num_centers, args.modality,
                args.timesteps, args.redu_dim,
                with_relu=args.with_relu,
                base_model=args.arch,
                activation=args.activation,
                seqvlad_type=args.seqvlad_type,
                dropout=args.dropout, partial_bn=not args.no_partialbn)
    # print(model)

    crop_size = model.crop_size
    scale_size = model.scale_size
    input_mean = model.input_mean
    input_std = model.input_std
    policies = model.get_optim_policies()
    if args.two_steps is not None:
        print('two step training ')
        sub_policies = model.get_sub_optim_policies()

    train_augmentation = model.get_augmentation()



    model = torch.nn.DataParallel(model, device_ids=args.gpus).cuda()




    if args.resume:
        if os.path.isfile(args.resume):
            print(("=> loading checkpoint '{}'".format(args.resume)))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            

            model_dict = model.state_dict()

            if args.resume_type == 'tsn':
                args.start_epoch = 0

                ## exclude certain module
                pretrained_dict = checkpoint['state_dict']

                excluded_modules = ['module.new_fc', 'module.base_model.global_pool']
                res_state_dict =  filter_excluded_module(pretrained_dict, excluded_modules)
                model_dict.update(res_state_dict) 

            elif args.resume_type =='same':
                args.start_epoch = 0
                
                pretrained_dict = checkpoint['state_dict']
                res_state_dict = init_from_tsn_model(model_dict, pretrained_dict)
                model_dict.update(res_state_dict) 
            else:
                print('==> resume_type must be one of same/tsn')
                exit()

            model.load_state_dict(model_dict)

            print(("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.evaluate, checkpoint['epoch'])))
        else:
            print(("=> no checkpoint found at '{}'".format(args.resume)))

    cudnn.benchmark = True

    # Data loading code
    if args.modality != 'RGBDiff':
        normalize = GroupNormalize(input_mean, input_std)
    else:
        normalize = IdentityTransform()

    if args.modality == 'RGB':
        data_length = 1
    elif args.modality in ['Flow', 'RGBDiff']:
        data_length = 5

    train_loader = torch.utils.data.DataLoader(
        TSNDataSet(args.sources, args.train_list, timesteps=args.timesteps,
                   new_length=data_length,
                   modality=args.modality,
                   sampling_method=args.sampling_method,
                   reverse=args.reverse,
                   image_tmpl="image_{:05d}.jpg" if args.modality in ["RGB", "RGBDiff"] else args.flow_prefix+"{}_{:04d}.jpg",
                   transform=torchvision.transforms.Compose([
                       train_augmentation,
                       Stack(roll=args.arch == 'BNInception'),
                       ToTorchFormatTensor(div=args.arch != 'BNInception'),
                       normalize,
                   ])),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        TSNDataSet(args.sources, args.val_list, timesteps=args.timesteps,
                   new_length=data_length,
                   modality=args.modality,
                   image_tmpl="image_{:05d}.jpg" if args.modality in ["RGB", "RGBDiff"] else args.flow_prefix+"{}_{:04d}.jpg",
                   random_shift=False,
                   transform=torchvision.transforms.Compose([
                       GroupScale(int(scale_size)),
                       GroupCenterCrop(crop_size),
                       Stack(roll=args.arch == 'BNInception'),
                       ToTorchFormatTensor(div=args.arch != 'BNInception'),
                       normalize,
                   ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # define loss function (criterion) and optimizer
    if args.loss_type == 'nll':
        nll = torch.nn.CrossEntropyLoss().cuda()
        centerloss = CenterLoss(num_class, args.num_centers*args.redu_dim).cuda()
        criterion = [nll, centerloss]
    else:
        raise ValueError("Unknown loss type")

    for group in policies:
        print(('group: {} has {} params, lr_mult: {}, decay_mult: {}'.format(
            group['name'], len(group['params']), group['lr_mult'], group['decay_mult'])))


    if args.two_steps is not None:
        for group in sub_policies:
            print(('group: {} has {} params, lr_mult: {}, decay_mult: {}'.format(
                group['name'], len(group['params']), group['lr_mult'], group['decay_mult'])))

    if args.optim == 'SGD':
        optimizer = torch.optim.SGD(policies, args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        optimizer4center = torch.optim.SGD(centerloss.parameters(), lr=0.5)
        if args.two_steps is not None:
            sub_optimizer = torch.optim.SGD(sub_policies, args.lr, momentum=args.momentum, weight_decay=args.weight_decay)



    # elif args.optim == 'Adam':
    #     print('use Adam optimizer ... ...')
    #     optimizer = torch.optim.Adam(policies, args.lr, weight_decay=args.weight_decay)
    #     optimizer4center = torch.optim.SGD(centerloss.parameters(), lr =args.lr)

    #     if args.two_steps is not None:
    #         sub_optimizer = torch.optim.Adam(sub_policies, args.lr, weight_decay=args.weight_decay)


    else:
        print('optimzer: {} is not implimented, please use SGD or Adam'.format(args.optim))
        exit()
    if args.evaluate:
        validate(val_loader, model, criterion, args.lossweight, 0)
        return
    

    for epoch in range(args.start_epoch, args.epochs):
        if args.two_steps is not None and epoch < args.two_steps:
            adjust_learning_rate(sub_optimizer, epoch, args.lr_steps)

            train(train_loader, model, criterion, sub_optimizer, optimizer4center, args.lossweight, epoch)
        else:
            adjust_learning_rate(optimizer, epoch, args.lr_steps)

            train(train_loader, model, criterion, optimizer, optimizer4center, args.lossweight, epoch)

        # print(dir(model))
        # print(type(model.parameters.global_pool))
        # print(model['global_pool'])
        # train for one epoch
        

        # evaluate on validation set
        if (epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1:
            prec1 = validate(val_loader, model, criterion, args.lossweight, (epoch + 1) * len(train_loader))

            # remember best prec@1 and save checkpoint
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
            }, is_best)


def init_from_tsn_model(model_dict, tsn_model_state):
    res_state_dict = OrderedDict()

    # 1. filter out unnecessary keys
    res_state_dict = {k: v for k, v in tsn_model_state.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    # model_dict.update(pretrained_dict) 

    return res_state_dict

def filter_excluded_module(state_dict, excluded_modules=None):
    # if excluded_module is not None:
    #     for key
    if excluded_modules is None:
        return state_dict

    res_state_dict = OrderedDict()

    for k, v in state_dict.items():

        is_skip = False
        for em in excluded_modules:
            if k.startswith(em):
                is_skip = True
                break
        if not is_skip:
            res_state_dict[k] = v
        else:
            print('%s module is skipped' %(k))
            # print k, v
    return res_state_dict

def train(train_loader, model, criterion, optimizer, optimizer4center, lossweight, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    nll_losses = AverageMeter()
    center_losses = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    if args.no_partialbn:
        model.module.partialBN(False)
    else:
        model.module.partialBN(True)

    # switch to train mode
    model.train()
    end = time.time()
    for i, (input, target) in enumerate(train_loader):


        # print('##### i:', i)
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda(async=True)
        # print('input size ====>', input.size())
        # print('input size', input.size())
        # input = input.view(-1,3,224,224)
        # print('input size ====>', input.size())
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        vlad_output, output = model(input_var)
        # loss = criterion(output, target_var)
        nll_loss = criterion[0](output, target_var)
        center_loss = lossweight * criterion[1](target_var, vlad_output)
        loss =  nll_loss + center_loss


        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1,5))

        nll_losses.update(nll_loss.data[0], input.size(0))
        center_losses.update(center_loss.data[0], input.size(0))

        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))


        # compute gradient and do SGD step
        optimizer.zero_grad()
        optimizer4center.zero_grad()
        loss.backward()

        if args.clip_gradient is not None:
            total_norm = clip_grad_norm(model.parameters(), args.clip_gradient)
            if total_norm > args.clip_gradient:
                print("clipping gradient: {} with coef {}".format(total_norm, args.clip_gradient / total_norm))


        optimizer.step()
        optimizer4center.step()
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print(('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Nll_loss {nll_loss.val:.4f} ({nll_loss.avg:.4f})\t'
                  'Center_loss {center_loss.val:.4f} ({center_loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, nll_loss=nll_losses, center_loss=center_losses, top1=top1, top5=top5, lr=optimizer.param_groups[-1]['lr'])))


def validate(val_loader, model, criterion, lossweight, iter, logger=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    nll_losses = AverageMeter()
    center_losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        vlad_output, output = model(input_var)

        nll_loss = criterion[0](output, target_var)
        center_loss = lossweight * criterion[1](target_var, vlad_output)
        loss =  nll_loss + center_loss


        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1,5))
        nll_losses.update(nll_loss.data[0], input.size(0))
        center_losses.update(center_loss.data[0], input.size(0))


        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print(('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Nll_loss {nll_loss.val:.4f} ({nll_loss.avg:.4f})\t'
                  'Center_loss {center_loss.val:.4f} ({center_loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses, nll_loss=nll_losses, center_loss=center_losses,
                   top1=top1, top5=top5)))

    print(('Testing Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f} Nll_loss {nll_loss.avg:.4f} Center_loss {center_loss.avg:.4f}'
          .format(top1=top1, top5=top5, loss=losses, nll_loss=nll_losses, center_loss=center_losses)))

    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    filename = '_'.join((args.snapshot_pref, args.modality.lower(), filename))
    torch.save(state, filename)
    if is_best:
        best_name = '_'.join((args.snapshot_pref, args.modality.lower(), 'model_best.pth.tar'))
        shutil.copyfile(filename, best_name)


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


def adjust_learning_rate(optimizer, epoch, lr_steps):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    decay = 0.1 ** (sum(epoch >= np.array(lr_steps)))
    lr = args.lr * decay
    decay = args.weight_decay
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']
        param_group['weight_decay'] = decay * param_group['decay_mult']


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
