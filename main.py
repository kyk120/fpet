import torch
from torch import nn
from torch.optim import AdamW
from torch.nn import functional as F
from tqdm import tqdm
import timm
from timm.models import create_model
from timm.scheduler.cosine_lr import CosineLRScheduler
from argparse import ArgumentParser
from vtab import *
from utils import save, load, load_config, set_seed, QLinear, AverageMeter, log
import adaptformer
import lora
import fpet
from torch.utils.tensorboard import SummaryWriter
import time
import datetime


def train(args, model, dl, opt, scheduler, epoch, log_writer):
    model.train()
    model = model.cuda()
    pbar = tqdm(range(epoch))
    start_train = time.time()
    for ep in pbar:
        model.train()
        model = model.cuda()
        for i, batch in enumerate(dl):
            x, y = batch[0].cuda(), batch[1].cuda()
            out = model(x)
            loss = F.cross_entropy(out, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        if scheduler is not None:
            scheduler.step(ep)
        if ep % 10 == 9:
            end_train = time.time()
            start_test = time.time()
            acc = test(vit, test_dl)
            end_test = time.time()
            train_time = (end_train-start_train)/(len(train_dl)*10)
            test_time = (end_test-start_test)/len(test_dl)
            log(args, acc, train_time, test_time, loss.item(), ep)
            log_writer.add_scalar('acc', acc, ep)
            log_writer.add_scalar('train_time', train_time, ep)
            log_writer.add_scalar('test_time', test_time, ep)
            log_writer.add_scalar('loss', loss.item(), ep)
            log_writer.flush()
            if acc > args.best_acc:
                args.best_acc = acc
                args.best_ep = ep
                save(args, model)
            pbar.set_description('best_acc ' + str(args.best_acc))
            start_train = time.time()

    model = model.cpu()
    return model


@torch.no_grad()
def test(model, dl):
    model.eval()
    acc = AverageMeter()
    model = model.cuda()
    for batch in tqdm(dl):
        x, y = batch[0].cuda(), batch[1].cuda()
        out = model(x).data
        acc.update(out, y)
    return acc.result().item()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--bit', type=int, default=1, choices=[1, 2, 4, 8, 32])
    parser.add_argument('--dim', type=int, default=32)
    parser.add_argument('--scale', type=float, default=1)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--wd', type=float, default=1e-4)
    parser.add_argument('--model', type=str, default='vit_base_patch16_224_in21k')
    parser.add_argument('--dataset', type=str, default='cifar')
    parser.add_argument('--method', type=str, default='adaptformer',
                        choices=['adaptformer', 'adaptformer-bihead', 'lora', 'lora-bihead'])
    parser.add_argument('--eval', action='store_true', default=False)
    parser.add_argument('--config_path', type=str, default='.')
    parser.add_argument('--model_path', type=str, default='.')
    parser.add_argument('--load_config', action='store_true', default=False)
    parser.add_argument('--log_path', type=str, default='.')
    parser.add_argument('--r_layer', default=False, type=int)
    args = parser.parse_args()
    print(args)
    if args.eval or args.load_config:
        load_config(args)
    set_seed(args.seed)
    args.best_acc = 0
    args.best_ep = 0
    vit = create_model(args.model, checkpoint_path='ViT-B_16.npz', drop_path_rate=0.1)
    train_dl, test_dl = get_data(args.dataset, normalize=False)
    
    save_folder = os.path.join(args.model_path, args.dataset)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    log_writer = SummaryWriter(log_dir=os.path.join(args.model_path, args.dataset))

    if args.method == 'adaptformer':
        adaptformer.set_adapter(vit, dim=args.dim, s=args.scale, bit=args.bit)
        vit.reset_classifier(get_classes_num(args.dataset))
    elif args.method == 'lora':
        lora.set_adapter(vit, dim=args.dim, s=args.scale, bit=args.bit)
        vit.reset_classifier(get_classes_num(args.dataset))
    
    if args.r_layer != False:
        fpet.apply_fpet.apply_patch(vit, method = args.method, r_layer = args.r_layer)
        print(f'FPET applied!')

    if not args.eval:
        import shutil
        shutil.copytree('./configs', os.path.join(save_folder, 'codes/configs'))
        shutil.copytree('./fpet', os.path.join(save_folder, 'codes/fpet'))
        shutil.copy('adaptformer.py', os.path.join(save_folder, 'codes/adaptformer.py'))
        shutil.copy('lora.py', os.path.join(save_folder, 'codes/lora.py'))
        shutil.copy('main.py', os.path.join(save_folder, 'codes/main.py'))
        shutil.copy('utils.py', os.path.join(save_folder, 'codes/utils.py'))
        shutil.copy('vtab.py', os.path.join(save_folder, 'codes/vtab.py'))
        
        import sys
        with open(os.path.join(save_folder, 'params.sh'), 'w+') as out:
            sys.argv[0] = os.path.join(os.getcwd(), sys.argv[0])
            out.write('#!/bin/bash\n')
            out.write('python3 ')
            out.write(' '.join(sys.argv))
            out.write('\n')

        start_total = time.time()

    if not args.eval:
        trainable = []
        trainable_n = []
        for n, p in vit.named_parameters():
            if ('adapter' in n or 'head' in n) and p.requires_grad:
                trainable.append(p)
                trainable_n.append(n)
            else:
                p.requires_grad = False
        print(f'{trainable_n=}')
        opt = AdamW(trainable, lr=args.lr, weight_decay=args.wd)
        scheduler = CosineLRScheduler(opt, t_initial=100,
                                      warmup_t=10, lr_min=1e-5, warmup_lr_init=1e-6, decay_rate=0.1)
        vit = train(args, vit, train_dl, opt, scheduler, epoch=100, log_writer=log_writer)
        
        end_total = time.time()
        total_time = end_total - start_total
        log_stats = {"best_epoch": args.best_ep, "best_acc": args.best_acc, "total_train_time": str(datetime.timedelta(seconds=int(total_time)))}
        log(args, 0, 0, 0, 0, 0, log_stats=log_stats)

    else:
        load(args, vit)
        args.best_acc = test(vit, test_dl)

    print('best_acc:', args.best_acc)
