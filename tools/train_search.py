import time
import copy
import datetime
import os
import sys

cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)

import random
import numpy as np
import logging
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F

from torchvision import transforms
from segmentron.data.dataloader import get_segmentation_dataset
from segmentron.models.nas_model import get_supernet
from segmentron.solver.loss import get_segmentation_loss
from segmentron.solver.optimizer import get_optimizer, get_arch_optimizer
from segmentron.solver.lr_scheduler import get_scheduler, get_arch_scheduler
from segmentron.utils.distributed import *
from segmentron.utils.score import SegmentationMetric
from segmentron.utils.filesystem import save_checkpoint
from segmentron.utils.options import parse_args
from segmentron.utils.default_setup import default_setup
from segmentron.utils.visualize import show_flops_params
from segmentron.config import cfg
from tensorboardX import SummaryWriter
from segmentron.core.utils.visualization import process_image, save_heatmap, save_connectivity

class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)

        # image transform
        input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(cfg.DATASET.MEAN, cfg.DATASET.STD),
        ])
        # dataset and dataloader
        data_kwargs = {'transform': input_transform, 'base_size': cfg.TRAIN.BASE_SIZE,
                       'crop_size': cfg.TRAIN.CROP_SIZE}
        train_dataset = get_segmentation_dataset(cfg.DATASET.NAME, split='train', mode='train', **data_kwargs)
        search_dataset = get_segmentation_dataset(cfg.DATASET.NAME, split='val', mode='val', **data_kwargs)
        val_dataset = get_segmentation_dataset(cfg.DATASET.NAME, split='val', mode=cfg.DATASET.MODE, **data_kwargs)
        self.iters_per_epoch = len(train_dataset) // (args.num_gpus * cfg.TRAIN.BATCH_SIZE)
        self.max_iters = cfg.TRAIN.EPOCHS * self.iters_per_epoch

        train_sampler = make_data_sampler(train_dataset, shuffle=True, distributed=args.distributed)
        train_batch_sampler = make_batch_data_sampler(train_sampler, cfg.TRAIN.BATCH_SIZE, self.max_iters, drop_last=True)
        search_sampler = make_data_sampler(search_dataset, shuffle=True, distributed=args.distributed)
        search_batch_sampler = make_batch_data_sampler(search_sampler, cfg.TRAIN.BATCH_SIZE, self.max_iters, drop_last=True)
        val_sampler = make_data_sampler(val_dataset, shuffle=True, distributed=args.distributed)
        val_batch_sampler = make_batch_data_sampler(val_sampler, cfg.TEST.BATCH_SIZE, drop_last=True)

        self.train_loader = data.DataLoader(dataset=train_dataset,
                                            batch_sampler=train_batch_sampler,
                                            num_workers=cfg.DATASET.WORKERS,
                                            pin_memory=True)
        self.search_loader = data.DataLoader(dataset=search_dataset,
                                            batch_sampler=search_batch_sampler,
                                            num_workers=cfg.DATASET.WORKERS,
                                            pin_memory=True)
        self.val_loader = data.DataLoader(dataset=val_dataset,
                                          batch_sampler=val_batch_sampler,
                                          num_workers=cfg.DATASET.WORKERS,
                                          pin_memory=True)

        # create criterion
        self.criterion = get_segmentation_loss(cfg.MODEL.MODEL_NAME, use_ohem=cfg.SOLVER.OHEM,
                                               aux=cfg.SOLVER.AUX, aux_weight=cfg.SOLVER.AUX_WEIGHT,
                                               ignore_index=cfg.DATASET.IGNORE_INDEX).to(self.device)
        
        # create network
        self.model = get_supernet(self.criterion).to(self.device)
        
        # print params and flops
        if get_rank() == 0:
            try:
                show_flops_params(copy.deepcopy(self.model), args.device)
            except Exception as e:
                logging.warning('get flops and params error: {}'.format(e))

        if cfg.MODEL.BN_TYPE not in ['BN']:
            logging.info('Batch norm type is {}, convert_sync_batchnorm is not effective'.format(cfg.MODEL.BN_TYPE))
        elif args.distributed and cfg.TRAIN.SYNC_BATCH_NORM:
            self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            logging.info('SyncBatchNorm is effective!')
        else:
            logging.info('Not use SyncBatchNorm!')

        # optimizer, for model just includes encoder, decoder(head and auxlayer).
        self.optimizer = get_optimizer(self.model)
        self.arch_optimizer = get_arch_optimizer(self.model)

        # lr scheduling
        self.lr_scheduler = get_scheduler(self.optimizer, max_iters=self.max_iters,
                                          iters_per_epoch=self.iters_per_epoch)
        self.arch_lr_scheduler = get_arch_scheduler(self.arch_optimizer, max_iters=self.max_iters,
                                          iters_per_epoch=self.iters_per_epoch)

        # resume checkpoint if needed
        self.start_epoch = 0
        if args.resume and os.path.isfile(args.resume):
            name, ext = os.path.splitext(args.resume)
            assert ext == '.pkl' or '.pth', 'Sorry only .pth and .pkl files supported.'
            logging.info('Resuming training, loading {}...'.format(args.resume))
            resume_sate = torch.load(args.resume)
            self.model.load_state_dict(resume_sate['state_dict'])
            self.start_epoch = resume_sate['epoch']
            logging.info('resume train from epoch: {}'.format(self.start_epoch))
            if resume_sate['optimizer'] is not None and resume_sate['lr_scheduler'] is not None:
                logging.info('resume optimizer and lr scheduler from resume state..')
                self.optimizer.load_state_dict(resume_sate['optimizer'])
                self.lr_scheduler.load_state_dict(resume_sate['lr_scheduler'])

        if args.distributed:
            self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[args.local_rank],
                                                             output_device=args.local_rank,
                                                             find_unused_parameters=True)

        # evaluation metrics
        self.metric = SegmentationMetric(train_dataset.num_class, args.distributed)
        self.best_pred = 0.0


    def train(self):
        self.save_to_disk = get_rank() == 0
        epochs, max_iters, iters_per_epoch = cfg.TRAIN.EPOCHS, self.max_iters, self.iters_per_epoch
        log_per_iters, val_per_iters = self.args.log_iter, self.args.val_epoch * self.iters_per_epoch

        start_time = time.time()
        logging.info('Start training, Total Epochs: {:d} = Total Iterations {:d}'.format(epochs, max_iters))

        self.model.train()
        iteration = self.start_epoch * iters_per_epoch if self.start_epoch > 0 else 0
        val_iter = iter(self.search_loader)

        for (images, targets, _) in self.train_loader:
            epoch = iteration // iters_per_epoch + 1
            iteration += 1

            # arch search
            if epoch > cfg.ARCH.SEARCH_EPOCH:
                val_batch = next(val_iter, None)
                if val_batch is None:  # val_iter has reached its end
                    # val_sampler.set_epoch(epoch) ???
                    val_iter = iter(self.search_loader)
                    val_batch = next(val_iter)
                image_search, targets_search, fileName_search = val_batch
                image_search = image_search.to(self.device)
                targets_search = targets_search.to(self.device)

                self.model.arch_train()
                assert self.model.arch_training
                self.arch_optimizer.zero_grad()
                arch_loss_dict = self.model.loss(image_search, targets_search)
                losses = sum(loss for loss in arch_loss_dict.values())

                loss_dict_reduced = reduce_loss_dict(arch_loss_dict)
                arch_losses_reduced = sum(loss for loss in loss_dict_reduced.values()).item()

                self.arch_optimizer.zero_grad()
                losses.backward()
                self.arch_optimizer.step()
                # self.arch_lr_scheduler.step()
                self.model.arch_eval()
            else:
                arch_losses_reduced = 0

            assert not self.model.arch_training
            images = images.to(self.device)
            targets = targets.to(self.device)

            outputs = self.model(images)
            loss_dict = self.criterion(outputs, targets)

            losses = sum(loss for loss in loss_dict.values())

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = reduce_loss_dict(loss_dict)
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())

            self.optimizer.zero_grad()
            losses.backward()
            self.optimizer.step()
            if cfg.ARCH.SEARCHSPACE == 'GeneralizedFastSCNN':
                self.model.step()
            self.lr_scheduler.step()
            if cfg.ARCH.OPTIMIZER == 'sgd':
                self.arch_lr_scheduler.step()

            eta_seconds = ((time.time() - start_time) / iteration) * (max_iters - iteration)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

            if iteration % log_per_iters == 0 and self.save_to_disk:
                logging.info(
                    "Epoch: {:d}/{:d} || Iters: {:d}/{:d} || Lr: {:.6f} || "
                    "Loss: {:.4f}/{:.4f} || Cost Time: {} || Estimated Time: {}".format(
                        epoch, epochs, iteration % iters_per_epoch, iters_per_epoch,
                        self.optimizer.param_groups[0]['lr'], losses_reduced.item(), arch_losses_reduced,
                        str(datetime.timedelta(seconds=int(time.time() - start_time))),
                        eta_string))

            if iteration % self.iters_per_epoch == 0 and self.save_to_disk:
                save_checkpoint(self.model, epoch, self.optimizer, self.arch_optimizer, self.lr_scheduler, self.arch_lr_scheduler, is_best=False)
                writer.add_scalar('Train/train_loss', losses_reduced.item(), epoch)
                writer.add_scalar('Train/search_loss', arch_losses_reduced, epoch)
                if cfg.ARCH.SEARCHSPACE == 'GeneralizedFastSCNN' and epoch > cfg.ARCH.SEARCH_EPOCH:
                    writer.add_scalar('temperature', self.model.get_temperature(), epoch)
                    alpha1 = torch.sigmoid(self.model.net1_alphas).detach().cpu().numpy()
                    alpha2 = torch.sigmoid(self.model.net2_alphas).detach().cpu().numpy()
                    alpha1_path = os.path.join(cfg.TRAIN.LOG_SAVE_DIR, 'alpha1')
                    if not os.path.isdir(alpha1_path):
                        os.makedirs(alpha1_path)
                    alpha2_path = os.path.join(cfg.TRAIN.LOG_SAVE_DIR, 'alpha2')
                    if not os.path.isdir(alpha2_path):
                        os.makedirs(alpha2_path)
                    heatmap1 = save_heatmap(alpha1, os.path.join(alpha1_path, "%s_alpha1.png"%str(epoch).zfill(5)),save=True)
                    heatmap2 = save_heatmap(alpha2, os.path.join(alpha2_path, "%s_alpha2.png"%str(epoch).zfill(5)),save=True)
                    # writer.add_image('alpha/net1', heatmap1, epoch)
                    # writer.add_image('alpha/net2', heatmap2, epoch)
                    writer.add_image('alpha/net1', heatmap1, epoch)
                    writer.add_image('alpha/net2', heatmap2, epoch)
                    network_path = os.path.join(cfg.TRAIN.LOG_SAVE_DIR, 'network')
                    if not os.path.isdir(network_path):
                        os.makedirs(network_path)
                    connectivity_plot = save_connectivity(alpha1, alpha2,
                                                          self.model.net1_connectivity_matrix,
                                                          self.model.net2_connectivity_matrix,
                                                          os.path.join(network_path, "%s_network.png" %str(epoch).zfill(5)),
                                                          save=True
                                                         )
                    writer.add_image('network', connectivity_plot, epoch)

            if not self.args.skip_val and iteration % val_per_iters == 0 and epoch > cfg.ARCH.SEARCH_EPOCH:
                self.validation(epoch)
                self.model.train()

        total_training_time = time.time() - start_time
        total_training_str = str(datetime.timedelta(seconds=total_training_time))
        logging.info(
            "Total training time: {} ({:.4f}s / it)".format(
                total_training_str, total_training_time / max_iters))

    def validation(self, epoch):
        self.metric.reset()
        if self.args.distributed:
            model = self.model.module
        else:
            model = self.model
        torch.cuda.empty_cache()
        model.eval()
        model.arch_eval()
        for i, (image, target, filename) in enumerate(self.val_loader):
            image = image.to(self.device)
            target = target.to(self.device)

            with torch.no_grad():
                if cfg.DATASET.MODE == 'val' or cfg.TEST.CROP_SIZE is None:
                    output = model(image)[0]
                else:
                    size = image.size()[2:]
                    pad_height = cfg.TEST.CROP_SIZE[0] - size[0]
                    pad_width = cfg.TEST.CROP_SIZE[1] - size[1]
                    image = F.pad(image, (0, pad_height, 0, pad_width))
                    output = model(image)[0]
                    output = output[..., :size[0], :size[1]]

            self.metric.update(output, target)
            pixAcc, mIoU = self.metric.get()
            logging.info("[EVAL] Sample: {:d}, pixAcc: {:.3f}, mIoU: {:.3f}".format(i + 1, pixAcc * 100, mIoU * 100))
        pixAcc, mIoU = self.metric.get()
        logging.info("[EVAL END] Epoch: {:d}, pixAcc: {:.3f}, mIoU: {:.3f}".format(epoch, pixAcc * 100, mIoU * 100))
        writer.add_scalar('Val/pixAcc', pixAcc * 100, epoch)
        writer.add_scalar('Val/mIoU', mIoU * 100, epoch)
        synchronize()
        if self.best_pred < mIoU and self.save_to_disk:
            self.best_pred = mIoU
            logging.info('Epoch {} is the best model, best pixAcc: {:.3f}, mIoU: {:.3f}, save the model..'.format(epoch, pixAcc * 100, mIoU * 100))
            save_checkpoint(model, epoch, is_best=True)


if __name__ == '__main__':
    args = parse_args()
    # get config
    cfg.update_from_file(args.config_file)
    cfg.update_from_list(args.opts)
    cfg.PHASE = 'train'
    cfg.ROOT_PATH = root_path
    cfg.check_and_freeze()

    # setup python train environment, logger, seed..
    default_setup(args)
    os.environ['PYTHONHASHSEED'] = str(cfg.SEED)
    random.seed(cfg.SEED)
    np.random.seed(cfg.SEED)
    torch.manual_seed(cfg.SEED)
    torch.cuda.manual_seed(cfg.SEED)
    torch.cuda.manual_seed_all(cfg.SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # This can slow down training

    # create a trainer and start train
    writer = SummaryWriter(logdir=os.path.join(cfg.TRAIN.LOG_SAVE_DIR, 'log'+str(cfg.TIME_STAMP)))
    trainer = Trainer(args)
    trainer.train()
