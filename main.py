#!/usr/bin/env python

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import utils as torchvision_utils
from tensorboardX import SummaryWriter

import argparse, os, sys, subprocess
import setproctitle, colorama
import numpy as np
from tqdm import tqdm
from glob import glob
from os.path import *

import models, losses, datasets
from utils import flow_utils, tools
from utils.flow2image import f2i
from utils.FlowNetPytorch import flow_transforms

# fp32 copy of parameters for update
global param_copy

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--start_epoch', type=int, default=1)
    parser.add_argument('--total_epochs', type=int, default=10000)
    parser.add_argument('--batch_size', '-b', type=int, default=8, help="Batch size")
    parser.add_argument('--train_n_batches', type=int, default=-1, help='Number of min-batches per epoch. If < 0, it will be determined by training_dataloader')
    parser.add_argument('--crop_size', type=int, nargs='+', default=[256, 256], help="Spatial dimension to crop training samples for training")
    parser.add_argument('--gradient_clip', type=float, default=None)
    parser.add_argument("--rgb_max", type=float, default=255.)

    parser.add_argument("--weight_decay", type=float, default=4e-4)
    parser.add_argument("--bias_decay", type=float, default=0.0)

    parser.add_argument('--number_workers', '-nw', '--num_workers', type=int, default=8)
    parser.add_argument('--number_gpus', '-ng', type=int, default=-1, help='number of GPUs to use')
    parser.add_argument('--no_cuda', action='store_true')

    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--name', default='run', type=str, help='a name to append to the save directory')
    parser.add_argument('--save', '-s', default='./work', type=str, help='directory for saving')

    parser.add_argument('--train_transforms', action='store_true', help='use image augmentation during training')

    parser.add_argument('--validation_frequency', type=int, default=5, help='validate every n epochs')
    parser.add_argument('--validation_n_batches', type=int, default=-1)
    parser.add_argument('--validation_log_images', action='store_true')
    parser.add_argument('--render_validation', action='store_true', help='run inference (save flows to file) and every validation_frequency epoch')

    parser.add_argument('--inference', action='store_true')
    parser.add_argument('--inference_size', type=int, nargs='+', default = [-1,-1], help='spatial size divisible by 64. default (-1,-1) - largest possible valid size would be used')
    parser.add_argument('--inference_batch_size', type=int, default=1)
    parser.add_argument('--inference_n_batches', type=int, default=-1)
    parser.add_argument('--save_flow', action='store_true', help='save predicted flows to file')

    parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
    parser.add_argument('--log_frequency', '--summ_iter', type=int, default=1, help="Log every n batches")

    parser.add_argument('--skip_training', action='store_true')
    parser.add_argument('--skip_validation', action='store_true')

    parser.add_argument('--fp16', action='store_true', help='Run model in pseudo-fp16 mode (fp16 storage fp32 math).')
    parser.add_argument('--fp16_scale', type=float, default=1024., help='Loss scaling, positive power of 2 values can improve fp16 convergence.')

    parser.add_argument("--pwcnet_md", type=int, default=4)

    parser.add_argument("--scheduler_milestones", type=int, nargs='+', default=[300000, 400000, 500000])

    tools.add_arguments_for_module(parser, models, argument_for_class='model', default='FlowNet2')

    tools.add_arguments_for_module(parser, losses, argument_for_class='loss', default='L1Loss')

    tools.add_arguments_for_module(parser, torch.optim, argument_for_class='optimizer', default='Adam', skip_params=['params'])

    tools.add_arguments_for_module(parser, torch.optim.lr_scheduler, argument_for_class='scheduler', default='MultiStepLR')
    
    tools.add_arguments_for_module(parser, datasets, argument_for_class='training_dataset', default='MpiSintelFinal', 
                                    skip_params=['is_cropped'],
                                    parameter_defaults={'root': './MPI-Sintel/flow/training'})
    
    tools.add_arguments_for_module(parser, datasets, argument_for_class='validation_dataset', default='MpiSintelClean', 
                                    skip_params=['is_cropped'],
                                    parameter_defaults={'root': './MPI-Sintel/flow/training',
                                                        'replicates': 1})
    
    tools.add_arguments_for_module(parser, datasets, argument_for_class='inference_dataset', default='MpiSintelClean', 
                                    skip_params=['is_cropped'],
                                    parameter_defaults={'root': './MPI-Sintel/flow/training',
                                                        'replicates': 1})

    main_dir = os.path.dirname(os.path.realpath(__file__))
    os.chdir(main_dir)

    # Parse the official arguments
    with tools.TimerBlock("Parsing Arguments") as block:
        args = parser.parse_args()
        if args.number_gpus < 0 : args.number_gpus = torch.cuda.device_count()

        # Get argument defaults (hastag #thisisahack)
        parser.add_argument('--IGNORE',  action='store_true')
        defaults = vars(parser.parse_args(['--IGNORE']))

        # Print all arguments, color the non-defaults
        for argument, value in sorted(vars(args).items()):
            reset = colorama.Style.RESET_ALL
            color = reset if value == defaults[argument] else colorama.Fore.MAGENTA
            block.log('{}{}: {}{}'.format(color, argument, value, reset))

        args.model_class = tools.module_to_dict(models)[args.model]
        args.optimizer_class = tools.module_to_dict(torch.optim)[args.optimizer]
        args.loss_class = tools.module_to_dict(losses)[args.loss]
        args.scheduler_class = tools.module_to_dict(torch.optim.lr_scheduler)[args.scheduler]

        args.training_dataset_class = tools.module_to_dict(datasets)[args.training_dataset]
        args.validation_dataset_class = tools.module_to_dict(datasets)[args.validation_dataset]
        args.inference_dataset_class = tools.module_to_dict(datasets)[args.inference_dataset]

        args.cuda = not args.no_cuda and torch.cuda.is_available()
        args.current_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).rstrip()
        args.log_file = join(args.save, 'args.txt')

        # dict to collect activation gradients (for training debug purpose)
        args.grads = {}

        if args.inference:
            args.skip_validation = True
            args.skip_training = True
            args.total_epochs = 1
            args.inference_dir = "{}/inference".format(args.save)

    args.scheduler_gamma = 0.5

    print('Source Code')
    print(('  Current Git Hash: {}\n'.format(args.current_hash)))

    # Change the title for `top` and `pkill` commands
    setproctitle.setproctitle(args.save)

    # Dynamically load the dataset class with parameters passed in via "--argument_[param]=[value]" arguments
    with tools.TimerBlock("Initializing Datasets") as block:
        args.effective_batch_size = args.batch_size * args.number_gpus
        args.effective_inference_batch_size = args.inference_batch_size * args.number_gpus
        args.effective_number_workers = args.number_workers * args.number_gpus
        gpuargs = {'num_workers': args.effective_number_workers, 
                   'pin_memory': True, 
                   'drop_last' : True} if args.cuda else {}
        inf_gpuargs = gpuargs.copy()
        inf_gpuargs['num_workers'] = args.number_workers

        if args.train_transforms:
            args.training_dataset_transforms = flow_transforms.Compose([
                flow_transforms.RandomTranslate(10),
                flow_transforms.RandomRotate(10, 5),
                flow_transforms.RandomCrop((args.crop_size[0], args.crop_size[1])),
                flow_transforms.RandomVerticalFlip(),
                flow_transforms.RandomHorizontalFlip()
            ])

        if exists(args.training_dataset_root):
            train_dataset = args.training_dataset_class(args, True, **tools.kwargs_from_args(args, 'training_dataset'))
            block.log('Training Dataset: {}'.format(args.training_dataset))
            block.log('Training Input: {}'.format(' '.join([str([d for d in x.size()]) for x in train_dataset[0][0]])))
            block.log('Training Targets: {}'.format(' '.join([str([d for d in x.size()]) for x in train_dataset[0][1]])))
            train_loader = DataLoader(train_dataset, batch_size=args.effective_batch_size, shuffle=True, **gpuargs)

        if exists(args.validation_dataset_root):
            validation_dataset = args.validation_dataset_class(args, True, **tools.kwargs_from_args(args, 'validation_dataset'))
            block.log('Validation Dataset: {}'.format(args.validation_dataset))
            block.log('Validation Input: {}'.format(' '.join([str([d for d in x.size()]) for x in validation_dataset[0][0]])))
            block.log('Validation Targets: {}'.format(' '.join([str([d for d in x.size()]) for x in validation_dataset[0][1]])))
            validation_loader = DataLoader(validation_dataset, batch_size=args.effective_batch_size, shuffle=False, **gpuargs)

        if exists(args.inference_dataset_root):
            inference_dataset = args.inference_dataset_class(args, False, **tools.kwargs_from_args(args, 'inference_dataset'))
            block.log('Inference Dataset: {}'.format(args.inference_dataset))
            block.log('Inference Input: {}'.format(' '.join([str([d for d in x.size()]) for x in inference_dataset[0][0]])))
            block.log('Inference Targets: {}'.format(' '.join([str([d for d in x.size()]) for x in inference_dataset[0][1]])))
            inference_loader = DataLoader(inference_dataset, batch_size=args.effective_inference_batch_size, shuffle=False, **inf_gpuargs)

    # Dynamically load model and loss class with parameters passed in via "--model_[param]=[value]" or "--loss_[param]=[value]" arguments
    with tools.TimerBlock("Building {} model".format(args.model)) as block:
        class ModelAndLoss(nn.Module):
            def __init__(self, args):
                super(ModelAndLoss, self).__init__()
                kwargs = tools.kwargs_from_args(args, 'model')
                self.model = args.model_class(args, **kwargs)
                kwargs = tools.kwargs_from_args(args, 'loss')
                self.loss = args.loss_class(args, **kwargs)
                
            def forward(self, data, target, inference=False ):
                output = self.model(data)

                loss_values = self.loss(output, target)

                if not inference :
                    return loss_values
                else :
                    return loss_values, output

        model_and_loss = ModelAndLoss(args)

        block.log('Effective Batch Size: {}'.format(args.effective_batch_size))
        block.log('Number of parameters: {}'.format(sum([p.data.nelement() if p.requires_grad else 0 for p in model_and_loss.parameters()])))

        # assing to cuda or wrap with dataparallel, model and loss 
        if args.cuda and (args.number_gpus > 0) and args.fp16:
            block.log('Parallelizing')
            model_and_loss = nn.DataParallel(model_and_loss, device_ids=list(range(args.number_gpus)))

            block.log('Initializing CUDA')
            model_and_loss = model_and_loss.cuda().half()
            torch.cuda.manual_seed(args.seed) 
            param_copy = [param.clone().type(torch.cuda.FloatTensor).detach() for param in model_and_loss.parameters()]

        elif args.cuda and args.number_gpus > 0:
            block.log('Initializing CUDA')
            model_and_loss = model_and_loss.cuda()
            block.log('Parallelizing')
            model_and_loss = nn.DataParallel(model_and_loss, device_ids=list(range(args.number_gpus)))
            torch.cuda.manual_seed(args.seed) 

        else:
            block.log('CUDA not being used')
            torch.manual_seed(args.seed)

        block.log("Initializing save directory: {}".format(args.save))
        if not os.path.exists(args.save):
            os.makedirs(args.save)

        train_logger = SummaryWriter(log_dir = os.path.join(args.save, 'train'), comment = 'training')
        validation_logger = SummaryWriter(log_dir = os.path.join(args.save, 'validation'), comment = 'validation')

    # Dynamically load the optimizer with parameters passed in via "--optimizer_[param]=[value]" arguments 
    with tools.TimerBlock("Initializing {} Optimizer".format(args.optimizer)) as block:
        kwargs = tools.kwargs_from_args(args, 'optimizer')
        if args.fp16:
            params_groups = [
                {'params': [param_copy[i] for i, p in enumerate(model_and_loss.named_parameters()) if 'weight' in p[0] and p[1].requires_grad],
                 'weight_decay': args.weight_decay},
                {'params': [param_copy[i] for i, p in enumerate(model_and_loss.named_parameters()) if 'bias' in p[0] and p[1].requires_grad],
                 'weight_decay': args.bias_decay}]
            optimizer = args.optimizer_class(params_groups, **kwargs)
        else:
            params_groups = [
                {'params': [p[1] for p in model_and_loss.named_parameters() if 'weight' in p[0] and p[1].requires_grad],
                 'weight_decay': args.weight_decay},
                {'params': [p[1] for p in model_and_loss.named_parameters() if 'bias' in p[0] and p[1].requires_grad],
                 'weight_decay': args.bias_decay}]
            optimizer = args.optimizer_class(params_groups, **kwargs)
        # for param, default in list(kwargs.items()):
        #     block.log("{} = {} ({})".format(param, default, type(default)))

    # Dynamically load the scheduler with parameters passed in via "--scheduler_[param]=[value]" arguments 
    with tools.TimerBlock("Initializing {} LR Scheduler".format(args.scheduler)) as block:
        kwargs = tools.kwargs_from_args(args, 'scheduler')
        scheduler = args.scheduler_class(optimizer, **kwargs)
        # for param, default in list(kwargs.items()):
        #     block.log("{} = {} ({})".format(param, default, type(default)))

    with tools.TimerBlock("Loading {} checkpoint".format(args.model)) as block:
        def load_checkpoint(model, optimizer, scheduler, save_path):
            epoch = 0
            iteration = 0
            best_epe = 99999.9
            checkpoint = torch.load(save_path)
            try:
                # Models saved by this code
                epoch = checkpoint['epoch']
                iteration = checkpoint['iteration']
                best_epe = checkpoint['best_EPE']
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                try:
                    scheduler.state_dict()
                    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                except AttributeError:
                    pass
            except KeyError:
                try:
                    # FlowNet2 pretrained models
                    model.load_state_dict(checkpoint['state_dict'])
                except KeyError:
                    # PWCDCNet pretrained model
                    pretrained_dict = checkpoint
                    model_dict = model_and_loss.module.model.state_dict()
                    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                    model_dict.update(pretrained_dict)
                    model.load_state_dict(pretrained_dict)
                
            return model, optimizer, scheduler, epoch, iteration, best_epe

        # Load weights if needed, otherwise randomly initialize
        if args.resume and os.path.isfile(args.resume):
            model_and_loss.module.model, optimizer, scheduler, args.start_epoch, args.start_iteration, best_err = load_checkpoint(model_and_loss.module.model, optimizer, scheduler, args.resume)
            block.log("Loaded checkpoint '{}'".format(args.resume))
            block.log("Start epoch {}".format(args.start_epoch))
            block.log("Best EPE {}".format(best_err))

        elif args.resume and args.inference:
            block.log("No checkpoint found at '{}'".format(args.resume))
            quit()

        else:
            args.start_iteration = 0
            block.log("Random initialization")

    with tools.TimerBlock("Optimizer and Scheduler states") as block:
        block.log("Optimizer state:")
        block.log(optimizer)
        try:
            ss = scheduler.state_dict()
            block.log("LR Scheduler state:")
            block.log(ss)
        except AttributeError:
            pass

    # Log all arguments to file
    for argument, value in sorted(vars(args).items()):
        block.log2file(args.log_file, '{}: {}'.format(argument, value))

    # Reusable function for training and validataion
    def train(args, epoch, start_iteration, data_loader, model, optimizer, scheduler, logger, is_validate=False, offset=0, max_flows_to_show=8):
        running_statistics = None  # Initialize below when the first losses are collected
        all_losses = None  # Initialize below when the first losses are collected
        total_loss = 0

        if is_validate:
            model.eval()
            title = 'Validating Epoch {}'.format(epoch)
            args.validation_n_batches = np.inf if args.validation_n_batches < 0 else args.validation_n_batches
            progress = tqdm(tools.IteratorTimer(data_loader), ncols=100, total=np.minimum(len(data_loader), args.validation_n_batches), leave=True, position=offset, desc=title)
        else:
            model.train()
            title = 'Training Epoch {}'.format(epoch)
            args.train_n_batches = np.inf if args.train_n_batches < 0 else args.train_n_batches
            progress = tqdm(tools.IteratorTimer(data_loader), ncols=120, total=np.minimum(len(data_loader), args.train_n_batches), smoothing=.9, miniters=1, leave=True, position=offset, desc=title)

        def convert_flow_to_image(flow_converter, flows_viz):
            imgs = []
            for flow_pair in flows_viz:
                for flow in flow_pair:
                    flow = flow.numpy().transpose((1, 2, 0))
                    img = flow_converter._flowToColor(flow)
                    imgs.append(torch.from_numpy(img.transpose((2, 0, 1))))
                epe_img = torch.sqrt(torch.sum(torch.pow(flow_pair[0] - flow_pair[1], 2), dim=0))
                max_epe = torch.max(epe_img)
                if max_epe == 0:
                    max_epe = torch.ones(1)
                epe_img = epe_img / max_epe
                epe_img = (255 * epe_img).type(torch.uint8)
                epe_img = torch.stack((epe_img, epe_img, epe_img), dim=0)
                imgs.append(epe_img)
            return imgs

        max_iters = min(len(data_loader),
                        (args.validation_n_batches if (is_validate and args.validation_n_batches > 0) else len(data_loader)),
                        (args.train_n_batches if (not is_validate and args.train_n_batches > 0) else len(data_loader)))

        if is_validate:
            flow_converter = f2i.Flow()
            collect_flow_interval = int(np.ceil(float(max_iters) / max_flows_to_show))
            flows_viz = []

        last_log_batch_idx = 0
        last_log_time = progress._time()
        for batch_idx, (data, target) in enumerate(progress):
            global_iteration = start_iteration + batch_idx

            data, target = [Variable(d) for d in data], [Variable(t) for t in target]
            if args.cuda and args.number_gpus == 1:
                data, target = [d.cuda(async=True) for d in data], [t.cuda(async=True) for t in target]

            optimizer.zero_grad() if not is_validate else None
            losses, output = model(data[0], target[0], inference=True)
            losses = [torch.mean(loss_value) for loss_value in losses]
            loss_val = losses[0] # Collect first loss for weight update
            total_loss += loss_val.item()
            loss_values = [v.item() for v in losses]

            if is_validate and batch_idx % collect_flow_interval == 0:
                flows_viz.append((target[0][0].detach().cpu(), output[0].detach().cpu()))

            if is_validate and args.validation_log_images and batch_idx == (max_iters - 1):
                imgs = convert_flow_to_image(flow_converter, flows_viz)
                imgs = torchvision_utils.make_grid(imgs, nrow=3, normalize=False, scale_each=False)
                logger.add_image('target/predicted flows', imgs, global_iteration)

            # gather loss_labels, direct return leads to recursion limit error as it looks for variables to gather'
            loss_labels = list(model.module.loss.loss_labels)

            assert not np.isnan(total_loss)

            if not is_validate and args.fp16:
                loss_val.backward()
                if args.gradient_clip:
                    torch.nn.utils.clip_grad_norm(model.parameters(), args.gradient_clip)

                params = list(model.parameters())
                for i in range(len(params)):
                   param_copy[i].grad = params[i].grad.clone().type_as(params[i]).detach()
                   param_copy[i].grad.mul_(1./args.loss_scale)
                optimizer.step()
                for i in range(len(params)):
                    params[i].data.copy_(param_copy[i].data)

            elif not is_validate:
                loss_val.backward()
                if args.gradient_clip:
                    torch.nn.utils.clip_grad_norm(model.parameters(), args.gradient_clip)
                optimizer.step()

            # Update hyperparameters if needed
            if not is_validate:
                scheduler.step()
                loss_labels.append('lr')
                loss_values.append(optimizer.param_groups[0]['lr'])

            loss_labels.append('load')
            loss_values.append(progress.iterable.last_duration)

            if running_statistics is None:
                running_statistics = np.array(loss_values)
                all_losses = np.zeros((len(data_loader), len(loss_values)), np.float32)
            else:
                running_statistics += np.array(loss_values)
            all_losses[batch_idx] = loss_values.copy()
            title = '{} Epoch {}'.format('Validating' if is_validate else 'Training', epoch)

            progress.set_description(title + ' ' + tools.format_dictionary_of_losses(loss_labels, running_statistics / (batch_idx + 1)))

            if ((((global_iteration + 1) % args.log_frequency) == 0 and not is_validate) or
                    (batch_idx == max_iters - 1)):

                global_iteration = global_iteration if not is_validate else start_iteration

                logger.add_scalar('batch logs per second', (batch_idx - last_log_batch_idx) / (progress._time() - last_log_time), global_iteration)
                last_log_time = progress._time()
                last_log_batch_idx = batch_idx

                for i, key in enumerate(loss_labels):
                    logger.add_scalar('average batch ' + str(key), all_losses[:batch_idx + 1, i].mean(), global_iteration)
                    logger.add_histogram(str(key), all_losses[:batch_idx + 1, i], global_iteration)

            if ( is_validate and ( batch_idx == args.validation_n_batches) ):
                break

            if ( (not is_validate) and (batch_idx == (args.train_n_batches)) ):
                break

        progress.close()

        return total_loss / float(batch_idx + 1), (batch_idx + 1)

    # Reusable function for inference
    def inference(args, epoch, data_loader, model, offset=0):

        model.eval()
        
        if args.save_flow or args.render_validation:
            flow_folder = "{}/inference/{}.epoch-{}-flow-field".format(args.save,args.name.replace('/', '.'),epoch)
            if not os.path.exists(flow_folder):
                os.makedirs(flow_folder)

        
        args.inference_n_batches = np.inf if args.inference_n_batches < 0 else args.inference_n_batches

        progress = tqdm(data_loader, ncols=100, total=np.minimum(len(data_loader), args.inference_n_batches), desc='Inferencing ', 
            leave=True, position=offset)

        statistics = []
        total_loss = 0
        for batch_idx, (data, target) in enumerate(progress):
            if args.cuda:
                data, target = [d.cuda(async=True) for d in data], [t.cuda(async=True) for t in target]
            data, target = [Variable(d) for d in data], [Variable(t) for t in target]

            # when ground-truth flows are not available for inference_dataset, 
            # the targets are set to all zeros. thus, losses are actually L1 or L2 norms of compute optical flows, 
            # depending on the type of loss norm passed in
            with torch.no_grad():
                losses, output = model(data[0], target[0], inference=True)
                
            losses = [torch.mean(loss_value) for loss_value in losses] 
            loss_val = losses[0] # Collect first loss for weight update
            total_loss += loss_val.item()
            loss_values = [v.item() for v in losses]

            # gather loss_labels, direct return leads to recursion limit error as it looks for variables to gather'
            loss_labels = list(model.module.loss.loss_labels)

            statistics.append(loss_values)
            # import IPython; IPython.embed()
            if args.save_flow or args.render_validation:
                for i in range(args.inference_batch_size):
                    _pflow = output[i].data.cpu().numpy().transpose(1, 2, 0)
                    flow_utils.writeFlow( join(flow_folder, '%06d.flo'%(batch_idx * args.inference_batch_size + i)),  _pflow)

            progress.set_description('Inference Averages for Epoch {}: '.format(epoch) + tools.format_dictionary_of_losses(loss_labels, np.array(statistics).mean(axis=0)))
            progress.update(1)

            if batch_idx == (args.inference_n_batches - 1):
                break

        progress.close()

        return

    def generate_checkpoint_state(model, optimizer, scheduler, arch, epoch, iteration, best_epe, is_training):
        state = {
            'arch' : arch,
            'epoch': epoch,
            'iteration': iteration,
            'best_EPE': best_epe,
            'model_state_dict': model.state_dict()
        }
        if is_training:
            state['optimizer_state_dict'] = optimizer.state_dict()
            try:
                scheduler.state_dict()
                state['scheduler_state_dict'] = scheduler.state_dict()
            except AttributeError:
                pass
        return state

    # Primary epoch loop
    best_err = 1e8
    progress = tqdm(list(range(args.start_epoch, args.total_epochs + 1)), miniters=1, ncols=100, desc='Overall Progress', leave=True, position=0)
    offset = 1
    last_epoch_time = progress._time()
    global_iteration = args.start_iteration

    for epoch in progress:
        if not args.skip_training:
            train_loss, iterations = train(args=args, epoch=epoch, start_iteration=global_iteration, data_loader=train_loader, model=model_and_loss, optimizer=optimizer, scheduler=scheduler, logger=train_logger, offset=offset)
            global_iteration += iterations
            offset += 1

            # save checkpoint after every validation_frequency number of epochs
            if ((epoch - 1) % args.validation_frequency) == 0:
                checkpoint_progress = tqdm(ncols=100, desc='Saving Checkpoint', position=offset)
                tools.save_checkpoint(generate_checkpoint_state(model_and_loss.module.model, optimizer, scheduler, args.model, epoch+1, global_iteration, train_loss, True), 
                                      False, args.save, args.model, filename = 'train-checkpoint.pth.tar')
                checkpoint_progress.update(1)
                checkpoint_progress.close()

        if not args.skip_validation and (epoch % args.validation_frequency) == 0:
            validation_loss, _ = train(args=args, epoch=epoch, start_iteration=global_iteration, data_loader=validation_loader, model=model_and_loss, optimizer=optimizer, scheduler=scheduler, logger=validation_logger, is_validate=True, offset=offset)
            offset += 1

            is_best = False
            if validation_loss < best_err:
                best_err = validation_loss
                is_best = True

            checkpoint_progress = tqdm(ncols=100, desc='Saving Checkpoint', position=offset)
            tools.save_checkpoint(generate_checkpoint_state(model_and_loss.module.model, optimizer, scheduler, args.model, epoch+1, global_iteration, best_err, False), 
                                  is_best, args.save, args.model)
            checkpoint_progress.update(1)
            checkpoint_progress.close()
            offset += 1

        if args.inference or (args.render_validation and (epoch % args.validation_frequency) == 0):
            stats = inference(args=args, epoch=epoch, data_loader=inference_loader, model=model_and_loss, offset=offset)
            offset += 1

        train_logger.add_scalar('seconds per epoch', progress._time() - last_epoch_time, epoch)
        last_epoch_time = progress._time()
    print("\n")
