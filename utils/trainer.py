from thop import profile
import torch
import os
import time
from model.segmentors import MambaSeg
from utils.func import IOStream
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.optim import AdamW
from utils.scheduler import PolynomialLR, WarmupPolyLR, WarmupCosineLR
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from utils.metrics import MetricsSemseg
from utils.labels import dataset_info

REDUCE_LR_SCHEDULERS = ['reduce01', 'reduce05']

class Trainer():

    def __init__(self, cfg):

        self.cfg = cfg

        # Initialize TensorBoard
        self.writer = SummaryWriter(log_dir=self.cfg['TRAIN']['log_dir'])

        # Initialize log printer
        self.printer = IOStream(self.cfg['TRAIN']['log_dir'] + '/run.log')
        print(str(self.cfg))
        self.printer.cprint(str(self.cfg))

        # Dataset and loader
        semseg_ignore_label, semseg_class_names, _ = dataset_info(semseg_num_classes=self.cfg['DATASET']['classes'])
        if self.cfg['DATASET']['name'] == 'DDD17Event':
            from datasets.ddd17_dataset import DDD17Event
            self.training_dataset = DDD17Event(root=self.cfg['DATASET']['path'], split='train',
                                               event_representation=self.cfg['DATASET']['event_representation'],
                                               nr_events_data=1, delta_t_per_data=self.cfg['DATASET']['delta_t'],
                                               nr_bins_per_data=self.cfg['DATASET']['nr_bins'],
                                               require_paired_data=self.cfg['DATASET']['require_paired_data'],
                                               augmentation=True, fixed_duration=self.cfg['DATASET']['fixed_duration'],
                                               random_crop=True)
            self.validation_dataset = DDD17Event(root=self.cfg['DATASET']['path'], split='test',
                                                 event_representation=self.cfg['DATASET']['event_representation'],
                                                 nr_events_data=1, delta_t_per_data=self.cfg['DATASET']['delta_t'],
                                                 nr_bins_per_data=self.cfg['DATASET']['nr_bins'],
                                                 require_paired_data=self.cfg['DATASET']['require_paired_data'],
                                                 augmentation=False, fixed_duration=self.cfg['DATASET']['fixed_duration'],
                                                 random_crop=False)
            self.training_loader = DataLoader(self.training_dataset, num_workers=self.cfg['NUM_WORKERS'],
                                              batch_size=self.cfg['TRAIN']['batch_size'], shuffle=True)
            self.validation_loader = DataLoader(self.validation_dataset, num_workers=self.cfg['NUM_WORKERS'],
                                                batch_size=self.cfg['TRAIN']['batch_size'])
        elif self.cfg['DATASET']['name'] == 'DSECEvent':
            from datasets.dsec_dataset import DSECEvent
            self.training_dataset = DSECEvent(self.cfg['DATASET']['path'], nr_events_data=1,
                                              delta_t_per_data=self.cfg['DATASET']['delta_t'], 
                                              nr_events_window=self.cfg['DATASET']['nr_events'], 
                                              augmentation=True,
                                              mode='train', event_representation=self.cfg['DATASET']['event_representation'],
                                              nr_bins_per_data=self.cfg['DATASET']['nr_bins'],
                                              require_paired_data=self.cfg['DATASET']['require_paired_data'],
                                              semseg_num_classes=self.cfg['DATASET']['classes'],
                                              fixed_duration=self.cfg['DATASET']['fixed_duration'], random_crop=True)
            self.validation_dataset = DSECEvent(self.cfg['DATASET']['path'], nr_events_data=1,
                                                delta_t_per_data=self.cfg['DATASET']['delta_t'], 
                                                nr_events_window=self.cfg['DATASET']['nr_events'], 
                                                augmentation=False,
                                                mode='val', event_representation=self.cfg['DATASET']['event_representation'],
                                                nr_bins_per_data=self.cfg['DATASET']['nr_bins'],
                                                require_paired_data=self.cfg['DATASET']['require_paired_data'],
                                                semseg_num_classes=self.cfg['DATASET']['classes'],
                                                fixed_duration=self.cfg['DATASET']['fixed_duration'], random_crop=False)
            self.training_loader = DataLoader(self.training_dataset, num_workers=self.cfg['NUM_WORKERS'],
                                              batch_size=self.cfg['TRAIN']['batch_size'], shuffle=True)
            self.validation_loader = DataLoader(self.validation_dataset, num_workers=self.cfg['NUM_WORKERS'],
                                                batch_size=self.cfg['TRAIN']['batch_size'])

        # Model
        self.model = MambaSeg(ver_img=self.cfg['MODEL']['version_img'], 
                           ver_ev=self.cfg['MODEL']['version_ev'], 
                           num_classes=self.cfg['DATASET']['classes'], 
                           num_channels_img=self.cfg['DATASET']['img_chnls'], 
                           pretrained_img=self.cfg['MODEL']['pretrained_img'], 
                           num_channels_ev=self.cfg['DATASET']['nr_bins'], 
                           pretrained_ev=self.cfg['MODEL']['pretrained_ev'],
                           img_size=self.cfg['DATASET']['img_size'],
                           if_viz=False)

        # Optimizer, learning rate scheduler and loss function
        self.opt = AdamW(params=self.model.parameters(), lr=self.cfg['TRAIN']['lr_init'])
        self.cur_iter = 0
        max_epochs = int(self.cfg['TRAIN']['num_epochs'])
        total_iters = int(max_epochs * len(self.training_loader))
        warmup_iters = int(self.cfg['TRAIN']['warmup_iters'])
        assert warmup_iters <= total_iters, f"Warmup iterations should be less than total iterations, and the total iterations is {total_iters}."

        if self.cfg['TRAIN']['lr_scheduler'] == 'polynomial':
            self.scheduler = PolynomialLR(optimizer=self.opt, total_iters=total_iters)
        elif self.cfg['TRAIN']['lr_scheduler'] == 'warmpoly':
            self.scheduler = WarmupPolyLR(optimizer=self.opt, T_max=total_iters, cur_iter=self.cur_iter, 
                                          warmup_factor=1.0 / 3, warmup_iters=warmup_iters, power=0.8)
        elif self.cfg['TRAIN']['lr_scheduler'] == 'StepLR':
            self.scheduler = torch.optim.lr_scheduler.StepLR(optimizer=self.opt, step_size=10, gamma=0.92, 
                                                             last_epoch=-1, verbose='deprecated')
        elif self.cfg['TRAIN']['lr_scheduler'] == 'warmupcosine':
            self.scheduler = WarmupCosineLR(optimizer=self.opt, T_max=(max_epochs * total_iters))
        elif self.cfg['TRAIN']['lr_scheduler'] == 'reduce01':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=self.opt, mode='max', factor=0.1,
                                                                        patience=7, verbose=True)
        elif self.cfg['TRAIN']['lr_scheduler'] == 'reduce05':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=self.opt, mode='max', factor=0.5,
                                                                        patience=5, verbose=True)
        elif self.cfg['TRAIN']['lr_scheduler'] == 'fixed':
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=self.opt, lr_lambda=lambda epoch: 1.0)



        self.criterion = CrossEntropyLoss(ignore_index=semseg_ignore_label)

        # Put the model into the computing device
        self.model.to(self.cfg['DEVICE'])

        # Evaluation metrics
        self.metrics = MetricsSemseg(self.cfg['DATASET']['classes'], semseg_ignore_label, semseg_class_names)


    def train(self):

        self.backup()

        best_mIOU = 0.0

        start = time.time()
        for epoch_id in range(self.cfg['TRAIN']['num_epochs']):
            print("Training step [{:3d}/{:3d}]".format(epoch_id + 1, self.cfg['TRAIN']['num_epochs']))
            self.train_epoch(epoch_id)
            if (epoch_id + 1) >= 1:
                print("Testing step [{:3d}/{:3d}]".format(epoch_id + 1, self.cfg['TRAIN']['num_epochs']))
                best_mIOU, eval_outs = self.eval(epoch_id, best_mIOU)
                # 根据eval_outs(Loss, mIOU, Acc)更新学习率
                if self.cfg['TRAIN']['lr_scheduler'] in REDUCE_LR_SCHEDULERS:
                    # 如果使用ReduceLROnPlateau，则需要传入eval_outs[1]（mIOU）作为指标
                    self.scheduler.step(eval_outs[1])
        self.writer.close()
        end = time.gmtime(time.time() - start)
        print('Total training time is:', time.strftime("%H:%M:%S", end))

    def train_epoch(self, epoch_id):

        training_loss = 0.0
        count = 0
        iteration = 0

        self.model.train()
        print('Current learning rate: %e'%(self.opt.state_dict()['param_groups'][0]['lr']))
        for ev_rep, img, label in tqdm(self.training_loader):
            self.cur_iter = epoch_id * len(self.training_loader) + iteration
            self.scheduler.cur_iter = self.cur_iter  # dynamic update cur_iter
            ev_rep, img, label = ev_rep.type(torch.FloatTensor).to(self.cfg['DEVICE']),\
                img.type(torch.FloatTensor).to(self.cfg['DEVICE']), label.to(self.cfg['DEVICE'])
            # Forward propagation and back propagation
            self.opt.zero_grad()
            pred = self.model(ev_rep, img)
            pred_label = pred.argmax(dim=1)
            loss = self.criterion(pred, label)
            loss.backward()
            self.opt.step()
            if self.cfg['TRAIN']['lr_scheduler'] not in REDUCE_LR_SCHEDULERS:
                self.scheduler.step()     #
            # Statistics
            count += self.cfg['TRAIN']['batch_size']
            iteration += 1
            training_loss += loss.item() * self.cfg['TRAIN']['batch_size']
            self.metrics.update_batch(pred_label, label)
        # Logger
        scores = self.metrics.get_metrics_summary()
        print("Loss: {:.4f}, mIOU: {:.4f}, Accuracy: {:.4f}".format(training_loss * 1.0 / count, scores['mean_iou'],
                                                                    scores['acc']))
        log_str = "[Train]  Epoch: {:d}, Loss: {:.4f}, mIOU: {:.4f}, Accuracy: {:.4f}".format(epoch_id + 1,
                                                                                              training_loss * 1.0 / count,
                                                                                              scores['mean_iou'],
                                                                                              scores['acc'])
        self.printer.cprint(log_str)
        self.writer.add_scalar('train/' + 'train_loss', training_loss * 1.0 / count, epoch_id + 1)
        self.writer.add_scalar('train/' + 'train_mIOU', scores['mean_iou'], epoch_id + 1)
        self.writer.add_scalar('train/' + 'train_acc', scores['acc'], epoch_id + 1)
        # Save the model
        if (epoch_id + 1) % self.cfg['TRAIN']['save_every_n_epochs'] == 0:
            torch.save({'epoch': epoch_id + 1, 'state_dict': self.model.state_dict()},
                       os.path.join(self.cfg['TRAIN']['log_dir'], 'checkpoint_epoch_' + str((epoch_id + 1)) + '.pth'))
            print("Save the model at {}".format(os.path.join(self.cfg['TRAIN']['log_dir'], 'checkpoint_epoch_' + str((epoch_id + 1)) + '.pth')))
        # Reset the evaluation metrics
        self.metrics.reset()

    def eval(self, epoch_id, best_mIOU):

        testing_loss = 0.0
        count = 0

        self.model.eval()

        with torch.no_grad():
            for ev_rep, img, label in tqdm(self.validation_loader):
                ev_rep, img, label = ev_rep.type(torch.FloatTensor).to(self.cfg['DEVICE']),\
                    img.type(torch.FloatTensor).to(self.cfg['DEVICE']), label.to(self.cfg['DEVICE'])
                pred = self.model(ev_rep, img)
                pred_label = pred.argmax(dim=1)
                loss = self.criterion(pred, label)
                # Statistics
                count += self.cfg['TRAIN']['batch_size']
                testing_loss += loss.item() * self.cfg['TRAIN']['batch_size']
                self.metrics.update_batch(pred_label, label)
        # Logger
        scores = self.metrics.get_metrics_summary()
        eval_outs = [testing_loss * 1.0 / count, scores['mean_iou'], scores['acc']]
        print("Loss: {:.4f}, mIOU: {:.4f}, Accuracy: {:.4f}".format(testing_loss * 1.0 / count,
                                                                    scores['mean_iou'], scores['acc']))
        log_str = "[Test]   Epoch: {:d}, Loss: {:.4f}, mIOU: {:.4f}, Accuracy: {:.4f}".format(epoch_id + 1,
                                                                                              testing_loss * 1.0 / count,
                                                                                              scores['mean_iou'],
                                                                                              scores['acc'])
        self.printer.cprint(log_str)
        self.writer.add_scalar('test/' + 'test_loss', testing_loss * 1.0 / count, epoch_id + 1)
        self.writer.add_scalar('test/' + 'test_mIOU', scores['mean_iou'], epoch_id + 1)
        self.writer.add_scalar('test/' + 'test_acc', scores['acc'], epoch_id + 1)
        # Save the model
        if scores['mean_iou'] >= best_mIOU:
            best_mIOU = scores['mean_iou']
            torch.save({'epoch': epoch_id + 1, 'state_dict': self.model.state_dict()},
                       os.path.join(self.cfg['TRAIN']['log_dir'], 'best_model' + '.pth'))
            print("Save the best model at {}".format(os.path.join(self.cfg['TRAIN']['log_dir'], 'best_model' + '.pth')))
            print('New best mIOU is %.4f'%best_mIOU)
            self.printer.cprint('New best mIOU is %.4f'%best_mIOU)
        # Reset the evaluation metrics
        self.metrics.reset()
        return best_mIOU, eval_outs

    def backup(self):
        root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
        if not os.path.exists(self.cfg['TRAIN']['log_dir'] + '/' + 'Backup'):
            os.makedirs(self.cfg['TRAIN']['log_dir'] + '/' + 'Backup')
        model_saving_dir = os.path.join(self.cfg['TRAIN']['log_dir'], 'Backup')
        os.system('cp %s '%(os.path.join(root, 'configs/DDD17.yaml')) + model_saving_dir + '/' + 'DDD17.yaml.backup')
        os.system('cp %s '% (os.path.join(root, 'configs/DSEC_Semantic.yaml')) + model_saving_dir + '/' + 'DSEC_Semantic.yaml.backup')
        os.system('cp %s '%(os.path.join(root, 'utils/trainer.py')) + model_saving_dir + '/' + 'trainer.py.backup')
        os.system('cp %s '%(os.path.join(root, 'model/fuse_modules.py')) + model_saving_dir + '/' + 'fuse_modules.py.backup')
        os.system('cp %s '%(os.path.join(root, 'model/segmentors.py')) + model_saving_dir + '/' + 'segmentors.py.backup')
        os.system('cp %s '%(os.path.join(root, 'datasets/data_util.py')) + model_saving_dir + '/' + 'data_util.py.backup')
        os.system('cp %s '%(os.path.join(root, 'datasets/ddd17_dataset.py')) + model_saving_dir + '/' + 'ddd17_dataset.py.backup')
        os.system('cp %s '%(os.path.join(root, 'datasets/extract_data_tools/DSEC/sequence.py')) + model_saving_dir + '/' + 'sequence.py.backup')

    def model_summary(self):

        self.printer.cprint(str(self.model))
        # dummy_input = torch.randn(16, self.cfg['DATASET']['nr_bins'], 440, 640).to(self.cfg['DEVICE'])
        # dummy_input2 = torch.randn(16, self.cfg['DATASET']['img_chnls'], 440, 640).to(self.cfg['DEVICE'])
        dummy_input = torch.randn(1, self.cfg['DATASET']['nr_bins'], 200, 346).to(self.cfg['DEVICE'])
        dummy_input2 = torch.randn(1, self.cfg['DATASET']['img_chnls'], 200, 346).to(self.cfg['DEVICE'])
        flops, params = profile(self.model, (dummy_input, dummy_input2))
        print('Number of parameters: %.2fM'%(params/1e6))
        print('Number of GFLOPs: %.2fG'%(flops/1e9))
        # Model Summary
        self.printer.cprint('Number of parameters: %.2fM'%(params/1e6))
        self.printer.cprint('Number of GFLOPs: %.2fG'%(flops/1e9))
