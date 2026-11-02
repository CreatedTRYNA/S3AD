import os
from argparse import Namespace
from timm.data.constants import IMAGENET_DEFAULT_MEAN
from timm.data.constants import IMAGENET_DEFAULT_STD
import torchvision.transforms.functional as F

from configs.__base__ import *


class cfg(cfg_common, cfg_dataset_default):

	def __init__(self):
		# super(cfg, self).__init__()
		cfg_common.__init__(self)
		cfg_dataset_default.__init__(self)

		self.fvcore_b = 1
		self.fvcore_c = 3
		self.seed = 42
		self.size = 256
		self.epoch_full = 1000
		self.warmup_epochs = 3
		self.test_start_epoch = self.epoch_full
		self.test_per_epoch = 20
		self.batch_train = 20  # official 16
		self.batch_test_per = 20
		self.lr = 0.001 * self.batch_train / 20
		# self.lr = 0.001
		self.weight_decay = 0.0001
		self.metrics = [
			'mAUROC_sp_max', 'mAP_sp_max', 'mF1_max_sp_max',
			'mAUROC_px', 'mAP_px', 'mF1_max_px', 'mAUPRO_px',
			'mF1_px_0.2_0.8_0.1', 'mAcc_px_0.2_0.8_0.1', 'mIoU_px_0.2_0.8_0.1', 'mIoU_max_px',
		]

		# ==> data
		self.data.type = 'MPDDDataset'
		self.data.root = os.path.join(os.environ.get('S3AD_DATA_ROOT', 'datasets'), 'MPDD')
		self.data.views = []
		self.data.use_sample = False
		self.data.cls_names = ['bracket_black', 'bracket_brown', 'bracket_white', 'connector', 'metal_plate', 'tubes']

		self.data.train_transforms = [
			dict(type='Resize', size=(self.size, self.size), interpolation=F.InterpolationMode.BILINEAR),
			dict(type='CenterCrop', size=(self.size, self.size)),
			dict(type='ToTensor'),
			dict(type='Normalize', mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, inplace=True),
		]
		self.data.test_transforms = self.data.train_transforms
		self.data.target_transforms = [
			dict(type='Resize', size=(self.size, self.size), interpolation=F.InterpolationMode.BILINEAR),
			dict(type='CenterCrop', size=(self.size, self.size)),
			dict(type='ToTensor'),
		]
				
		# 提速：为 test 的异常图高斯平滑设置较小 sigma（训练器读取此值）
		self.gaussian_sigma_test = 4
		self.trainer.data.prefetch_factor = 8
		self.trainer.data.preload_test_to_ram = True
		self.trainer.data.pretransform_to_tensor = True
		# ==> modal
		self.setup_teacher_model('resnet34')  # 默认使用ResNet34，可以改为'clip_rn50'  'resnet34'

	def setup_teacher_model(self, teacher_type='resnet34'):
		"""设置teacher模型配置"""
		self.model_t = Namespace()
		self.model_s = dict(depths_decoder=[3, 9, 9, 7])
		
		if teacher_type == 'resnet34':
			# 原始ResNet34配置
			self.model_t.name = 'timm_resnet34'
			self.model_t.kwargs = dict(
				pretrained=False, 
				checkpoint_path='model/pretrain/resnet34-43635321.pth',
				strict=False, 
				features_only=True, 
				out_indices=[1, 2, 3]
			)
			print("配置ResNet34作为teacher网络")
			
		elif teacher_type == 'clip_rn50':
			# CLIP ResNet50配置
			self.model_t.name = 'clip_resnet50'
			self.model_t.kwargs = dict(
				pretrained=False,
				checkpoint_path='',  # CLIP权重会自动下载
				model_name='RN50',
				pretrained_dataset='openai',  # 或 'laion2b_s34b_b79k'
				strict=False,
				features_only=True,
				out_indices=[1, 2, 3]
			)
			print("配置CLIP ResNet50作为teacher网络")
			
		elif teacher_type == 'timm_resnet50':
			# 标准ResNet50配置
			self.model_t.name = 'timm_resnet50'
			self.model_t.kwargs = dict(
				pretrained=True,
				checkpoint_path='',
				strict=False,
				features_only=True,
				out_indices=[1, 2, 3]
			)
			print("配置标准ResNet50作为teacher网络")
			
		else:
			raise ValueError(f"不支持的teacher类型: {teacher_type}")
		
		# 设置S3AD模型配置
		self.model = Namespace()
		self.model.name = 'S3AD'
		self.model.kwargs = dict(
			pretrained=False, 
			checkpoint_path='', 
			strict=True, 
			model_t=self.model_t,
			model_s=self.model_s
		)

		# ==> evaluator
		self.evaluator.kwargs = dict(metrics=self.metrics, pooling_ks=None, max_step_aupro=100, use_adeval=False)
		# self.evaluator.kwargs = dict(metrics=self.metrics, pooling_ks=[16, 16], max_step_aupro=100)

		# ==> optimizer
		self.optim.lr = self.lr
		self.optim.kwargs = dict(name='adamw', betas=(0.9, 0.999), eps=1e-8, weight_decay=self.weight_decay, amsgrad=False)

		# ==> trainer
		self.trainer.name = 'S3ADMDPPTrainer'
		# 内存感知落盘参数（可调）
		self.trainer.mem_spill = Namespace(enabled=True, threshold_gb=200, min_interval_batches=10)
		self.trainer.logdir_sub = ''
		self.trainer.resume_dir = ''
		self.trainer.epoch_full = self.epoch_full
		self.trainer.scheduler_kwargs = dict(
			name='step', lr_noise=None, noise_pct=0.67, noise_std=1.0, noise_seed=42, lr_min=self.lr / 1e2,
			warmup_lr=self.lr / 1e3, warmup_iters=-1, cooldown_iters=0, warmup_epochs=self.warmup_epochs, cooldown_epochs=0, use_iters=True,
			patience_iters=0, patience_epochs=0, decay_iters=0, decay_epochs=int(self.epoch_full * 0.8), cycle_decay=0.1, decay_rate=0.1)
		self.trainer.mixup_kwargs = dict(mixup_alpha=0.8, cutmix_alpha=1.0, cutmix_minmax=None, prob=0.0, switch_prob=0.5, mode='batch', correct_lam=True, label_smoothing=0.1)
		self.trainer.test_start_epoch = self.test_start_epoch
		self.trainer.test_per_epoch = self.test_per_epoch

		self.trainer.data.batch_size = self.batch_train
		self.trainer.data.batch_size_per_gpu_test = self.batch_test_per
		# DataLoader settings for MPDD (speed oriented)
		self.trainer.data.num_workers_per_gpu = 8
		self.trainer.data.pin_memory = True
		self.trainer.data.persistent_workers = True

		# ==> loss
		self.loss.loss_terms = [
			dict(type='L2Loss', name='pixel', lam=1.0),
		]

		# ==> logging
		self.logging.log_terms_train = [
			dict(name='batch_t', fmt=':>5.3f', add_name='avg'),
			dict(name='data_t', fmt=':>5.3f'),
			dict(name='optim_t', fmt=':>5.3f'),
			dict(name='lr', fmt=':>7.6f'),
			dict(name='pixel', suffixes=[''], fmt=':>5.3f', add_name='avg'),
			dict(name='cos', suffixes=[''], fmt=':>5.3f', add_name='avg'),
		]
		self.logging.log_terms_test = [
			dict(name='batch_t', fmt=':>5.3f', add_name='avg'),
			dict(name='pixel', suffixes=[''], fmt=':>5.3f', add_name='avg'),
			dict(name='cos', suffixes=[''], fmt=':>5.3f', add_name='avg'),
		]

