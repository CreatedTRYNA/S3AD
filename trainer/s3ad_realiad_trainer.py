import os
import glob
import shutil
import time
import copy

import tabulate
import torch
import numpy as np

from util.util import makedirs, log_cfg, able, log_msg, get_log_terms, update_log_term
from util.net import get_timepc, reduce_tensor
from ._base_trainer import BaseTrainer
from . import TRAINER

try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as ApexDDP
    from apex.parallel import convert_syncbn_model as ApexSyncBN
except:
    from timm.layers.norm_act import convert_sync_batchnorm as ApexSyncBN


@TRAINER.register_module
class S3ADRealiADTrainer(BaseTrainer):
    def __init__(self, cfg):
        super(S3ADRealiADTrainer, self).__init__(cfg)

    def set_input(self, inputs):
        # 非阻塞搬运 + channels_last
        self.imgs = inputs['img'].to('cuda', non_blocking=True)
        try:
            self.imgs = self.imgs.to(memory_format=torch.channels_last)
        except Exception:
            pass
        self.imgs_mask = inputs['img_mask'].to('cuda', non_blocking=True)
        self.cls_name = inputs['cls_name']
        self.anomaly = inputs['anomaly']
        self.bs = self.imgs.shape[0]

    def forward(self):
        self.feats_t, self.feats_s = self.net(self.imgs)

    # 新增：实现优化步骤，确保权重随训练更新
    def optimize_parameters(self):
        if self.mixup_fn is not None:
            self.imgs, _ = self.mixup_fn(self.imgs, torch.ones(self.imgs.shape[0], device=self.imgs.device))
        with self.amp_autocast():
            self.forward()
            loss_mse = self.loss_terms['pixel'](self.feats_t, self.feats_s)
        self.backward_term(loss_mse, self.optim)
        update_log_term(self.log_terms.get('pixel'),
                        reduce_tensor(loss_mse, self.world_size).clone().detach().item(),
                        1, self.master)

    @torch.inference_mode()
    def test(self):
        # 说明：该 test 方法为 Real-IAD 数据专用的低内存实现
        # 关键策略：
        # - 不在内存中累计全量结果；按批次写入磁盘（每类分别维护一个结果文件列表）
        # - 采用两阶段：阶段1写出所有批次的 amaps/masks/labels；阶段2逐类读取与拼接到 numpy，再调用 Evaluator.run
        # - 在分布式下，每个 rank 写自身文件，master 负责收集聚合后评估
        if self.master:
            if os.path.exists(self.tmp_dir):
                shutil.rmtree(self.tmp_dir)
            os.makedirs(self.tmp_dir, exist_ok=True)
        self.reset(isTrain=False)

        # 第一阶段：遍历测试集，计算并按批次落盘，避免内存累积
        batch_idx = 0
        # 更稳健的长度：使用实际 loader 长度（兼容 dict/多测试集可在外层控制）
        test_length = len(self.test_loader)
        test_iter = iter(self.test_loader)

        # 每个 rank 的中间结果目录
        rank_dir = os.path.join(self.tmp_dir, f"rank_{self.rank}")
        if self.master:
            os.makedirs(rank_dir, exist_ok=True)
        else:
            # 非 master 也需要目录
            os.makedirs(rank_dir, exist_ok=True)

        while batch_idx < test_length:
            t1 = get_timepc()
            batch_idx += 1
            test_data = next(test_iter)
            self.set_input(test_data)
            # 使用 autocast 加速前向与异常图计算
            with self.amp_autocast():
                self.forward()
                loss_mse = self.loss_terms['pixel'](self.feats_t, self.feats_s)
            update_log_term(self.log_terms.get('pixel'),
                            reduce_tensor(loss_mse, self.world_size).clone().detach().item(),
                            1, self.master)

            # 计算 anomaly maps（按批次，仅保留 numpy 后写盘）
            gaussian_sigma = getattr(self.cfg, 'gaussian_sigma_test', 4)
            anomaly_map, _ = self.evaluator.cal_anomaly_map(
                self.feats_t, self.feats_s,
                [self.imgs.shape[2], self.imgs.shape[3]],
                uni_am=False,
                amap_mode='add',
                gaussian_sigma=gaussian_sigma
            )
            # 二值化 mask 并转为 numpy
            self.imgs_mask[self.imgs_mask > 0.5], self.imgs_mask[self.imgs_mask <= 0.5] = 1, 0
            masks_np = self.imgs_mask.cpu().numpy().astype(int)
            # 预测图与标签
            amaps_np = anomaly_map  # already numpy
            cls_np = np.array(self.cls_name)
            anom_np = self.anomaly.cpu().numpy().astype(int)

            # 按 batch 写出一个文件，包含本批的所有样本（降低文件数量，同时避免大拼接）
            # 文件名包含 batch 序号与 rank，便于后续收集
            out_file = os.path.join(rank_dir, f"batch_{batch_idx:06d}.pth")
            torch.save(
                {
                    'imgs_masks': masks_np,
                    'anomaly_maps': amaps_np,
                    'cls_names': cls_np,
                    'anomalys': anom_np,
                },
                out_file,
                _use_new_zipfile_serialization=False
            )

            # 及时释放中间张量，进一步缓解峰值内存
            del masks_np, amaps_np, cls_np, anom_np

            t2 = get_timepc()
            update_log_term(self.log_terms.get('batch_t'), t2 - t1, 1, self.master)
            if self.master:
                print(f'\r{batch_idx}/{test_length}', end='')
                if batch_idx % self.cfg.logging.test_log_per == 0 or batch_idx == test_length:
                    msg = able(self.progress.get_msg(batch_idx, test_length, 0, 0, prefix='Test'), self.master, None)
                    log_msg(self.logger, msg)

        # 分布式：等待所有 rank 完成写盘后，master 收集
        if self.cfg.dist:
            # 每个 rank 写出一个完成标记文件
            done_flag = os.path.join(rank_dir, 'DONE')
            with open(done_flag, 'w') as f:
                f.write('done')

            if self.master:
                # 轮询直到所有 rank 的 DONE 出现
                all_done = False
                while not all_done:
                    rank_done = [
                        os.path.exists(os.path.join(self.tmp_dir, f"rank_{r}", 'DONE'))
                        for r in range(self.world_size)
                    ]
                    if all(rank_done):
                        all_done = True
                    else:
                        time.sleep(1)

                # 收集所有 rank 的批文件路径
                rank_batch_files = []
                for r in range(self.world_size):
                    rdir = os.path.join(self.tmp_dir, f"rank_{r}")
                    files = sorted(glob.glob(os.path.join(rdir, 'batch_*.pth')))
                    rank_batch_files.extend(files)
        else:
            # 非分布式，直接使用当前 rank 目录
            rank_batch_files = sorted(glob.glob(os.path.join(rank_dir, 'batch_*.pth')))

        # 第二阶段：逐类读取批文件并评估，避免一次性加载全量到内存
        if self.master:
            # 预分发：将每个 batch 文件按类别拆分到 per_class 目录，避免为每个类别重复扫描所有 batch 文件
            per_class_dir = os.path.join(self.tmp_dir, 'per_class')
            os.makedirs(per_class_dir, exist_ok=True)
            for bf in rank_batch_files:
                data = torch.load(bf, weights_only=False)
                # 针对当前批次中的每个类别，将其子集写到 per_class/{cls}/ 下
                unique_cls = np.unique(data['cls_names'])
                for cls in unique_cls:
                    mask = data['cls_names'] == cls
                    if np.any(mask):
                        cls_dir = os.path.join(per_class_dir, str(cls))
                        os.makedirs(cls_dir, exist_ok=True)
                        out_file = os.path.join(cls_dir, os.path.basename(bf))
                        torch.save(
                            {
                                'imgs_masks': data['imgs_masks'][mask],
                                'anomaly_maps': data['anomaly_maps'][mask],
                                'cls_names': data['cls_names'][mask],
                                'anomalys': data['anomalys'][mask],
                            },
                            out_file,
                            _use_new_zipfile_serialization=False
                        )

            # 准备指标记录与汇总消息
            msg = {}

            # 取训练数据集的类别列表供迭代（BaseTrainer 在 reset 时已设置）
            # 与其它 trainer 保持一致，使用 self.cls_names 作为评估类别顺序
            for idx, cls_name in enumerate(self.cls_names):
                # 为当前类别，收集所有批次中属于该类别的样本，并按 Evaluator.run 期望的结构拼接
                cls_imgs_masks = []
                cls_anomaly_maps = []
                cls_cls_names = []
                cls_anomalys = []

                # 仅扫描该类别目录下的文件，避免 O(C*B) 次重复 I/O
                class_files = sorted(glob.glob(os.path.join(per_class_dir, str(cls_name), 'batch_*.pth')))
                for cf in class_files:
                    data = torch.load(cf, weights_only=False)
                    # 直接追加（文件已按类别拆分，无需再筛选）
                    cls_imgs_masks.append(data['imgs_masks'])
                    cls_anomaly_maps.append(data['anomaly_maps'])
                    cls_cls_names.append(data['cls_names'])
                    cls_anomalys.append(data['anomalys'])

                # 若该类没有样本，跳过评估并提示
                if len(cls_cls_names) == 0:
                    log_msg(self.logger, f"Warning: No test data found for class '{cls_name}', skipping evaluation.")
                    continue

                # 拼接为 Evaluator.run 所需的 results 字典（仅该类的数据，避免全量内存）
                results_cls = {
                    'imgs_masks': np.concatenate(cls_imgs_masks, axis=0),
                    'anomaly_maps': np.concatenate(cls_anomaly_maps, axis=0),
                    'cls_names': np.concatenate(cls_cls_names, axis=0),
                    'anomalys': np.concatenate(cls_anomalys, axis=0),
                }

                # 评估当前类指标
                metric_results = self.evaluator.run(results_cls, cls_name, self.logger)

                # 汇总日志输出，保持与原 OmniadTrainer 一致的表格结构与 recorder 行为
                msg['Name'] = msg.get('Name', [])
                msg['Name'].append(cls_name)
                avg_act = True if len(self.cls_names) > 1 and idx == len(self.cls_names) - 1 else False
                if avg_act:
                    msg['Name'].append('Avg')

                for metric in self.metrics:
                    metric_result = metric_results[metric] * 100
                    self.metric_recorder[f'{metric}_{cls_name}'].append(metric_result)
                    max_metric = max(self.metric_recorder[f'{metric}_{cls_name}'])
                    max_metric_idx = self.metric_recorder[f'{metric}_{cls_name}'].index(max_metric) + 1

                    msg[metric] = msg.get(metric, [])
                    msg[metric].append(metric_result)

                    msg[f'{metric} (Max)'] = msg.get(f'{metric} (Max)', [])
                    msg[f'{metric} (Max)'].append(f'{max_metric:.3f} ({max_metric_idx:<3d} epoch)')

                    if avg_act:
                        metric_result_avg = sum(msg[metric]) / len(msg[metric])
                        self.metric_recorder[f'{metric}_Avg'].append(metric_result_avg)
                        max_metric_avg = max(self.metric_recorder[f'{metric}_Avg'])
                        max_metric_idx_avg = self.metric_recorder[f'{metric}_Avg'].index(max_metric_avg) + 1
                        msg[metric].append(metric_result_avg)
                        msg[f'{metric} (Max)'].append(f'{max_metric_avg:.3f} ({max_metric_idx_avg:<3d} epoch)')

            if len(msg) > 0:
                table = tabulate.tabulate(msg, headers='keys', tablefmt='pipe', floatfmt='.3f', numalign='center', stralign='center')
                log_msg(self.logger, f"\n{table}")

            # 清理中间结果避免占盘
            try:
                shutil.rmtree(self.tmp_dir)
            except Exception:
                pass