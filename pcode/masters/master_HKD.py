# -*- coding: utf-8 -*-
import copy
import functools
import itertools
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import distributed as dist

import pcode.create_aggregator as create_aggregator
import pcode.create_coordinator as create_coordinator
import pcode.master_utils as master_utils
import pcode.utils.checkpoint as checkpoint
from pcode import create_dataset, create_optimizer
from pcode.masters.master_base import MasterBase
from pcode.utils.early_stopping import EarlyStoppingTracker
from pcode.utils.tensor_buffer import TensorBuffer


# 定义MasterpFedSD类，继承自MasterBase类
class MasterHKD(MasterBase):
    def __init__(self, conf):
        # 调用父类MasterBase的构造函数，进行基础的初始化操作
        super().__init__(conf)
        assert self.conf.fl_aggregate is not None

        # 定义字典self.local_models用于存储本地模型，通过使用deepcopy来从master_model中复制
        self.local_models = dict(
            (
                client_id,
                copy.deepcopy(self.master_model)
            )
            for client_id in range(1, 1 + conf.n_clients)
        )

        # 定义字典self.personalized_global_global用于存储全局模型，通过使用deepcopy来从master_model中复制
        self.personalized_global_models = dict(
            (
                client_id,
                copy.deepcopy(self.master_model)
            )
            for client_id in range(1, 1 + conf.n_clients)
        )

        # 记录日志，表示主进程正在初始化本地模型
        conf.logger.log(f"Master initialize the local_models")

        # 初始化参数self.M，用于控制一些算法的操作
        self.M = 1
        # selected clients'id from begin
        # 定义集合self.activated_ids来存储被选中客户端的ID
        self.activated_ids = set()
        # self.is_cluster = False
        # 定义布尔值self.is_part_update用于判断是否进行部分模型更新
        self.is_part_update = False

        # 若需要进行部分模型更新
        if self.is_part_update:
            # 如果conf.data中包含'cifar'，则根据权重键索引生成self.head
            if 'cifar' in conf.data:
                self.head = [self.master_model.weight_keys[i] for i in [2]]
            # 使用itertools.chain.from_iterable将self.head展平为一维列表，并将结果赋值给self.head
            self.head = list(itertools.chain.from_iterable(self.head))

        # cluster
        # if self.is_cluster:
        #     self.K = 1 # 0...k-1
        #     self.quasi_global_models = dict(
        #         (
        #             client_id,
        #             copy.deepcopy(self.master_model)
        #         )
        #         for client_id in range(self.K)
        #     )

        # save arguments to disk.
        # 将配置参数保存在磁盘
        conf.is_finished = False
        checkpoint.save_arguments(conf)

    # 定义run方法，用于执行分布式联邦学习的主要逻辑
    def run(self):
        flatten_local_models = None
        # 使用循环迭代来执行联邦学习的通讯轮次
        for comm_round in range(1, 1 + self.conf.n_comm_rounds):
            # 将当前通信轮次comm_round保存到配置参数self.conf.graph.comm_round中
            self.conf.graph.comm_round = comm_round
            # 记录日志，用于记录当前正在执行的轮次
            self.conf.logger.log(
                f"Master starting one round of federated learning: (comm_round={comm_round})."
            )

            # get random n_local_epochs.
            # 从master_utils模块中获取随机的本地训练轮次列表list_of_local_n_epochs，用于控制每个客户端的本地训练轮次
            list_of_local_n_epochs = master_utils.get_n_local_epoch(
                conf=self.conf, n_participated=self.conf.n_participated
            )
            self.list_of_local_n_epochs = list_of_local_n_epochs

            # random select clients from a pool.
            # 从可选的客户端池中随机选择一部分客户端，返回选中客户端的ID列表
            selected_client_ids = self._random_select_clients()
            # 若启用部分模型更新，则更新个性化全局模型
            if self.is_part_update:
                self._update_personalized_global_models(selected_client_ids)   # partitial update
            
            # detect early stopping.
            # self._check_early_stopping()

            # init the activation tensor and broadcast to all clients (either start or stop).
            # 调用_activate_selected_clients函数来初始化激活状态张量并广播给所有选中的客户端，以指示客户端是否开始或停止训练
            self._activate_selected_clients(
                selected_client_ids, self.conf.graph.comm_round, list_of_local_n_epochs
            )

            # will decide to send the model or stop the training.
            # 根据是否完成训练的标志来决定是否向选中的客户端发送模型或停止训练
            if not self.conf.is_finished:
                # broadcast the model to activated clients.
                self._send_model_to_selected_clients(selected_client_ids)
            else:
                dist.barrier()
                self.conf.logger.log(
                    f"Master finished the federated learning by early-stopping: (current comm_rounds={comm_round},"
                    f" total_comm_rounds={self.conf.n_comm_rounds})"
                )
                return
            # wait to receive the local models.
            # 等待接收来自选中客户端的本地模型，将其展平为flatten_local_models
            flatten_local_models = self._receive_models_from_selected_clients(
                selected_client_ids
            )
            # 将选中的clients添加到activated_ids集合中
            self.activated_ids.update(selected_client_ids)

            # 根据selected_client_ids选择相应的本地模型并使用update_local_models函数进行更新
            self.update_local_models(selected_client_ids, flatten_local_models)

            # aggregate the local models and evaluate on the validation dataset.
            # 聚合本地模型并在验证数据集上进行评估
            self._aggregate_model(selected_client_ids)
            # 如果启用了个性化测试，则调用函数self._update_client_performance来更新客户端性能
            if self.conf.personal_test:
                self._update_client_performance(selected_client_ids)
            # 记录当前通讯轮次的完成，并进入下一循环轮次
            self.conf.logger.log(f"Master finished one round of federated learning.\n")
            
        # formally stop the training (the master has finished all communication rounds).
        # 通过调用dist.barrier()等待所有进程完成并同步
        dist.barrier()
        # 当主节点完成所有通信轮次后，通过调用self._finishing()正式停止训练
        self._finishing()

    # 定义_activate_selected_clients()方法，用于激活选中的客户端并向它们发送激活信息
    def _activate_selected_clients(
        self, selected_client_ids, comm_round, list_of_local_n_epochs
    ):
        # Activate the selected clients:
        # the first row indicates the client id,
        # the second row indicates the current_comm_round,
        # the third row indicates the expected local_n_epochs

        # 将selected_client_ids列表转换为NumPy数组
        selected_client_ids = np.array(selected_client_ids)

        # 创建一个大小为（4, len(selected_client_ids）的张量activation_msg
        activation_msg = torch.zeros((4, len(selected_client_ids)))
        # 其中第一行表示客户端ID
        activation_msg[0, :] = torch.Tensor(selected_client_ids)
        # 第二行表示当前通讯轮次
        activation_msg[1, :] = comm_round
        # 第三行表示预期的本地训练轮次
        activation_msg[2, :] = torch.Tensor(list_of_local_n_epochs)

        # 通过遍历选中的客户端 ID，检查每个客户端是否已经在之前的通信轮次中激活过，
        # 将激活状态（1表示已激活，0表示未激活）存储在 is_activate_before 列表中，并将其放入 activation_msg 的第四行
        is_activate_before = [1 if id in self.activated_ids else 0 for id in selected_client_ids]      
        activation_msg[3, :] = torch.Tensor(is_activate_before)

        # 使用broadcast方法将张量从源进程广播到所有进程，使所有选中的客户端都能收到激活信息
        dist.broadcast(tensor=activation_msg, src=0)
        # 记录日志信息表明主节点已经激活了选中的客户端
        self.conf.logger.log(f"Master activated the selected clients.")
        # 使用 dist.barrier() 进行进程同步，以确保所有进程在激活消息发送完毕后继续执行后续操作
        dist.barrier()

    # 定义_send_model_to_selected_clients()方法，实现server向选中的client发送模型
    def _send_model_to_selected_clients(self, selected_client_ids):
        # the master_model can be large; the client_models can be small and different.
        # 日志记录主节点正在将模型发给工作节点
        self.conf.logger.log(f"Master send the models to workers.")
        # 循环遍历client的排名和被选择client的ID，enumerate函数用于同时获取元素的索引和值
        for worker_rank, selected_client_id in enumerate(selected_client_ids, start=1):
            # 根据is_part_update的值，确定要发送的全局模型的状态字典
            # 如果is_part_update为False——不使用局部更新，则使用主模型（master_model）的状态字典
            if not self.is_part_update:
                global_model_state_dict = self.master_model.state_dict()
            # 若使用局部更新，则使用与选中的客户端 ID 相对应的个性化全局模型（personalized_global_models）的状态字典
            else:
                global_model_state_dict = self.personalized_global_models[selected_client_id].state_dict()

            # 创建一个TensorBuffer对象flatten_model,将全局模型的参数值转换为一维向量
            flatten_model = TensorBuffer(list(global_model_state_dict.values()))
            # 使用dist.send()方法将该一维张量发送给工作节点worker_rank
            dist.send(tensor=flatten_model.buffer, dst=worker_rank)
            # 日志记录master已经将全局模型发送给了第worker_rank个client
            self.conf.logger.log(
                f"\tMaster send the global model to process_id={worker_rank}."
            )

            # 如果选中的客户端 ID 在已激活的客户端 ID 集合中（即已在之前的通信轮次中激活过），则继续将本地模型发送给工作节点
            if selected_client_id in self.activated_ids:
                # send local model
                # 获取与选中的客户端 ID 相对应的本地模型的状态字典（local_models）
                local_model_state_dict = self.local_models[selected_client_id].state_dict()
                # 将参数值转换为一维向量
                flatten_local_model = TensorBuffer(list(local_model_state_dict.values()))
                # 使用dist.send()方法将该张量发送给工作节点
                dist.send(tensor=flatten_local_model.buffer, dst=worker_rank)
                # 日志记录
                self.conf.logger.log(
                    f"\tMaster send local models to process_id={worker_rank}."
                )
        dist.barrier()

    # 定义_aggregate_model()方法，用于在server中聚合被选中client的模型
    def _aggregate_model(self, selected_client_ids):
        # 调用 _average_model 方法对选中客户端的模型进行平均处理，
        # 并使用 load_state_dict 方法将平均后的模型参数加载到主模型（master_model）中。这样，主模型将更新为选中客户端模型的平均值
        self.master_model.load_state_dict(self._average_model(selected_client_ids).state_dict())

    # 定义_update_personalized_global_models()方法，用于更新个性化全局模型
    def _update_personalized_global_models(self, selected_client_ids):
        # 如果当前通信轮次为1，即首轮通信，那么不进行个性化全局模型的更新，直接返回
        if self.conf.graph.comm_round == 1:
            return
        # 获取主模型（master_model）的参数字典
        w_master = self.master_model.state_dict()
        # 对于每一个被选中的client
        for id in selected_client_ids:
            # 获取本地模型（local_model）的参数字典
            w_local = self.local_models[id].state_dict()
            # 在个性化全局模型字典中创建副本（w_personalized_global）
            w_personalized_global = copy.deepcopy(w_master)
            # 根据本地模型字典（w_local）的参数更新副本中与头部（self.head）中的键对应的参数
            for key in w_local.keys():
                if key in self.head:
                    w_personalized_global[key] = w_local[key]
            # 使用 load_state_dict 方法将更新后的副本加载到对应的个性化全局模型（personalized_global_models）中
            self.personalized_global_models[id].load_state_dict(w_personalized_global)

    # 定义_average_model()方法，用于计算所选中client模型的平均值
    def _average_model(self, client_ids, weights=None):
        # 首先创建一个和主模型结构相同的副本_model_avg，并将其参数初始化为0
        _model_avg = copy.deepcopy(self.master_model)
        for param in _model_avg.parameters():
            param.data = torch.zeros_like(param.data)

        # 遍历选中客户端的ID，逐个对应参数进行累加操作
        for id in client_ids:
            # 通过zip函数将_model_avg和对应客户端模型的参数逐层进行对应，然后将对应参数的数据值累加到_model_avg的参数上，再除以客户端数量，以实现平均操作
            for avg_param, client_param in zip(_model_avg.parameters(), self.local_models[id].parameters()):
                avg_param.data += client_param.data.clone() / len(client_ids)
        # 返回计算得到的平均模型 _model_avg。该模型的参数值是选中客户端模型参数的平均值，用于在主模型中更新参数。
        return _model_avg

    # 定义_update_client_performance()方法，用于更新所选中的client的性能指标
    def _update_client_performance(self, selected_client_ids):
        # get client model best performance on personal test distribution
        # 首先判断当前训练轮数是第一轮还是后续轮
        if self.conf.graph.comm_round == 1:
            # 如果是第一轮则测试所有客户端的性能指标
            test_client_ids = self.client_ids
        # 否则，只测试所选中的客户端
        else:
            test_client_ids = selected_client_ids
        # 遍历测试客户端的ID，对每一个客户端进行个性化测试
        for client_id in test_client_ids:
            # 调用函数do_validation_personal()对客户端进行个性化验证，并将验证结果保存在self.curr_personal_perfs 字典中，以客户端 ID 作为键
            self.curr_personal_perfs[client_id] = master_utils.do_validation_personal(
                self.conf,
                self.client_coordinators[client_id],
                self.local_models[client_id],
                self.criterion,
                self.metrics,
                [self.local_test_loaders[client_id]],
                label=f"local_test_loader_client_{client_id}",
            )
        # 调用 _compute_best_mean_client_performance 方法计算并更新最佳的客户端性能指标
        self._compute_best_mean_client_performance() 

    # 定义update_local_models函数，用于根据提供的扁平化本地模型参数，更新对应客户端的本地模型
    def update_local_models(self, selected_client_ids, flatten_local_models):
        # 使用master_utils.recover_models函数从扁平化的本地模型参数中恢复本地模型
        _, local_models = master_utils.recover_models(
            self.conf, self.client_models, flatten_local_models
        )
        # 对于每个选定的客户端ID，将恢复的本地模型加载到self.local_models中相应的客户端模型中
        for selected_client_id in selected_client_ids:
           self.local_models[selected_client_id].load_state_dict(local_models[selected_client_id].state_dict())

    # 定义_check_early_stopping函数，用于检查是否需要早停
    def _check_early_stopping(self):
        # 初始化meet_flag为False
        meet_flag = False

        # consider both of target_perf and early_stopping
        # 根据配置参数和当前性能情况，判断是否满足早停的条件
        # 如果同时设置了目标性能（target_perf）和早停机制，则判断是否满足目标性能要求
        if self.conf.target_perf is not None:
            assert 100 >= self.conf.target_perf > 0

            # meet the target perf.
            # 如果满足目标性能要求，则将meet_flag设置为True
            if (
                self.coordinator.key_metric.cur_perf is not None
                and self.coordinator.key_metric.cur_perf > self.conf.target_perf
            ):
                self.conf.logger.log("Master early stopping: meet target perf.")
                self.conf.meet_target = True
                meet_flag = True
            # or has no progress and early stop it.
            # 如果未满足目标性能要求但达到了早停条件，则同样将meet_flag设置为True
            elif self.early_stopping_tracker(self.coordinator.key_metric.cur_perf):
                self.conf.logger.log(
                    "Master early stopping: not meet target perf but has no patience."
                )
                meet_flag = True
        # only consider the early stopping.
        # 如果只考虑早停机制而不设置目标性能，则根据早停条件判断是否满足早停要求。如果满足早停要求，则将meet_flag设置为True
        else:
            if self.early_stopping_tracker(self.coordinator.key_metric.cur_perf):
                meet_flag = True

        # 若满足早停条件
        if meet_flag:
            # we perform the early-stopping check:
            # (1) before the local training and (2) after the update of the comm_round.
            # 记录当前通讯轮数的上一轮通讯轮数为_comm_round
            _comm_round = self.conf.graph.comm_round - 1
            # 将当前通讯轮数设置为-1，表示停止通讯论数的更新
            self.conf.graph.comm_round = -1
            # 调用_finish函数进行早停
            self._finishing(_comm_round)


# 定义函数get_model_diff，用于计算两个模型之间的差异
def get_model_diff(model1, model2):
    params_dif = []
    # 遍历两个模型的参数，计算参数之间的差异，并将差异值存储在params_dif列表中。
    for param_1, param_2 in zip(model1.parameters(), model2.parameters()):
        params_dif.append((param_1 - param_2).view(-1))
    # 然后，将params_dif连接起来，并计算其范数作为模型之间的差异值，并将其作为结果返回
    params_dif = torch.cat(params_dif)
    return torch.norm(params_dif).item()

