# -*- coding: utf-8 -*-
import functools

import torch
import torch.nn.functional as F

import pcode.create_dataset as create_dataset
import pcode.datasets.mixup_data as mixup
import pcode.utils.checkpoint as checkpoint
from pcode.utils.logging import dispaly_best_test_stat, display_test_stat
from pcode.utils.mathdict import MathDict
from pcode.utils.stat_tracker import RuntimeTracker


# inference函数用于在给定的模型上进行推断并获取损失和准确度
def inference(
    conf, model, criterion, metrics, data_batch, tracker=None, is_training=True
):
    """Inference on the given model and get loss and accuracy."""
    # do the forward pass and get the output.
    # 执行前向传播，得到模型的输出
    output = model(data_batch["input"])

    # evaluate the output and get the loss, performance.
    # 根据是否使用mixup和是否处于训练模式，选择不同的处理方式计算损失和准确度
    # 如果使用mixup且处于训练模式，则使用mixup_criterion函数计算混合损失，并分别计算目标target_a和target_b对应的性能
    if conf.use_mixup and is_training:
        loss = mixup.mixup_criterion(
            criterion,
            output,
            data_batch["target_a"],
            data_batch["target_b"],
            data_batch["mixup_lambda"],
        )

        performance_a = metrics.evaluate(loss, output, data_batch["target_a"])
        performance_b = metrics.evaluate(loss, output, data_batch["target_b"])
        performance = [
            data_batch["mixup_lambda"] * _a + (1 - data_batch["mixup_lambda"]) * _b
            for _a, _b in zip(performance_a, performance_b)
        ]
    # 否则，直接使用criterion函数计算损失，并计算目标target对应的性能
    else:
        loss = criterion(output, data_batch["target"])
        performance = metrics.evaluate(loss, output, data_batch["target"])

    # update tracker.
    # 如果提供了tracker对象，则更新指标跟踪器，将损失和性能添加到跟踪器中
    if tracker is not None:
        tracker.update_metrics(
            [loss.item()] + performance, n_samples=data_batch["input"].size(0)
        )
    # 将计算得到的损失和模型输出作为函数的输出返回
    return loss, output


def do_validation(
    conf,
    coordinator,
    model,
    criterion,
    metrics,
    data_loaders,
    performance=None,
    label=None,
):
    """Evaluate the model on the test dataset and save to the checkpoint."""
    # wait until the whole group enters this function, and then evaluate.
    conf.logger.log(f"Master enters the validation phase.")
    if performance is None:
        performance = get_avg_perf_on_dataloaders(
            conf, coordinator, model, criterion, metrics, data_loaders, label
        )

    # remember the best performance and display the val info.
    coordinator.update_perf(performance)
    dispaly_best_test_stat(conf, coordinator)

    # save to the checkpoint.
    conf.logger.log(f"Master finished the validation.")
    if not conf.train_fast:
        checkpoint.save_to_checkpoint(
            conf,
            {
                "arch": conf.arch,
                "current_comm_round": conf.graph.comm_round,
                "best_perf": coordinator.best_trackers["top1"].best_perf,
                "state_dict": model.state_dict(),
            },
            coordinator.best_trackers["top1"].is_best,
            dirname=conf.checkpoint_root,
            filename="checkpoint.pth.tar",
            save_all=conf.save_all_models,
        )
        conf.logger.log(f"Master saved to checkpoint.")


# do_validation_personal 函数用于在个性化测试数据集上评估模型的性能，并将结果保存到检查点
def do_validation_personal(
    # 该函数接受以下参数
    # 配置对象，包含训练和评估的参数配置
    conf,
    # 协调器对象，用于跟踪和管理评估的性能指标
    coordinator,
    # 要评估的模型
    model,
    # 评估指标的损失函数
    criterion,
    # 评估指标的度量标准
    metrics,
    # 数据加载器列表，包含个性化测试数据集的加载器
    data_loaders,
    # 先前的性能指标，初始化为None
    performance=None,
    # 标签，用于区分个性化数据集
    label=None,
):
    """Evaluate the model on the test dataset and save to the checkpoint."""
    # wait until the whole group enters this function, and then evaluate.
    # 首先记录日志，个性化验证阶段开始
    conf.logger.log(f"Personal validation phase for {label}.")
    # 如果未提供先前的性能指标，则调用 get_avg_perf_on_dataloaders 函数计算模型在个性化测试数据集上的性能指标
    if performance is None:
        performance = get_avg_perf_on_dataloaders(
            conf, coordinator, model, criterion, metrics, data_loaders, label
        )

    # remember the best performance and display the val info.
    # 更新协调器的性能指标，并显示最佳的测试统计信息
    coordinator.update_perf(performance)
    dispaly_best_test_stat(conf, coordinator)

    # 记录日志，指示个性化验证阶段结束
    conf.logger.log(f"Finished the personal validation for {label}.")

    # 返回最终性能指标
    return performance


# get_avg_perf_on_dataloaders 函数用于计算模型在多个数据加载器上的平均性能
def get_avg_perf_on_dataloaders(
        # 该函数接受以下参数
        # 配置对象，包含训练和评估的参数配置
        conf,
        # 协调器对象，用于跟踪和管理评估的性能指标
        coordinator,
        # 要评估的模型
        model,
        # 评估指标的损失函数
        criterion,
        # 评估指标的度量标准
        metrics,
        # 数据加载器列表，包含个性化测试数据集的加载器
        data_loaders,
        # 标签，用于区分个性化数据集
        label,
):
    # 打印日志，指示从多个数据加载器获取平均性能
    print(f"\tGet averaged performance from {len(data_loaders)} data_loaders.")
    # 初始化performance列表为空
    performance = []

    # 遍历每一个数据加载器，并使用validate 函数对模型在数据加载器上进行评估，将评估结果添加到performance列表中
    for idx, data_loader in enumerate(data_loaders):
        _performance = validate(
            conf,
            coordinator,
            model,
            criterion,
            metrics,
            data_loader,
            label=f"{label}-{idx}" if label is not None else "test_loader",
        )
        performance.append(MathDict(_performance))
    # 使用 functools.reduce 函数对性能指标进行求和，并除以加载器的数量，以得到平均性能
    performance = functools.reduce(lambda a, b: a + b, performance) / len(performance)
    # 返回计算得到的平均性能指标
    return performance


# validate() 函数是用于模型评估的函数，它对给定的数据集进行模型评估并返回评估指标的结果
def validate(
    # 配置对象，包含了模型评估所需的配置信息
    conf,
    # 协调器对象，用于跟踪评估指标的进展
    coordinator,
    # 待评估模型
    model,
    # 损失函数，用于计算模型的损失
    criterion,
    # 评估指标，用于计算模型的性能指标
    metrics,
    # 数据加载器，用于加载评估数据
    data_loader,
    # 用于标识评估过程的标签，默认为"test_loader"
    label="test_loader",
    # 是否显示评估结果，默认为True
    display=True,
):
    """A function for model evaluation."""
    if data_loader is None:
        return None

    # switch to evaluation mode.
    # 切换到评估模式
    model.eval()

    # place the model to the device.
    # 如果conf对象中的on_cuda标志为True，则将模型移动到GPU上
    if conf.graph.on_cuda:
        model = model.cuda()

    # evaluate on test_loader.
    # 创建一个RuntimeTracker对象，用于跟踪metrics.metric_names中指定的指标
    tracker_te = RuntimeTracker(metrics_to_track=metrics.metric_names)

    # 函数迭代遍历data_loader，加载数据批次并检查模型的性能。它使用create_dataset.load_data_batch函数加载数据批次，
    # 然后调用inference函数对数据批次进行推断
    # 结果由tracker_te对象进行跟踪
    for _input, _target in data_loader:
        # load data and check performance.
        # 加载数据并检查性能
        data_batch = create_dataset.load_data_batch(
            conf, _input, _target, is_training=False
        )

        with torch.no_grad():
            inference(
                conf,
                model,
                criterion,
                metrics,
                data_batch,
                tracker_te,
                is_training=False,
            )

    # place back model to the cpu.
    # 如果conf对象中的on_cuda标志为True，则将模型移回到CPU上
    if conf.graph.on_cuda:
        model = model.cpu()

    # display the test stat.
    # 调用tracker_te对象作为函数，根据跟踪的指标计算性能
    perf = tracker_te()
    # 如果label参数不为None，则调用display_test_stat函数显示测试统计信息
    if label is not None:
        display_test_stat(conf, coordinator, tracker_te, label)
    # 如果display标志为True，则将性能作为消息记录
    if display:
        # conf.logger.log(f"The validation performance = {perf}.")
        conf.logger.log(f"The validation/test performance = {perf} for {label}.")
    return perf


def ensembled_validate(
    conf,
    coordinator,
    models,
    criterion,
    metrics,
    data_loader,
    label="test_loader",
    ensemble_scheme=None,
):
    """A function for model evaluation."""
    if data_loader is None:
        return None

    # switch to evaluation mode.
    for model in models:
        model.eval()

        # place the model to the device.
        if conf.graph.on_cuda:
            model = model.cuda()

    # evaluate on test_loader.
    tracker_te = RuntimeTracker(metrics_to_track=metrics.metric_names)

    for _input, _target in data_loader:
        # load data and check performance.
        data_batch = create_dataset.load_data_batch(
            conf, _input, _target, is_training=False
        )

        with torch.no_grad():
            # ensemble.
            if (
                ensemble_scheme is None
                or ensemble_scheme == "avg_losses"
                or ensemble_scheme == "avg_logits"
            ):
                outputs = []
                for model in models:
                    outputs.append(model(data_batch["input"]))
                output = sum(outputs) / len(outputs)
            elif ensemble_scheme == "avg_probs":
                outputs = []
                for model in models:
                    outputs.append(F.softmax(model(data_batch["input"])))
                output = sum(outputs) / len(outputs)

            # eval the performance.
            loss = torch.FloatTensor([0])
            performance = metrics.evaluate(loss, output, data_batch["target"])

        # update the tracker.
        tracker_te.update_metrics(
            [loss.item()] + performance, n_samples=data_batch["input"].size(0)
        )

    # place back model to the cpu.
    for model in models:
        if conf.graph.on_cuda:
            model = model.cpu()

    # display the test stat.
    if label is not None:
        display_test_stat(conf, coordinator, tracker_te, label)
    perf = tracker_te()
    conf.logger.log(f"The performance of the ensenmbled model: {perf}.")
    return perf


def recover_models(conf, client_models, flatten_local_models, use_cuda=True):
    # init the local models.
    import copy
    num_models = len(flatten_local_models)
    local_models = {}

    for client_idx, flatten_local_model in flatten_local_models.items():
        arch = conf.clientid2arch[client_idx]
        _model = copy.deepcopy(client_models[arch])
        _model_state_dict = _model.state_dict()
        flatten_local_model.unpack(_model_state_dict.values())
        _model.load_state_dict(_model_state_dict)
        local_models[client_idx] = _model.cuda() if conf.graph.on_cuda else _model

        # turn off the grad for local models.
        # for param in local_models[client_idx].parameters():
        #     param.requires_grad = False
    return num_models, local_models


def get_n_local_epoch(conf, n_participated):
    if conf.min_local_epochs is None:
        return [conf.local_n_epochs] * n_participated
    else:
        # here we only consider to (uniformly) randomly sample the local epochs.
        assert conf.min_local_epochs > 1.0
        random_local_n_epochs = conf.random_state.uniform(
            low=conf.min_local_epochs, high=conf.local_n_epochs, size=n_participated
        )
        return random_local_n_epochs
