import os
import random
from datetime import timedelta

# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import pcode.utils.checkpoint as checkpoint
import pcode.utils.logging as logging
import pcode.utils.param_parser as param_parser
import pcode.utils.topology as topology
from parameters import get_args
from pcode.masters.master_fedavg import MasterFedAvg
# from pcode.masters.master_fedfomo import MasterFedFomo
# from pcode.masters.master_fedper import MasterFedPer
# from pcode.masters.master_lg_fedavg import MasterLGFedAvg
# from pcode.masters.master_local_train import MasterLocalTrain
from pcode.masters.master_pFedSD import MasterpFedSD
# from pcode.masters.master_pFedMe import MasterpFedMe
# from pcode.masters.master_tlkt import MasterTLKT
from pcode.workers.worker_fedavg import WorkerFedAvg
# from pcode.workers.worker_fedfomo import WorkerFedFomo
# from pcode.workers.worker_fedper import WorkerFedPer
# from pcode.workers.worker_lg_fedavg import WorkerLGFedAvg
# from pcode.workers.worker_local_train import WorkerLocalTrain
from pcode.workers.worker_pFedSD import WorkerpFedSD

# from pcode.workers.worker_pFedme import WorkerpFedMe
# from pcode.workers.worker_tlkt import WorkerTLKT


def main(rank, size, conf, port):
    # init the distributed world.
    # 初始化分布式环境
    try:
        # 设置主节点地址为本地主机（127.0.0.1）
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        # 设置主节点端口
        os.environ['MASTER_PORT'] = port
        # 使用"gloo"后端初始化分布式进程组，指定当前进程的排名（rank）、进程组的大小（size）、超时时间为300分钟
        dist.init_process_group("gloo", rank=rank, world_size=size, timeout=timedelta(minutes=300))
    # 若初始化分布式环境出现AttributeError异常，则将conf.distributed0设置为False
    except AttributeError as e:
        print(f"failed to init the distributed world: {e}.")
        conf.distributed = False

    # init the config.
    # 初始化配置
    init_config(conf)

    # 根据配置中的算法选择不同的实现类
    if conf.algo == "fedavg":
        master_func = MasterFedAvg
        worker_func = WorkerFedAvg
    # elif conf.algo == "fedprox":
    #     master_func = MasterFedAvg
    #     worker_func = WorkerFedAvg
    # elif conf.algo == "fedper":
    #     master_func = MasterFedPer
    #     worker_func = WorkerFedPer
    # elif conf.algo == "lg_fedavg":
    #     master_func = MasterLGFedAvg
    #     worker_func = WorkerLGFedAvg
    # elif conf.algo == "pFedme":
    #     master_func = MasterpFedMe
    #     worker_func = WorkerpFedMe    
    # elif conf.algo == "fedfomo":
    #     master_func = MasterFedFomo
    #     worker_func = WorkerFedFomo
    # elif conf.algo == "local_training":
    #     master_func = MasterLocalTrain
    #     worker_func = WorkerLocalTrain
    # elif conf.algo == "tlkt":
    #     master_func = MasterTLKT
    #     worker_func = WorkerTLKT
    elif conf.algo == "pFedSD":
        # MasterpFedSD用于控制和管理分布式训练的主进程
        master_func = MasterpFedSD
        # WorkerpFedSD用于执行具体训练任务的工作进程
        worker_func = WorkerpFedSD
        

    else:
        raise NotImplementedError

    # start federated learning.
    # 首先根据配置中的conf.graph.rank来判断当前进程是否为0，如果是0，则将master_func(conf)赋值给process，
    # 否则将worker_func(conf)赋值给process
    process = master_func(conf) if conf.graph.rank == 0 else worker_func(conf)
    # 调用process.run()方法来运行相应的进程
    process.run()


# 定义函数init_config来初始化配置函数
def init_config(conf):
    # define the graph for the computation.
    # 使用topology.define_graph_topology函数来定义图形结构，并设置进程的排名
    conf.graph = topology.define_graph_topology(
        world=conf.world,
        world_conf=conf.world_conf,
        n_participated=conf.n_participated,
        on_cuda=conf.on_cuda,
    )
    conf.graph.rank = dist.get_rank()

    # init related to randomness on cpu.
    # 初始化与CPU上的随机性相关的设置，根据配置中的same_seed_process决定是否使用相同的随机种子，并设置随机种子
    if not conf.same_seed_process:
        conf.manual_seed = 1000 * conf.manual_seed + conf.graph.rank
    np.random.seed(conf.manual_seed)
    conf.random_state = np.random.RandomState(conf.manual_seed)
    torch.manual_seed(conf.manual_seed)
    random.seed(conf.manual_seed)

    # configure cuda related.
    # 配置与CUDA相关的设置，包括设置CUDA随机种子、设置当前进程使用的CUDA设备等
    if conf.graph.on_cuda:
        assert torch.cuda.is_available()
        torch.cuda.manual_seed(conf.manual_seed)
        # torch.cuda.set_device(conf.graph.primary_device)
        # device_id = conf.graph.rank % torch.cuda.device_count()
        torch.cuda.set_device(conf.graph.rank % torch.cuda.device_count())
        # print(torch.cuda.current_device())
        torch.backends.cudnn.enabled = True
        # torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.benchmark = False
        # torch.backends.cudnn.deterministic = True if conf.train_fast else False
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed_all(conf.manual_seed)

    # init the model arch info.
    # 初始化模型架构信息，将配置中的complex_arch解析为字典形式的模型架构信息
    conf.arch_info = (
        param_parser.dict_parser(conf.complex_arch)
        if conf.complex_arch is not None
        else {"master": conf.arch, "worker": conf.arch}
    )
    conf.arch_info["worker"] = conf.arch_info["worker"].split(":")

    # parse the fl_aggregate scheme.
    # 解析fl_aggregate方案，将配置中的fl_aggregate解析为字典形式，并将其各个键值对作为属性设置到配置对象中
    conf._fl_aggregate = conf.fl_aggregate
    conf.fl_aggregate = (
        param_parser.dict_parser(conf.fl_aggregate)
        if conf.fl_aggregate is not None
        else conf.fl_aggregate
    )
    [setattr(conf, f"fl_aggregate_{k}", v) for k, v in conf.fl_aggregate.items()]

    # define checkpoint for logging (for federated learning server).
    # 为日志记录定义检查点，根据配置中的checkpoint_dir和进程的排名初始化检查点
    checkpoint.init_checkpoint(conf, rank=str(conf.graph.rank))

    # configure logger.
    # 配置日志记录器，将日志记录器与指定的检查点目录关联
    conf.logger = logging.Logger(conf.checkpoint_dir)

    # display the arguments' info.
    # 显示参数信息，如果当前进程的排名为0，则使用logging.display_args函数显示配置参数的信息
    if conf.graph.rank == 0:
        logging.display_args(conf)

    # sync the processes.
    # 同步进程，使用dist.barrier()实现进程间的同步操作
    dist.barrier()


import time

# 可以实现分布式训练的并行启动和运行。主进程会创建多个子进程，每个子进程都会调用main函数，并传入相应的排名和配置参数。
# 主进程等待所有子进程完成后才会退出。每个子进程在main函数中根据自己的排名和配置参数执行相应的逻辑。
if __name__ == "__main__":
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"
    # 通过调用get_args()函数获取配置参数，该函数可能用于解析命令行参数或从配置文件中读取参数
    conf = get_args()
    # 计算参与训练的客户端数量
    conf.n_participated = int(conf.n_clients * conf.participation_ratio + 0.5)
    # conf.timestamp = str(int(time.time()))
    # 确定分布式环境中的进程数量，加1是为了包含主进程
    size = conf.n_participated + 1
    # 创建一个空列表，用于存储启动的进程对象
    processes = []
    # 设置进程启动的方法为"spawn"，该方法可以保证在Windows和macOS上的兼容性
    mp.set_start_method("spawn")
    # 使用for循环遍历进程的排名（0到size-1）
    for rank in range(size):
        # p = mp.Process(target=main, args=(rank, size, conf, conf.port)) 创建一个新的进程对象，
        # 目标函数为main，传入的参数为当前进程的排名、进程数量、配置参数和端口号
        p = mp.Process(target=main, args=(rank, size, conf, conf.port))
        # 启动进程
        p.start()
        # 将进程对象添加到进程列表中
        processes.append(p)
    # 使用另外一个for循环等待所有进程的结束运行
    for p in processes:
        p.join()
