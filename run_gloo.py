import os
import random
from datetime import timedelta

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
from pcode.masters.master_HKD import MasterHKD
from pcode.workers.worker_fedavg import WorkerFedAvg
from pcode.workers.worker_HKD import WorkerHKD


def main(rank, size, conf, port):
    try:
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = port
        dist.init_process_group("gloo", rank=rank, world_size=size, timeout=timedelta(minutes=300))
    except AttributeError as e:
        print(f"failed to init the distributed world: {e}.")
        conf.distributed = False

    init_config(conf)

    if conf.algo == "fedavg":
        master_func = MasterFedAvg
        worker_func = WorkerFedAvg
    elif conf.algo == "HKD_pFed":
        master_func = MasterHKD
        worker_func = WorkerHKD
    else:
        raise NotImplementedError

    process = master_func(conf) if conf.graph.rank == 0 else worker_func(conf)
    process.run()


def init_config(conf):
    conf.graph = topology.define_graph_topology(
        world=conf.world,
        world_conf=conf.world_conf,
        n_participated=conf.n_participated,
        on_cuda=conf.on_cuda,
    )
    conf.graph.rank = dist.get_rank()

    if not conf.same_seed_process:
        conf.manual_seed = 1000 * conf.manual_seed + conf.graph.rank
    np.random.seed(conf.manual_seed)
    conf.random_state = np.random.RandomState(conf.manual_seed)
    torch.manual_seed(conf.manual_seed)
    random.seed(conf.manual_seed)

    if conf.graph.on_cuda:
        assert torch.cuda.is_available()
        torch.cuda.manual_seed(conf.manual_seed)
        torch.cuda.set_device(conf.graph.rank % torch.cuda.device_count())
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed_all(conf.manual_seed)

    conf.arch_info = (
        param_parser.dict_parser(conf.complex_arch)
        if conf.complex_arch is not None
        else {"master": conf.arch, "worker": conf.arch}
    )
    conf.arch_info["worker"] = conf.arch_info["worker"].split(":")

    conf._fl_aggregate = conf.fl_aggregate
    conf.fl_aggregate = (
        param_parser.dict_parser(conf.fl_aggregate)
        if conf.fl_aggregate is not None
        else conf.fl_aggregate
    )
    [setattr(conf, f"fl_aggregate_{k}", v) for k, v in conf.fl_aggregate.items()]

    checkpoint.init_checkpoint(conf, rank=str(conf.graph.rank))

    conf.logger = logging.Logger(conf.checkpoint_dir)

    if conf.graph.rank == 0:
        logging.display_args(conf)

    dist.barrier()


if __name__ == "__main__":
    conf = get_args()
    conf.n_participated = int(conf.n_clients * conf.participation_ratio + 0.5)
    size = conf.n_participated + 1
    processes = []
    mp.set_start_method("spawn")
    for rank in range(size):
        p = mp.Process(target=main, args=(rank, size, conf, conf.port))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
