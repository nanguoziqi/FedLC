import os
import sys
import json
import time
import random
import pickle
import inspect
import importlib
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models as models
from torch import Tensor

from pathlib import Path
from argparse import ArgumentParser, Namespace
from collections import OrderedDict
from copy import deepcopy
from typing import Dict, List, OrderedDict

from functools import partial
from collections import OrderedDict
from typing import List, Optional
from sklearn import metrics
from html.parser import HTMLParser

from typing import List, Tuple, Union
from torch.utils.data import Dataset
from rich.console import Console
from rich.progress import track
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from collections import Counter



class BaseDataset(Dataset):
    def __init__(self) -> None:
        self.classes: List = None
        self.data: torch.Tensor = None
        self.targets: torch.Tensor = None
        self.train_data_transform = None
        self.train_target_transform = None
        self.general_data_transform = None
        self.general_target_transform = None
        self.enable_train_transform = True

    def __getitem__(self, index):
        #取出对应索引index的数据和标签
        data, targets = self.data[index], self.targets[index]
        if self.enable_train_transform and self.train_data_transform is not None:
            data = self.train_data_transform(data)
        if self.enable_train_transform and self.train_target_transform is not None:
            targets = self.train_target_transform(targets)
        if self.general_data_transform is not None:
            data = self.general_data_transform(data)
        if self.general_target_transform is not None:
            targets = self.general_target_transform(targets)
        return data, targets

    def __len__(self):
        return len(self.targets)


class CIFAR10(BaseDataset):
    def __init__(
        self,
        root,
        args=None,
        general_data_transform=None,
        general_target_transform=None,
        train_data_transform=None,
        train_target_transform=None,
    ):
        super().__init__()
        
        #root 是一个字符串，表示数据集下载和存储的根目录路径。
        #第一个参数 True 或 False 用于指定数据集的子集。True 表示加载训练集（train set），False 表示加载测试集（test set）。
        #download=True 表示如果数据集不在指定的 root 目录下，将会自动从互联网上下载数据集。
        train_part = torchvision.datasets.CIFAR10(root, True, download=True)
        test_part = torchvision.datasets.CIFAR10(root, False, download=True)

        #torch.Tensor()数据转换为 PyTorch 张量。permute([0, -1, 1, 2]) 是一个维度重排操作
        #(batch_size, channels, height, width) 变为(batch_size, height, width, channels)
        train_data = torch.Tensor(train_part.data).permute([0, -1, 1, 2]).float()
        test_data = torch.Tensor(test_part.data).permute([0, -1, 1, 2]).float()
        
        #在 CIFAR-10 的情况下，标签通常是一个二维张量，其中第一个维度是批处理大小（batch size），第二个维度是单个整数标签。
        #squeeze() 方法会移除这个大小为 1 的批处理维度，使得标签张量变为一维，每个元素代表一个图像的类别标签。
        train_targets = torch.Tensor(train_part.targets).long().squeeze()
        test_targets = torch.Tensor(test_part.targets).long().squeeze()
        
        self.data = torch.cat([train_data, test_data])
        self.targets = torch.cat([train_targets, test_targets])
        self.classes = train_part.classes
        self.general_data_transform = general_data_transform
        self.general_target_transform = general_target_transform
        self.train_data_transform = train_data_transform
        self.train_target_transform = train_target_transform

class CIFAR100(BaseDataset):
    def __init__(
        self,
        root,
        args,
        general_data_transform=None,
        general_target_transform=None,
        train_data_transform=None,
        train_target_transform=None,
    ):
        super().__init__()
        train_part = torchvision.datasets.CIFAR100(root, True, download=True)
        test_part = torchvision.datasets.CIFAR100(root, False, download=True)
        train_data = torch.Tensor(train_part.data).permute([0, -1, 1, 2]).float()
        test_data = torch.Tensor(test_part.data).permute([0, -1, 1, 2]).float()
        train_targets = torch.Tensor(train_part.targets).long().squeeze()
        test_targets = torch.Tensor(test_part.targets).long().squeeze()
        self.data = torch.cat([train_data, test_data])
        self.targets = torch.cat([train_targets, test_targets])
        self.classes = train_part.classes
        self.general_data_transform = general_data_transform
        self.general_target_transform = general_target_transform
        self.train_data_transform = train_data_transform
        self.train_target_transform = train_target_transform
        super_class = None
        if isinstance(args, Namespace):
            super_class = args.super_class
        elif isinstance(args, dict):
            super_class = args["super_class"]

        if super_class:
            # super_class: [sub_classes]
            CIFAR100_SUPER_CLASS = {
                0: ["beaver", "dolphin", "otter", "seal", "whale"],
                1: ["aquarium_fish", "flatfish", "ray", "shark", "trout"],
                2: ["orchid", "poppy", "rose", "sunflower", "tulip"],
                3: ["bottle", "bowl", "can", "cup", "plate"],
                4: ["apple", "mushroom", "orange", "pear", "sweet_pepper"],
                5: ["clock", "keyboard", "lamp", "telephone", "television"],
                6: ["bed", "chair", "couch", "table", "wardrobe"],
                7: ["bee", "beetle", "butterfly", "caterpillar", "cockroach"],
                8: ["bear", "leopard", "lion", "tiger", "wolf"],
                9: ["cloud", "forest", "mountain", "plain", "sea"],
                10: ["bridge", "castle", "house", "road", "skyscraper"],
                11: ["camel", "cattle", "chimpanzee", "elephant", "kangaroo"],
                12: ["fox", "porcupine", "possum", "raccoon", "skunk"],
                13: ["crab", "lobster", "snail", "spider", "worm"],
                14: ["baby", "boy", "girl", "man", "woman"],
                15: ["crocodile", "dinosaur", "lizard", "snake", "turtle"],
                16: ["hamster", "mouse", "rabbit", "shrew", "squirrel"],
                17: ["maple_tree", "oak_tree", "palm_tree", "pine_tree", "willow_tree"],
                18: ["bicycle", "bus", "motorcycle", "pickup_truck", "train"],
                19: ["lawn_mower", "rocket", "streetcar", "tank", "tractor"],
            }
            mapping = {}
            for super_cls, sub_cls in CIFAR100_SUPER_CLASS.items():
                for cls in sub_cls:
                    mapping[cls] = super_cls
            new_targets = []
            for cls in self.targets:
                new_targets.append(mapping[self.classes[cls]])
            self.targets = torch.tensor(new_targets, dtype=torch.long)

class SVHN(BaseDataset):
    def __init__(
        self,
        root,
        args=None,
        general_data_transform=None,
        general_target_transform=None,
        train_data_transform=None,
        train_target_transform=None,
    ):
        super().__init__()
        if not isinstance(root, Path):
            root = Path(root)
        train_part = torchvision.datasets.SVHN(root / "raw", "train", download=True)
        test_part = torchvision.datasets.SVHN(root / "raw", "test", download=True)
        train_data = torch.Tensor(train_part.data).float()
        test_data = torch.Tensor(test_part.data).float()
        train_targets = torch.Tensor(train_part.labels).long()
        test_targets = torch.Tensor(test_part.labels).long()

        self.data = torch.cat([train_data, test_data])
        self.targets = torch.cat([train_targets, test_targets])
        self.classes = list(range(10))
        self.general_data_transform = general_data_transform
        self.general_target_transform = general_target_transform
        self.train_data_transform = train_data_transform
        self.train_target_transform = train_target_transform


DATASETS = {
    "cifar10": CIFAR10,
    "cifar100":CIFAR100,
    "svhn":SVHN,
}

DATA_MEAN = {
    "cifar10": [0.4914, 0.4822, 0.4465],
    "cifar100": [0.5071, 0.4865, 0.4409],
    "femnist": [0.9637],
    "svhn": [0.4377, 0.4438, 0.4728],
}


DATA_STD = {
    "cifar10": [0.2023, 0.1994, 0.201],
    "cifar100": [0.2009, 0.1984, 0.2023],
    "femnist": [0.155],
    "svhn": [0.1201, 0.1231, 0.1052],
}




class DecoupledModel(nn.Module):
    def __init__(self):
        #调用父类nn.Module的初始化函数
        super(DecoupledModel, self).__init__()
        self.base: nn.Module = None
        self.classifier: nn.Module = None

    #检查模型是否可用
    def check_avaliability(self):
        if self.base is None or self.classifier is None:
            raise RuntimeError(
                "You need to re-write the base and classifier in your custom model class."
            )

    #前向传播
    def forward(self, x: Tensor) -> Tensor:
        return self.classifier(self.base(x))


#数据集的输入通道
INPUT_CHANNELS = {
    "femnist": 1,
    "cifar10": 3,
    "svhn": 3,
    "cifar100": 3,
}

#数据集类别数
NUM_CLASSES = {
    "svhn": 10,
    "femnist": 62,
    "cifar10": 10,
    "cifar100": 100,
}

class LeNet5(DecoupledModel):
    def __init__(self, dataset: str) -> None:
        super(LeNet5, self).__init__()
        feature_length = {
            "femnist": 256,
            "cifar10": 400,
            "svhn": 400,
            "cifar100": 400,
        }
        #特征提取部分
        self.base = nn.Sequential(
            OrderedDict(
                conv1=nn.Conv2d(INPUT_CHANNELS[dataset], 6, 5),
                bn1=nn.BatchNorm2d(6),
                activation1=nn.ReLU(),
                pool1=nn.MaxPool2d(2),
                conv2=nn.Conv2d(6, 16, 5),
                bn2=nn.BatchNorm2d(16),
                activation2=nn.ReLU(),
                pool2=nn.MaxPool2d(2),
                flatten=nn.Flatten(),
                fc1=nn.Linear(feature_length[dataset], 120),
                activation3=nn.ReLU(),
                fc2=nn.Linear(120, 84),
            )
        )

        #最后一个全连接层
        self.classifier = nn.Linear(84, NUM_CLASSES[dataset])

    def forward(self, x):
        return self.classifier(F.relu(self.base(x)))

# CNN used in FedAvg
class FedAvgCNN(DecoupledModel):
    def __init__(self, dataset: str):
        super(FedAvgCNN, self).__init__()
        features_length = {
            "femnist": 1,
            "cifar10": 1600,
            "cifar100": 1600,
            "svhn": 1600,
        }
        self.base = nn.Sequential(
            OrderedDict(
                conv1=nn.Conv2d(INPUT_CHANNELS[dataset], 32, 5),
                activation1=nn.ReLU(),
                pool1=nn.MaxPool2d(2),
                conv2=nn.Conv2d(32, 64, 5),
                activation2=nn.ReLU(),
                pool2=nn.MaxPool2d(2),
                flatten=nn.Flatten(),
                fc1=nn.Linear(features_length[dataset], 512),
            )
        )
        self.classifier = nn.Linear(512, NUM_CLASSES[dataset])

    def forward(self, x):
        return self.classifier(F.relu(self.base(x)))

class AlexNet(DecoupledModel):
    def __init__(self, dataset):
        super().__init__()

        #注意：如果不希望对参数进行预训练，请将“预训练”设置为False
        pretrained = True
        alexnet = models.alexnet(
            #加载预训练权重
            weights=models.AlexNet_Weights.DEFAULT if pretrained else None
        )
        self.base = alexnet
        self.classifier = nn.Linear(
            #获取了AlexNet原始分类器最后一层的输入特征数
            alexnet.classifier[-1].in_features, NUM_CLASSES[dataset]
        )
        #nn.Identity()是返回其输入
        self.base.classifier[-1] = nn.Identity()

MODELS = {
    "lenet5": LeNet5,
    "avgcnn": FedAvgCNN,
    "alex": AlexNet,
}




#转换为NumPy数组
def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.cpu().numpy()
    elif isinstance(x, list):
        return np.array(x)
    else:
        raise TypeError(
            f"input data should be torch.Tensor or built-in list. Now {type(x)}"
        )


class Metrics:
    def __init__(self, loss=None, predicts=None, targets=None):
        self._loss = loss if loss is not None else 0.0
        self._targets = targets if targets is not None else []
        self._predicts = predicts if predicts is not None else []

    #更新损失，预测值
    def update(self, other):
        if other is not None:
            self._predicts.extend(to_numpy(other._predicts))
            self._targets.extend(to_numpy(other._targets))
            self._loss += other._loss

    def _calculate(self, metric, **kwargs):
        #利用传入的metric函数来计算self._targets（真实值）和self._predicts（预测值）之间的某种指标
        return metric(self._targets, self._predicts, **kwargs)

    @property
    def loss(self):
        try:
            loss = self._loss / len(self._targets)
        except ZeroDivisionError:
            return 0
        return loss

    @property
    def accuracy(self):
        if self.size == 0:
            return 0
        score = self._calculate(metrics.accuracy_score)
        return score * 100
    """
    @property
    def corrects(self):
        return self._calculate(metrics.accuracy_score, normalize=False)
    """
    @property
    def size(self):
        return len(self._targets)




def fix_random_seed(seed: int) -> None:
    """修复FL训练的随机种子。

    Args：
    seed（int）：您喜欢的任意数字作为随机种子。
    """
    #设置Python的哈希随机种子
    os.environ["PYTHONHASHSEED"] = str(seed)
    #random 模块用于生成伪随机数
    random.seed(seed)
    #设置NumPy库的随机数生成器的种子
    np.random.seed(seed)
    #设置PyTorch库的随机数生成器的种子
    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
        #清除PyTorch的CUDA缓存
        torch.cuda.empty_cache()
        #设置所有CUDA随机数生成器的种子，确保在GPU上生成的随机数也是可复现的
        torch.cuda.manual_seed_all(seed)
    #设置cudnn为确定性模式。当这个标志被设置为True时，CuDNN的卷积操作将使用确定的算法，这意味着给定相同的输入和权重，卷积操作将总是产生相同的结果
    torch.backends.cudnn.deterministic = True
    #当这个标志被设置为False时，CuDNN会禁用自动调整卷积算法以寻找最快算法的功能
    torch.backends.cudnn.benchmark = False

#动态选择CUDA设备（内存最多）运行FL实验。torch.device:所选的CUDA设备。
def get_optimal_cuda_device(use_cuda: bool) -> torch.device:
    if not torch.cuda.is_available() or not use_cuda:
        return torch.device("cpu")
    #调用pynvml库中的nvmlInit函数,提供了监控和管理NVIDIA GPU状态的接口
    pynvml.nvmlInit()
    gpu_memory = []
    if "CUDA_VISIBLE_DEVICES" in os.environ.keys():
        #如果CUDA_VISIBLE_DEVICES存在，这行代码会将其值按逗号分割，并将每个分割出来的字符串转换为整数，然后将这些整数存储在列表gpu_ids中
        gpu_ids = [int(i) for i in os.environ["CUDA_VISIBLE_DEVICES"].split(",")]
        #使用assert语句来确保gpu_ids列表中的最大设备ID小于PyTorch可以检测到的CUDA设备总数
        assert max(gpu_ids) < torch.cuda.device_count()
    else:
        #使用range函数创建一个包含所有可用CUDA设备ID的列表
        gpu_ids = range(torch.cuda.device_count())

    for i in gpu_ids:
        #获取对应设备ID的NVML设备句柄
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        #获取该GPU设备的内存信息
        memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        #memory_info中提取free字段，这个字段表示GPU设备上当前的空闲内存量
        gpu_memory.append(memory_info.free)
    #转换成一个NumPy数组
    gpu_memory = np.array(gpu_memory)
    #使用np.argmax()函数找到gpu_memory数组中最大值的索引
    best_gpu_id = np.argmax(gpu_memory)
    return torch.device(f"cuda:{best_gpu_id}")

#将`src`中`.requires_grad=True`的所有参数收集到列表中并返回。返回：参数列表[，参数名称]。
def trainable_params(
    src: Union[OrderedDict[str, torch.Tensor], torch.nn.Module],
    detach=False,
    requires_name=False,
) -> Union[List[torch.Tensor], Tuple[List[torch.Tensor], List[str]]]:

    #detach()分离计算图，clone()深度复制
    func = (lambda x: x.detach().clone()) if detach else (lambda x: x)
    parameters = []
    keys = []
    #isinstance() 函数来判断一个对象是否是一个已知的类型
    if isinstance(src, OrderedDict):
        for name, param in src.items():
            if param.requires_grad:
                #如果param需要梯度，那么它将param传递给之前定义的func函数（该函数可能会从计算图中分离并克隆张量），并将结果添加到parameters列表中
                parameters.append(func(param))
                keys.append(name)
    elif isinstance(src, torch.nn.Module):
        #使用state_dict(keep_vars=True)方法获取模型的参数。这个方法返回一个有序字典（OrderedDict），其中包含了模型的参数
        for name, param in src.state_dict(keep_vars=True).items():
            if param.requires_grad:
                parameters.append(func(param))
                keys.append(name)

    if requires_name:
        return parameters, keys
    else:
        return parameters

#根据enable_log参数的值，决定是否启用日志，并将日志输出到控制台或指定的文件。如果启用了日志，它会创建一个Console对象来负责日志的输出，并设置相应的输出选项
class Logger:
    def __init__(
        self, stdout: Console, enable_log: bool, logfile_path: Union[Path, str]
    ):
        """此类用于解决库`rich`中进度条和日志函数之间的不兼容问题。
        Args：
        stdout（控制台）：`rich.Console。控制台`用于将信息打印到标准输出。
        enable_log（bool）：标志表示日志函数是否处于活动状态。
        logfile_path（Union[path，str]）：日志文件的路径。
        """

        self.stdout = stdout
        self.logfile_stream = None
        self.enable_log = enable_log
        if self.enable_log:
            self.logfile_stream = open(logfile_path, "w")
            #Console控制台对象
            self.logger = Console(
                file=self.logfile_stream, record=True, log_path=False, log_time=False
            )

    def log(self, *args, **kwargs):
        self.stdout.log(*args, **kwargs)
        if self.enable_log:
            #所有传递给self.log方法的参数（*args和**kwargs）也会被传递给self.logger.log方法
            self.logger.log(*args, **kwargs)

    #关闭类实例中与日志文件相关的流
    def close(self):
        if self.logfile_stream:
            self.logfile_stream.close()

@torch.no_grad()
def evalutate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion=torch.nn.CrossEntropyLoss(reduction="sum"),
    device=torch.device("cpu"),
) -> Metrics:

    model.eval()
    metrics = Metrics()
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y).item()
        pred = torch.argmax(logits, -1)
        metrics.update(Metrics(loss, pred, y))
    return metrics



def get_fedavg_argparser() -> ArgumentParser:
    #创建解析器
    parser = ArgumentParser()
    parser.add_argument(
        "-m", "--model", type=str, default="lenet5", choices=MODELS.keys()
    )
    parser.add_argument(
        "-d", "--dataset", type=str, choices=DATASETS.keys(), default="cifar10"
    )
    parser.add_argument("--seed", type=int, default=42)#用于运行实验的随机种子
    parser.add_argument("-jr", "--join_ratio", type=float, default=0.1)#（每轮客户数）/（总客户数）的比率。
    parser.add_argument("-ge", "--global_epoch", type=int, default=100)
    parser.add_argument("-le", "--local_epoch", type=int, default=5)
    parser.add_argument("-tg", "--test_gap", type=int, default=1)#对客户端执行测试的间隔回合。
    parser.add_argument("--eval_test", type=int, default=1)#在本地训练前后对加入的客户的测试集执行评估的非零值。
    parser.add_argument("--eval_val", type=int, default=0)
    parser.add_argument("--eval_train", type=int, default=0)
    parser.add_argument(
        "-op", "--optimizer", type=str, default="sgd", choices=["sgd", "adam"]
    )#客户端优化器
    parser.add_argument("-lr", "--local_lr", type=float, default=1e-2)#学习率
    parser.add_argument("-mom", "--momentum", type=float, default=0.0)#优化器动量
    parser.add_argument("-wd", "--weight_decay", type=float, default=0.0)#权重衰减
    parser.add_argument("-bs", "--batch_size", type=int, default=32)#用于客户端本地培训的数据批量大小。
    parser.add_argument("--external_model_params_file", type=str, default="")#外部模型参数的相对文件路径#####
    parser.add_argument("--use_cuda", type=int, default=1)#是否使用cuda
    parser.add_argument("--save_log", type=int, default=0)#保存算法运行日志
    parser.add_argument("--save_model", type=int, default=0)#保存模型参数
    parser.add_argument("--save_fig", type=int, default=1)#保存精度曲线
    parser.add_argument("--save_metrics", type=int, default=1)#保存度量统计信息
    return parser




class FedAvgServer:
    def __init__(
        self,
        algo: str = "FedAvg",
        args: Namespace = None,
        default_trainer=True,
    ):
        self.args = get_fedavg_argparser().parse_args() if args is None else args
        self.algo = algo

        #固定随机种子
        fix_random_seed(self.args.seed)
        #获取当前的时间，并格式化为"年-月-日-时:分:秒"的形式
        begin_time = str(
            time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(round(time.time())))
        )
        #创建输出目录
        self.output_dir = OUT_DIR / self.algo / begin_time
        
        #读取数据集信息
        with open(PROJECT_DIR / "data" / self.args.dataset / "args.json", "r") as f:
            self.args.dataset_args = json.load(f)

        #获取客户端信息
        try:
            partition_path = PROJECT_DIR / "data" / self.args.dataset / "partition.pkl"
            with open(partition_path, "rb") as f:
                partition = pickle.load(f)
        except:
            raise FileNotFoundError(f"Please partition {args.dataset} first.")
        
        self.train_clients: List[int] = partition["separation"]["train"]
        self.test_clients: List[int] = partition["separation"]["test"]
        self.val_clients: List[int] = partition["separation"]["val"]
        self.client_num: int = partition["separation"]["total"]

        #是否使用gpu，若使用gpu则选择最佳的gpu
        self.device = get_optimal_cuda_device(self.args.use_cuda)

        #将根据模型的名称返回一个类，
        #然后通过指示数据集并调用类来初始化模型对象。
        #最后将模型对象传输到目标设备。
        self.model: DecoupledModel = MODELS[self.args.model](
            dataset=self.args.dataset
        ).to(self.device)
        #调用模型的 check_avaliability() 方法来检查模型的可用性
        self.model.check_avaliability()

        #global_params_dict用于输出单个全局模型的传统FL，有序字典（OrderedDict）
        self.global_params_dict: OrderedDict[str, torch.Tensor] = None

        #trainable_params 用于从模型中提取可训练的参数。这个函数可能返回两个值：一个是可训练参数的张量列表，另一个是对应参数名称的列表。
        #函数中的 detach=True 参数意味着提取的参数不会被计算图所连接， requires_name=True 表示需要返回参数的名称。
        random_init_params, self.trainable_params_name = trainable_params(
            self.model, detach=True, requires_name=True
        )
        #OrderedDict是一个有序的字典类，它保持了元素被插入时的顺序。当遍历这个字典时，将按照参数名称在self.trainable_params_name列表中出现的顺序来访问它们。
        self.global_params_dict = OrderedDict(
            #zip函数将两个列表self.trainable_params_name和random_init_params组合成一个迭代器
            zip(self.trainable_params_name, random_init_params)
        )
        if (
            #预训练文件路径
            self.args.external_model_params_file
            and os.path.isfile(self.args.external_model_params_file)
        ):
            # 加载预训练参数
            self.global_params_dict = torch.load(
                self.args.external_model_params_file, map_location=self.device
            )

        #创建本地训练epoch列表
        self.clients_local_epoch: List[int] = [self.args.local_epoch] * self.client_num

        #确保所有算法都通过相同的客户端采样流运行。
        #如果采样发生在每一轮FL的开始，一些算法在客户端的隐式操作可能会干扰流。
        #每个全局epoch，随机选取客户端
        self.client_sample_stream = [
            random.sample(
                self.train_clients, max(1, int(self.client_num * self.args.join_ratio))
            )
            for _ in range(self.args.global_epoch)
        ]
        self.selected_clients: List[int] = []
        self.current_epoch = 0
        
        #用于在测试时控制某些特定方法的行为（并非所有方法都使用）
        self.test_flag = False

        ##用于日志记录的变量##
        #如果目录不存在，且需要保存日志、图表或指标，则创建目录
        if not os.path.isdir(self.output_dir) and (
            self.args.save_log or self.args.save_fig or self.args.save_metrics
        ):
            os.makedirs(self.output_dir, exist_ok=True)

        stdout = Console(log_path=False, log_time=False)
        self.logger = Logger(
            stdout=stdout,
            enable_log=self.args.save_log,
            logfile_path=OUT_DIR
            / self.algo
            / self.output_dir
            / f"{self.args.dataset}_log.html",
        )
        self.test_results: Dict[int, Dict[str, Dict[str, Metrics]]] = {}
        #创建进度条
        self.train_progress_bar = track(
            #stdout 是 Python 的标准库 sys 中的一个对象，表示标准输出
            range(self.args.global_epoch), "[bold green]Training...", console=stdout
        )

        #用于在日志中记录实验的开始，包括实验使用的算法和所有相关的参数
        self.logger.log("=" * 20, self.algo, "=" * 20)
        self.logger.log("Experiment Arguments:", dict(self.args._get_kwargs()))
        
        # init trainer
        self.trainer = None
        if default_trainer:
            self.trainer = FedAvgClient(
                deepcopy(self.model), self.args, self.logger, self.device
            )

    #全面的FL流程。
    def run(self):
        #提醒指定trainer
        begin = time.time()
        if self.trainer is None:
            raise RuntimeError(
                "Specify your unique trainer or set `default_trainer` as True."
            )
        
        self.train()
        end = time.time()
        total = end - begin
        self.logger.log(
            f"{self.algo}'s total running time: {int(total // 3600)} h {int((total % 3600) // 60)} m {int(total % 60)} s."
        )
        self.log_max_metrics()
        self.logger.close()       
        
        #保存模型
        if self.args.save_model:
            model_name = (
                f"{self.args.dataset}_{self.args.global_epoch}_{self.args.model}.pt"
            )
            torch.save(self.global_params_dict, self.output_dir / model_name)
        
    def train(self):
        avg_round_time = 0
        #全局训练的epoch
        for E in self.train_progress_bar:
            self.current_epoch = E

            #每test_gap轮测试一次##########################
            if (E + 1) % self.args.test_gap == 0:
                self.test()

            #被选取的客户端编号
            self.selected_clients = self.client_sample_stream[E]
            begin = time.time()
            self.train_one_round()
            end = time.time()
            #self.log_info()
            avg_round_time = (avg_round_time * (self.current_epoch) + (end - begin)) / (
                self.current_epoch + 1
            )

        self.logger.log(
            f"{self.algo}'s average time taken by each global epoch: {int(avg_round_time // 60)} m {(avg_round_time % 60):.2f} s."
        )

    #测试FL方法输出的功能。
    def test(self):
        self.test_flag = True
        client_ids = set(self.val_clients + self.test_clients)
        all_same = False
        if client_ids:
            if self.val_clients == self.train_clients == self.test_clients:
                all_same = True
                results = {
                    "all_clients": {
                            "train": Metrics(),
                            "val": Metrics(),
                            "test": Metrics(),
                    }
                }
            else:
                results = {
                    "val_clients": {
                            "train": Metrics(),
                            "val": Metrics(),
                            "test": Metrics(),
                    },
                    "test_clients": {
                            "train": Metrics(),
                            "val": Metrics(),
                            "test": Metrics(),
                    },
                }
            #对被选中的客户端进行测试
            for cid in client_ids:
                client_local_params = self.generate_client_params(cid)
                client_metrics = self.trainer.test(cid, client_local_params)

                #将测试结果加入results
                for split in ["train", "val", "test"]:
                    if all_same:
                        results["all_clients"][split].update(
                            client_metrics[split]
                        )
                    else:
                        if cid in self.val_clients:
                            results["val_clients"][split].update(
                                client_metrics[split]
                            )
                        if cid in self.test_clients:
                            results["test_clients"][split].update(
                                client_metrics[split]
                            )

            self.test_results[self.current_epoch + 1] = results

        self.test_flag = False

    def generate_client_params(self, client_id: int) -> OrderedDict[str, torch.Tensor]:

            return self.global_params_dict

    #在每一轮通信中（在服务器端）需要做的具体事情的功能。
    def train_one_round(self):
        delta_cache = []
        weight_cache = []
        for client_id in self.selected_clients:
            client_local_params = self.generate_client_params(client_id)
            (
                delta,
                weight,
            ) = self.trainer.train(
                client_id=client_id,
                local_epoch=self.clients_local_epoch[client_id],
                new_parameters=client_local_params,
            )

            delta_cache.append(delta)
            weight_cache.append(weight)

        self.aggregate(delta_cache, weight_cache)

    @torch.no_grad()
    def aggregate(
        self,
        delta_cache: List[OrderedDict[str, torch.Tensor]],
        weight_cache: List[int],
        return_diff=True,
    ):
        """
        此函数用于聚合从选定客户端接收的模型参数。聚合方法默认为加权平均。
        delta_cache（List[List[torc.Tensor]]）：“delta”表示本地训练前后客户端模型参数之间的差异。
        weight_cache（List[int]）：每个“delta”的权重（默认情况下为客户端数据集大小）。
        """
        #归一化
        weights = torch.tensor(weight_cache, device=self.device) / sum(weight_cache)
        if return_diff:
            delta_list = [list(delta.values()) for delta in delta_cache]
            #根据weights计算每个位置的加权和
            aggregated_delta = [
                #torch.stack`将这些元素堆叠成一个形状为`(N,)`的张量，其中`dim=-1`指定了堆叠的维度。
                #这意味着，如果`diff`包含了`N`个长度为`K`的列表，结果将是一个形状为`(N, K)`的张量。
                torch.sum(weights * torch.stack(diff, dim=-1), dim=-1)
                #zip(*delta_list),将delta_list转置
                for diff in zip(*delta_list)
            ]

            #对模型的全局参数进行更新
            for param, diff in zip(self.global_params_dict.values(), aggregated_delta):
                param.data -= diff
        else:
            #使用delta_cache中的增量值和一个权重列表weights来更新self.global_params_dict中的旧参数。
            #每次更新都是通过将增量值与相应的权重相乘，并沿最后一个维度求和来实现的。通常在模型的训练过程中使用，特别是当使用自定义的优化算法或者特定的参数更新策略时。
            for old_param, zipped_new_param in zip(
                self.global_params_dict.values(), zip(*delta_cache)
            ):
                old_param.data = (torch.stack(zipped_new_param, dim=-1) * weights).sum(
                    dim=-1
                )
        #如果strict为True（默认值），那么当状态字典中缺少键或有多余的键时，将会抛出一个错误。如果strict为False，那么只会加载匹配的键，并忽略不匹配的键。
        self.model.load_state_dict(self.global_params_dict, strict=False)
    
    #找出最大的acc
    def log_max_metrics(self):
        self.logger.log("=" * 20, self.algo, "Max Accuracy", "=" * 20)

        colors = {
            "before": "blue",
            "after": "red",
            "train": "yellow",
            "val": "green",
            "test": "cyan",
        }

        groups = ["val_clients", "test_clients"]
        if self.train_clients == self.val_clients == self.test_clients:
            groups = ["all_clients"]

        for group in groups:
            self.logger.log(f"{group}:")
            for split, flag in [
                ("train", self.args.eval_train),
                ("val", self.args.eval_val),
                ("test", self.args.eval_test),
            ]:
                if flag:
                    #将self.test_results字典转换成一个新的列表metrics_list，列表中的每个元素都是一个元组，元组的第一个元素是原字典的键，
                    #第二个元素是从原字典的值（一个嵌套字典）中提取的特定值，这个特定值是通过group、stage和split这三个键来索引得到的
                    metrics_list = list(
                        map(
                            lambda tup: (tup[0], tup[1][group][split]),
                            self.test_results.items(),
                        )
                    )
                    if len(metrics_list) > 0:
                        #出metrics_list中准确率（accuracy）最高的元组，并返回这个元组的epoch和accuracy值
                        epoch, max_acc = max(
                            [
                                (epoch, metrics.accuracy)
                                for epoch, metrics in metrics_list
                            ],
                            #当max()函数比较列表中的元素时，它会使用这个lambda函数来确定哪个元素是最大的
                            key=lambda tup: tup[1],
                        )
                        self.logger.log(
                            f"[{colors[split]}]({split})[/{colors[split]}] max_acc: {max_acc:.2f}% at epoch {epoch}"
                        )

                    #生成准确率的图像
                    if self.args.save_fig:
                        import matplotlib
                        from matplotlib import pyplot as plt

                        matplotlib.use("Agg")
                        plt.plot(
                                    [
                                    metrics.accuracy
                                    for epoch, metrics in metrics_list
                                    ],
                                    label=f"{split}",
                                )
                        plt.title(f"{self.algo}_{self.args.dataset}")
                        plt.ylim(0, 100)
                        plt.xlabel("Communication Rounds")
                        plt.ylabel("Accuracy")
                        plt.legend()
                        plt.savefig(
                            OUT_DIR / self.algo / self.output_dir / f"{self.args.dataset}.pdf",
                            bbox_inches="tight",
                        )




class FedAvgClient:
    def __init__(
        self,
        model: DecoupledModel,
        args: Namespace,
        logger: Logger,
        device: torch.device,
    ):
        self.args = args
        self.device = device
        self.client_id: int = None

        #加载数据集和客户端的数据索引
        try:
            partition_path = PROJECT_DIR / "data" / self.args.dataset / "partition.pkl"
            with open(partition_path, "rb") as f:
                partition = pickle.load(f)
        except:
            raise FileNotFoundError(f"Please partition {args.dataset} first.")

        #训练集索引和测试集索引
        self.data_indices: List[List[int]] = partition["data_indices"]

        #定义数据转换
        #通过均值和方差对将数据转化为标准正态分布
        general_data_transform = transforms.Compose(
            [
                transforms.Normalize(
                    DATA_MEAN[self.args.dataset], DATA_STD[self.args.dataset]
                )
            ]
            if self.args.dataset in DATA_MEAN and self.args.dataset in DATA_STD
            else []
        )
        general_target_transform = transforms.Compose([])
        train_data_transform = transforms.Compose([])
        train_target_transform = transforms.Compose([])

        self.dataset = DATASETS[self.args.dataset](
            root=PROJECT_DIR / "data" / args.dataset,
            args=args.dataset_args,
            general_data_transform=general_data_transform,
            general_target_transform=general_target_transform,
            train_data_transform=train_data_transform,
            train_target_transform=train_target_transform,
        )

        self.trainloader: DataLoader = None
        self.valloader: DataLoader = None
        self.testloader: DataLoader = None
        #Subset()可对数据集取子集
        self.trainset: Subset = Subset(self.dataset, indices=[])
        self.valset: Subset = Subset(self.dataset, indices=[])
        self.testset: Subset = Subset(self.dataset, indices=[])
        self.test_flag = False

        #将模型（model）移动到指定的设备（self.device）上
        self.model = model.to(self.device)
        self.local_epoch = self.args.local_epoch
        #创建了一个交叉熵损失函数（CrossEntropyLoss）的实例，并将其移动到相同的设备（self.device）上
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
        self.logger = logger

        #opt_state_dict用于存储优化器的状态
        self.opt_state_dict = {}
        if self.args.optimizer == "sgd":
            self.optimizer = torch.optim.SGD(
                #获得可训练的参数
                params=trainable_params(self.model),
                #设置学习率
                lr=self.args.local_lr,
                #设置SGD优化器的动量（momentum）。动量是一个帮助加速SGD在相关方向上移动，并抑制振荡的因子
                momentum=self.args.momentum,
                #设置权重衰减（weight decay）。权重衰减是一种正则化技术，通过在损失函数中添加权重的L2范数来惩罚大的权重，有助于防止模型过拟合
                weight_decay=self.args.weight_decay,
            )
        elif self.args.optimizer == "adam":
            self.optimizer = torch.optim.Adam(
                params=trainable_params(self.model),
                lr=self.args.local_lr,
                weight_decay=self.args.weight_decay,
            )
        #保存优化器在训练开始时的状态
        self.init_opt_state_dict = deepcopy(self.optimizer.state_dict())

    #测试功能。仅在FL测试回合中激活。
    def test(
        self, client_id: int, new_parameters: OrderedDict[str, torch.Tensor]
    ) -> Dict[str, Dict[str, Metrics]]:
        
        self.test_flag = True
        self.client_id = client_id
        self.load_dataset()
        self.set_parameters(new_parameters)

        results = {"train": Metrics(), "val": Metrics(), "test": Metrics()}
        results = self.evaluate()
        self.test_flag = False
        return results

    #此函数用于加载编号为“self.client_id”的客户端的数据索引。
    def load_dataset(self):
        
        self.trainset.indices = self.data_indices[self.client_id]["train"]
        self.testset.indices = self.data_indices[self.client_id]["test"]
        self.valset.indices = self.data_indices[self.client_id]["val"]
        #DataLoader 可以从给定的数据集中批量加载数据
        self.trainloader = DataLoader(self.trainset, self.args.batch_size, shuffle=True)
        self.valloader = DataLoader(self.valset, self.args.batch_size)
        self.testloader = DataLoader(self.testset, self.args.batch_size)

    #从服务器接收的负载模型参数。
    def set_parameters(self, new_parameters: OrderedDict[str, torch.Tensor]):

        self.optimizer.load_state_dict(
            self.opt_state_dict.get(self.client_id, self.init_opt_state_dict)
        )
        self.model.load_state_dict(new_parameters, strict=False)

    #评估功能。将在本地训练前后激活。
    @torch.no_grad()
    def evaluate(self, model: torch.nn.Module = None) -> Dict[str, Metrics]:
        
        # 评估时禁用训练数据转换
        self.dataset.enable_train_transform = False

        target_model = self.model if model is None else model
        target_model.eval()
        train_metrics = Metrics()
        val_metrics = Metrics()
        test_metrics = Metrics()
        criterion = torch.nn.CrossEntropyLoss(reduction="sum")

        if len(self.testset) > 0 and self.args.eval_test:
            test_metrics = evalutate_model(
                model=target_model,
                dataloader=self.testloader,
                criterion=criterion,
                device=self.device,
            )

        if len(self.valset) > 0 and self.args.eval_val:
            val_metrics = evalutate_model(
                model=target_model,
                dataloader=self.valloader,
                criterion=criterion,
                device=self.device,
            )

        if len(self.trainset) > 0 and self.args.eval_train:
            train_metrics = evalutate_model(
                model=target_model,
                dataloader=self.trainloader,
                criterion=criterion,
                device=self.device,
            )

        self.dataset.enable_train_transform = True
        return {"train": train_metrics, "val": val_metrics, "test": test_metrics}


    def train(
        self,
        client_id: int,
        local_epoch: int,
        new_parameters: OrderedDict[str, torch.Tensor],
        return_diff=True,
    ) -> Tuple[Union[OrderedDict[str, torch.Tensor], List[torch.Tensor]], int, Dict]:

        self.client_id = client_id
        self.local_epoch = local_epoch
        self.load_dataset()
        self.set_parameters(new_parameters)
        #eval_results = self.train_and_log()
        if self.local_epoch > 0:
            self.fit()
            #在本地培训结束时保存优化器的状态
            self.opt_state_dict[self.client_id] = deepcopy(self.optimizer.state_dict())
        
        
        if return_diff:
            delta = OrderedDict()
            for (name, p0), p1 in zip(
                new_parameters.items(), trainable_params(self.model)
            ):
                delta[name] = p0 - p1

            return delta, len(self.trainset)
        else:
            return (
                trainable_params(self.model, detach=True),
                len(self.trainset),
            )

    #模型训练
    def fit(self):

        #将模型设置为训练模式
        self.model.train()
        for _ in range(self.local_epoch):
            for x, y in self.trainloader:
                #当当前批大小为1时，模型中的batchNorm2d模块将引发错误。
                #因此，潜在大小为1的数据批被丢弃。
                if len(x) <= 1:
                    continue

                x, y = x.to(self.device), y.to(self.device)
                logit = self.model(x)
                loss = self.criterion(logit, y)
                #在反向传播（backpropagation）之前，需要清除之前累积的梯度。这行代码的作用是将优化器self.optimizer中的所有梯度清零
                self.optimizer.zero_grad()
                #执行反向传播，计算损失相对于模型参数的梯度
                loss.backward()
                #使用存储的梯度来更新模型的参数
                self.optimizer.step()




def get_fedlc_argparser() -> ArgumentParser:
    parser = get_fedavg_argparser()
    parser.add_argument("--tau", type=float, default=1.0)
    return parser

class FedLCServer(FedAvgServer):
    def __init__(
        self,
        algo: str = "FedLC",
        args: Namespace = None,
        default_trainer=False,
    ):
        if args is None:
            args = get_fedlc_argparser().parse_args()
        super().__init__(algo, args, default_trainer)
        self.trainer = FedLCClient(deepcopy(self.model), self.args, self.logger, self.device)




class FedLCClient(FedAvgClient):
    def __init__(self, model, args, logger, device):
        super().__init__(model, args, logger, device)
        self.label_distrib = torch.zeros(len(self.dataset.classes), device=self.device)

        def logit_calibrated_loss(logit, y):
            cal_logit = torch.exp(
                logit
                - (
                    self.args.tau
                    * torch.pow(self.label_distrib, -1 / 4)
                    .unsqueeze(0)
                    .expand((logit.shape[0], -1))
                )
            )

            y_logit = torch.gather(cal_logit, dim=-1, index=y.unsqueeze(1))
            loss = -torch.log(y_logit / cal_logit.sum(dim=-1, keepdim=True))
            
            return loss.sum() / logit.shape[0]

        self.criterion = logit_calibrated_loss

    def load_dataset(self):
        #继承的父类中的load_dataset方法
        super().load_dataset()
        #统计每个标签出现的次数。从训练集的索引获取标签，转为列表
        label_counter = Counter(self.dataset.targets[self.trainset.indices].tolist())
        #设置为零
        self.label_distrib.zero_()
        #遍历label_counter中的每个键值对，cls是标签，count是该标签出现的次数
        for cls, count in label_counter.items():
            self.label_distrib[cls] = max(1e-8, count)




PROJECT_DIR = Path(__file__).parent.absolute()
OUT_DIR = PROJECT_DIR / "out"

if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise ValueError(
            "Need to assign a method. Run like `python main.py <method> [args ...]`, e.g., python main.py fedavg -d cifar10 -m lenet5`"
        )

    #模型名称
    method = sys.argv[1]
    #模型参数
    args_list = sys.argv[2:]

    if method == 'fedavg':
        parser = get_fedavg_argparser()
        server = FedAvgServer(args=parser.parse_args(args_list))
    elif method == 'fedlc':
        parser = get_fedlc_argparser()
        server = FedLCServer(args=parser.parse_args(args_list))
    else:
        raise ValueError("There is no such model")
    server.run()
