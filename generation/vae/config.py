from typing import List, Any, Dict

from data.config import Data, Model, Checkpoint, Optimizer


class Train:
    def __init__(self, model: Model, start_epoch: int, epochs: int, checkpoint: Checkpoint, criterion: str,
                 optimizer: Optimizer):
        self.model: Model = model
        self.start_epoch = start_epoch
        self.epochs = epochs
        self.checkpoint: Checkpoint = checkpoint
        self.criterion = criterion
        self.optimizer: Optimizer = optimizer


class Config:
    def __init__(self, seed: int, gpu: Any, data: Data, train: Train):
        self.seed = seed
        self.gpu = gpu
        self.data: Data = data
        self.train: Train = train


def populate_classes(yaml_dict: Dict[str, Any]) -> Config:
    optimizer = Optimizer(**yaml_dict['train']['optimizer'])
    checkpoint = Checkpoint(**yaml_dict['train']['checkpoint'])
    model = Model(**yaml_dict['train']['model'])
    train = Train(model=model, checkpoint=checkpoint, optimizer=optimizer,
                  **{k: v for k, v in yaml_dict['train'].items() if k not in ['model', 'checkpoint', 'optimizer']})
    data = Data(**yaml_dict['data'])
    return Config(data=data, train=train, **{k: v for k, v in yaml_dict.items() if k not in ['data', 'train']})
