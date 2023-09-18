from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List

from data.config import Data, Model, Checkpoint, Optimizer


class ExperienceReplay:

    def __init__(self, buffer_size):
        self.buffer_size = buffer_size


STRATEGIES = {
    "experience replay": ExperienceReplay
}


class Train:
    def __init__(self, generator: Model, discriminator: Model, start_epoch: int, epochs: int, checkpoint: Checkpoint, criterion_generator: str,
                 optimizer_generator: Optimizer, criterion_discriminator: str, optimizer_discriminator: Optimizer,
                 strategies: Dict[str, Dict], metrics: Dict[str, Dict[str, Any]]):
        self.generator: Model = generator
        self.discriminator: Model = discriminator
        self.start_epoch = start_epoch
        self.epochs = epochs
        self.checkpoint: Checkpoint = checkpoint
        self.criterion_generator = criterion_generator
        self.optimizer_generator: Optimizer = optimizer_generator
        self.criterion_discriminator = criterion_discriminator
        self.optimizer_discriminator: Optimizer = optimizer_discriminator
        self.strategies = dict(map(lambda x: (x[0], STRATEGIES[x[0]](**x[1])), strategies.items()))
        self.metrics = metrics


class Config:
    def __init__(self, seed: int, gpu: Any, data: Data, train: Train):
        self.seed = seed
        self.gpu = gpu
        self.data: Data = data
        self.train: Train = train


def populate_classes(yaml_dict: Dict[str, Any]) -> Config:
    optimizer_generator = Optimizer(**yaml_dict['train']['optimizer_generator'])
    optimizer_discriminator = Optimizer(**yaml_dict['train']['optimizer_discriminator'])

    checkpoint = Checkpoint(**yaml_dict['train']['checkpoint'])
    generator = Model(**yaml_dict['train']['generator'])
    discriminator = Model(**yaml_dict['train']['discriminator'])
    train = Train(generator=generator, discriminator=discriminator, checkpoint=checkpoint, optimizer_generator=optimizer_generator,
                  optimizer_discriminator=optimizer_discriminator,
                  **{k: v for k, v in yaml_dict['train'].items() if
                     k not in ['generator', 'discriminator', 'checkpoint', 'optimizer_generator', 'optimizer_discriminator']})
    data = Data(**yaml_dict['data'])
    return Config(data=data, train=train, **{k: v for k, v in yaml_dict.items() if k not in ['data', 'train']})
