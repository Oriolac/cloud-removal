from typing import List, Any, Dict


class Model:
    def __init__(self, name: Any, kwargs: Dict[str, Any]):
        self.name = name
        self.kwargs = kwargs


class Checkpoint:
    def __init__(self, path: Any, model: Any, loss: Any = None):
        self.path = path
        self.model = model
        self.loss = loss


class Optimizer:
    def __init__(self, name: str, kwargs: Dict[str, Any]):
        self.name = name
        self.kwargs = kwargs


class Data:
    def __init__(self, dataset: Any, root_imgs: str, input: str, batch_size: int, num_workers: int):
        self.dataset = dataset
        self.root_imgs = root_imgs
        self.input = input
        self.batch_size = batch_size
        self.num_workers = num_workers
