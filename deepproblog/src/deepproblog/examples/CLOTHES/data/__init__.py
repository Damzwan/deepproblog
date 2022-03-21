import itertools
import json
import random
from abc import ABC
from pathlib import Path
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset as TorchDataset
from typing import Callable, List, Iterable, Tuple

from deepproblog.dataset import Dataset
from deepproblog.query import Query
from problog.logic import Term, list2term, Constant


class ClothGroupGenerator(object):

    def __init__(self, dataset, size):
        self.labelMap = {0: 'T-shirt/top', 1: 'Trouser', 2: 'Pullover', 3: 'Dress', 4: 'Coat', 5: 'Sandal',
                         6: 'Shirt', 7: 'Sneaker', 8: 'Bag', 9: 'Ankle boot'}
        self.tops = [0, 2, 6][:size]
        self.bots = [1, 3, 8][:size]
        self.shoe = [5, 7, 9][:size]
        random.seed(0)
        self.data = dict()
        for i in range(10):
            self.data[i] = []
        for datapoint in dataset:
            self.data[datapoint[1]].append(datapoint)
        print("Finished loading data")

    def __getitem__(self, item):
        return self.getRandom()

    def getRandom(self):
        return random.sample([random.choice(self.data[random.choice(self.tops)]),
                              random.choice(self.data[random.choice(self.bots)]),
                              random.choice(self.data[random.choice(self.shoe)])], 3)

    def getCloth(self, kind, index):
        return self.data[kind][index]

    def __len__(self):
        return sum(len(x) for x in self.data.values())


transform = transforms.Compose([transforms.ToTensor()])

datasets = {
    "train": torchvision.datasets.FashionMNIST(root='./data/', train=True, download=True, transform=transform),
    "test": torchvision.datasets.FashionMNIST(root='./data/', train=False, download=True, transform=transform)
}


class MNIST_Images(object):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, item):
        return self.dataset[int(item[0])][0]


# TODO look at addition_mil
# class MNIST(Dataset):
#     def __len__(self):
#         return len(self.dataset)
#
#     def to_query(self, i):
#         l = self.dataset[i][1]
#         l = Constant(self.val_list.index(l))
#
#         return Query(
#             Term("clothes", Term("tensor", Term(self.dataset, Term("a"))), l),
#             substitution={Term("a"): Constant(i)},
#         )
#
#     def __init__(self, dataset):
#         self.dataset = dataset
#         self.val_list = list(labelMap.values())


def clothesGroup(config, dataset: str, size):
    """Returns a dataset for binary addition"""
    return MNISTOperator(
        dataset_name=dataset,
        function_name="clothesGroup",
        size=size,
        arity=3,
    )


class MNISTOperator(Dataset, TorchDataset):
    def __getitem__(self, index: int) -> Tuple[list, list, list]:
        i1, i2, i3 = self.data_indices[index]
        c1, c2, c3 = self.data[i1], self.data[i2], self.data[i3]
        return c1[0].unsqueeze(0), c2[0].unsqueeze(0), c3[0].unsqueeze(0)

    def __init__(self, dataset_name, size, function_name: str, arity, training_data_size: int = 100):
        """Generic dataset for operator(img, img) style datasets.
             :param function_name: Name of Problog function to query.
             :param size: Size of number of the cloth groups
             :param arity: Number of arguments for the operator
             """
        self.dataset_name = dataset_name
        self.function_name = function_name
        self.arity = arity

        dataset_generator = ClothGroupGenerator(datasets[dataset_name], size)
        self.data = []
        self.data_indices = []

        for i in range(training_data_size):
            j = 3 * i
            self.data.extend(dataset_generator.getRandom())
            self.data_indices.append((j, j + 1, j + 2))

    def get_tensor_source(self):
        return MNIST_Images(self.data)

    def to_file_repr(self, i):
        """Old file represenation dump. Not a very clear format as multi-digit arguments are not separated"""
        return NotImplementedError()

    def to_json(self):
        """
        Convert to JSON, for easy comparisons with other systems.

        Format is [EXAMPLE, ...]
        EXAMPLE :- [ARGS, expected_result]
        ARGS :- [MULTI_DIGIT_NUMBER, ...]
        MULTI_DIGIT_NUMBER :- [mnist_img_id, ...]
        """
        return NotImplementedError()

    def to_query(self, ind: int) -> Query:
        """Generate queries"""

        # Build substitution dictionary for the arguments
        indices = self.data_indices[ind]
        subs = dict()
        var_names = []

        for i in range(self.arity):
            inner_vars = []
            t = Term(f"p{i}")
            subs[t] = Term(
                "tensor",
                Term(
                    self.dataset_name,
                    Constant(indices[i]),
                ),
            )
            inner_vars.append(t)
            var_names.append(inner_vars)

        # Build query
        return Query(
            Term(
                self.function_name,
                *(e[0] for e in var_names),
            ),
            subs,
            output_ind=(0, 1, 2)
        )

    # TODO no label
    def _get_label(self, i: int):
        return NotImplementedError()

    def __len__(self):
        return len(self.data_indices)
