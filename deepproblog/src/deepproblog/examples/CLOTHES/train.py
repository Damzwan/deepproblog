import sys
from json import dumps

import torch

from deepproblog.dataset import DataLoader
from deepproblog.engines import ApproximateEngine, ExactEngine
from deepproblog.evaluate import get_confusion_matrix
from deepproblog.examples.CLOTHES.network import Clothes_MNIST_Net
from deepproblog.examples.CLOTHES.data import clothesGroup

from deepproblog.heuristics import geometric_mean
from deepproblog.model import Model
from deepproblog.network import Network
from deepproblog.train import train_model
from deepproblog.utils import get_configuration, format_time_precise, config_to_string

i = int(sys.argv[1]) if len(sys.argv) > 1 else 0

parameters = {
    "method": ["exact"],
    "N": [1],
    "run": range(5),
    "exploration": [False],
}

configuration = get_configuration(parameters, i)
torch.manual_seed(configuration["run"])

name = "clothes" + config_to_string(configuration) + "_" + format_time_precise()
print(name)

train_set = clothesGroup(configuration["N"], "train", 20)
test_set = clothesGroup(configuration["N"], "test", 20)

print(train_set.to_queries())

network = Clothes_MNIST_Net()

# pretrain = configuration["pretrain"]
# if pretrain is not None and pretrain > 0:
#     network.load_state_dict(
#         torch.load("models/pretrained/all_{}.pth".format(configuration["pretrain"]))
#     )

net = Network(network, "cloth_mnist_net", batching=True)
net.optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)

model = Model("cloth.pl", [net])
if configuration["method"] == "exact":
    if configuration["exploration"] or configuration["N"] > 2:
        print("Not supported?")
        exit()
    model.set_engine(ExactEngine(model), cache=True)

model.add_tensor_source("train", train_set.get_tensor_source())
model.add_tensor_source("test", test_set.get_tensor_source())

loader = DataLoader(train_set, 2, False)
train = train_model(model, loader, 1, log_iter=100, profile=0)
model.save_state("snapshot/" + name + ".pth")
train.logger.comment(dumps(model.get_hyperparameters()))
train.logger.comment(
    "Accuracy {}".format(get_confusion_matrix(model, test_set, verbose=1).accuracy())
)
train.logger.write_to_file("log/" + name)
