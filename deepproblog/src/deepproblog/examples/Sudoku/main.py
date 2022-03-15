from deepproblog.examples.Sudoku.network import Sudoku_Net
from deepproblog.dataset import DataLoader
from deepproblog.network import Network
from deepproblog.model import Model
from deepproblog.engines import ApproximateEngine, ExactEngine
from deepproblog.train import train_model
from deepproblog.evaluate import get_confusion_matrix

from json import dumps
import torch


network = Sudoku_Net()

net = Network(network, "sudoku_net", batching=True)
net.optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)

model = Model("models/addition.pl", [net])

model.set_engine(ExactEngine(model), cache=True)
model.add_tensor_source("train", Sudoku_train)
model.add_tensor_source("test", Sudoku_test)

loader = DataLoader(train_set, 2, False)
train = train_model(model, loader, 1, log_iter=100, profile=0)
model.save_state("snapshot/sudoku.pth")
train.logger.comment(dumps(model.get_hyperparameters()))
train.logger.comment(
    "Accuracy {}".format(get_confusion_matrix(model, test_set, verbose=1).accuracy())
)
