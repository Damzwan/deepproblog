import torch
import random
import torchvision
from torch.utils.data import Dataset
from torchvision.transforms import transforms

TRAINING_DATA_SIZE = 100
TYPES_OF_CLOTHES = 1

class FashionMNIST_Group(Dataset):

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
# transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081, ))])

train_dataset = torchvision.datasets.FashionMNIST(root='./data/', train=True, download=True, transform=transform)
train_dataset = FashionMNIST_Group(train_dataset, TYPES_OF_CLOTHES)
test_loader = torch.utils.data.DataLoader(torchvision.datasets.FashionMNIST('./data/', train=False, transform=transform), batch_size=10, shuffle=True)

dataList = []
obsList = []
for i in range(TRAINING_DATA_SIZE):
    clothes = train_dataset.getRandom()
    dataList.append({'c1': clothes[0][0].unsqueeze(0), 'c2': clothes[1][0].unsqueeze(0), 'c3': clothes[2][0].unsqueeze(0)})
    obsList.append('')
