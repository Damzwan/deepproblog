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
        self.trainingData = []
        self.tops = [0, 2, 6][:size]
        self.bots = [1, 3, 8][:size]
        self.shoe = [5, 7, 9][:size]
        self.allClothes = self.tops + self.bots + self.shoe
        random.seed(0)
        self.data = dict()
        for i in range(10):
            self.data[i] = []
        for datapoint in dataset:
            self.data[datapoint[1]].append(datapoint)
        print("Finished loading data")
        self.createTrainingData()
        print("Finished creating training data")

    def createTrainingData(self):
        corrects, wrongs = TRAINING_DATA_SIZE / 2, TRAINING_DATA_SIZE / 2
        while corrects > 0 or wrongs > 0: # very unsafe, assumes more wrongs than rights
            newCombo = random.choices(self.allClothes, k=3) if wrongs > 0 else self.getRandomCorrect()
            correct = self.isCorrectCombination(newCombo)
            self.trainingData.append(([self.convertClothesClassToTensor(clothId) for clothId in newCombo], correct))
            if correct:
                corrects -= 1
            else:
                wrongs -= 1
        self.trainingData = random.sample(self.trainingData, len(self.trainingData)) # shuffle data

    def isCorrectCombination(self, combo):
        return all([bool(set(c) & set(combo)) for c in [self.tops, self.bots, self.shoe]])

    def convertClothesClassToTensor(self, classo):
        return random.sample(self.data[classo], 1)[0]

    def __getitem__(self, index):
        return self.trainingData[index]

    def getRandomCorrect(self):
        return random.sample([random.choice(self.tops), random.choice(self.bots), random.choice(self.shoe)], 3)

    def __len__(self):
        return self.trainingData


transform = transforms.Compose([transforms.ToTensor()])
# transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081, ))])

train_dataset = torchvision.datasets.FashionMNIST(root='./data/', train=True, download=True, transform=transform)
train_dataset = FashionMNIST_Group(train_dataset, TYPES_OF_CLOTHES)
test_loader = torch.utils.data.DataLoader(torchvision.datasets.FashionMNIST('./data/', train=False, transform=transform), batch_size=10, shuffle=True)

dataList = []
obsList = []
for combo in train_dataset:
    dataList.append({'c1': combo[0][0][0].unsqueeze(0), 'c2': combo[0][1][0].unsqueeze(0), 'c3': combo[0][2][0].unsqueeze(0)})
    r = '1' if combo[1] else '0'
    obsList.append(':- not clothesGroup(c1, c2, c3, ' + r + ').')
