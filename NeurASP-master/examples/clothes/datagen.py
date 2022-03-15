import torch
import torchvision
from torch.utils.data import Dataset
from torchvision.transforms import transforms

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081, ))])

train_dataset = torchvision.datasets.FashionMNIST(root='./data/', train=True, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(torchvision.datasets.FashionMNIST('./data/', train=False, transform=transform), batch_size=1000, shuffle=True)