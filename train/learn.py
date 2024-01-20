import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import resnet18, ResNet18_Weights

import os
import re
import torch
from torch.utils.data import Dataset
from PIL import Image

class CustomCircleDataset(Dataset):
    def __init__(self, base_directory, transform=None):
        self.base_directory = base_directory
        self.transform = transform
        self.images = []

        # ルートディレクトリ内の全サブディレクトリを走査
        for directory in os.listdir(base_directory):
            full_directory_path = os.path.join(base_directory, directory)
            
            if os.path.isdir(full_directory_path):
                self.images += [os.path.join(directory, img) for img in os.listdir(full_directory_path) if img.endswith(('.jpg', '.png'))]
        
        
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_rel_path = self.images[idx]
        img_full_path = os.path.join(self.base_directory, img_rel_path)
        image = Image.open(img_full_path).convert('RGB')

        # 画像名から座標を抽出
        coords = re.findall(r"(\d+\.\d+|\d+)", img_rel_path)
        x, y, r = float(coords[1]), float(coords[2]), float(coords[3])
        x, y, r = x / 224, y / 224, r / (224 * 1.414)
        label = torch.tensor([x, y, r], dtype=torch.float32)
        
        if self.transform:
            image = self.transform(image)

        return image, label


    
    


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, stride=2, padding=1)
        # 以下の数値は計算に基づいて適切に設定してください
        self.fc1 = nn.Linear(32 * 56 * 56, 128)  # 56は上記の計算から得られる値
        self.fc2 = nn.Linear(128, 3)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # フラット化
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
class SimpleCNN_1(nn.Module):
    def __init__(self):
        super(SimpleCNN_1, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)
        # 以下に更にレイヤーを追加することができます

        # 畳み込み層の後にフルコネクテッド層
        self.fc1 = nn.Linear(64 * 56 * 56, 500)
        self.fc2 = nn.Linear(500, 3)  # x座標、y座標、半径の出力

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 56 * 56)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class CustomResNet(nn.Module):
    def __init__(self, pretrained=False):
        super(CustomResNet, self).__init__()
        # ResNet18をベースとしたモデルを読み込む
        self.base_model = models.resnet18(pretrained=pretrained)
        # weights = ResNet18_Weights.DEFAULT
        # self.base_model = resnet18(weights=weights)
        # ResNetの最終層を置き換える
        num_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Linear(num_features, 3) # x, y座標と半径のための3つの出力

    def forward(self, x):
        return self.base_model(x)
    
class Classificate_Res(nn.Module):
    def __init__(self, pretrained=True):
        super(Classificate_Res, self).__init__()
        # ResNet-50のプリトレーニング済みモデルをロード
        self.resnet = models.resnet50(pretrained=pretrained)
        
        # 最終層を2クラス分類用にカスタマイズ
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, 2)

    def forward(self, x):
        return self.resnet(x)