from PIL import Image
import torchvision.models as models
import torch
from torchvision.transforms import Compose, ToTensor
import torch.nn as nn
import cv2
import torch.nn.functional as F

# 讀取模型
def model_for_captcha(model_path, num_classes):
    model = models.densenet201(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path))
    return model
    
# 辨識圖片
def captcha_answer(img, model, height, width, text_length):
    transforms = Compose([ToTensor()])
    img = transforms(img)
    model.eval()
    img = img.view(1, 3, height, width)
    output = model(img)
    output = output.view(-1, 10)
    output = nn.functional.softmax(output, dim=1)
    output = torch.argmax(output, dim=1)
    output = output.view(-1, text_length)[0]
    pred = ''.join([str(num.item()) for num in output])
    return pred

# 讀取圖片
def img_loader(img_path):
    img = Image.open(img_path)
    return img.convert("RGB")
    
# AttentionBlock, ModifiedDenseNet201 注意力機制，額外處理文字位置不固定問題
class AttentionBlock(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(AttentionBlock, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1)
        self.fc2 = nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1)

    def forward(self, x):
        out = self.pool(x)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        out = torch.sigmoid(out)
        out = x * out
        return out
        
class ModifiedDenseNet201(nn.Module):
    def __init__(self, num_classes, attention=False):
        super(ModifiedDenseNet201, self).__init__()
        self.densenet = models.densenet201()
        self.attention = attention

        if self.attention:
            self.att_block1 = AttentionBlock(256)
            self.att_block2 = AttentionBlock(512)
            self.att_block3 = AttentionBlock(1792)

        self.fc = nn.Linear(1920, num_classes)  # 根据您的任务调整输入大小

    def forward(self, x):
        features = self.densenet.features(x)

        if self.attention:
            features = self.att_block1(features)
            features = self.att_block2(features)
            features = self.att_block3(features)

        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out

def test_hct1():
    img = img_loader('./test/hct_1.png')
    model = model_for_captcha('./hct_densenet_1.pth', 40)
    pred = captcha_answer(img, model, 30, 80, 4)
    print(pred)

def test_hct2():
    img = img_loader('./test/hct_2.png')
    model = model_for_captcha('./hct_densenet_2.pth', 40)
    pred = captcha_answer(img, model, 30, 90, 4)
    print(pred)
    
def test_amigodog():
    img = img_loader('./test/amigodog.png')
    model = ModifiedDenseNet201(num_classes=60, attention=False)
    state_dict = torch.load('./amigodog_densenet.pth')
    model.load_state_dict(state_dict)
    
    pred = captcha_answer(img, model, 30, 180, 6)
    print(pred)

test_hct1()
test_hct2()
test_amigodog()