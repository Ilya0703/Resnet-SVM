import glob
import pickle
import cv2
from sklearn.preprocessing import normalize
import numpy as np
from torchvision.models import resnet50, ResNet50_Weights
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
weights = ResNet50_Weights.DEFAULT
model = resnet50(weights=weights)
model.eval()
model.to(device)
preprocess = weights.transforms().to(device)
loaded_model = pickle.load(open("resnet_svm.pkl", 'rb'))
pca = pickle.load(open('resnet_pca.pkl', 'rb'))
batch_size = 64


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def slide_extract(image, windowSize, pca=pca, clf=loaded_model):
    hIm, wIm = image.shape[:2]
    step = 16
    windows = []
    coords = []
    for w1, w2 in zip(range(wIm - windowSize[0], 0, -1 * step), range(wIm, windowSize[0], -1 * step)):
        for h1, h2 in zip(range(hIm - windowSize[1], 0, -1 * step), range(hIm, windowSize[1], -1 * step)):
            fd = image[h1:h2, w1:w2]
            fd = cv2.resize(fd, (64, 64))
            fd = np.array(Image.fromarray(fd))
            fd = torch.tensor(fd)
            fd = fd.permute(2, 1, 0)
            coords.append([w1, w2, h1, h2])
            windows.append(fd)
    tensors = torch.stack(windows, dim=0)
    tensors.to(device)
    tensors = preprocess(tensors).unsqueeze(0).to(device)
    tensors = tensors.squeeze(0)
    data_loader = DataLoader(tensors, batch_size=batch_size)
    img_emb = []
    for data in data_loader:
        a = (
            (model(data).detach()
             ))
        b = torch.split(a, 1, dim=0)
        for i in b:
            img_emb.append(np.array(i.squeeze(0).to('cpu')).tolist())
    img_emb = normalize(img_emb)
    img_emb = pca.transform(img_emb)
    coords_car = []
    for i in range(len(img_emb)):
        if clf.predict([img_emb[i]]) == 'Car':
            coords_car.append(coords[i])
            image = cv2.rectangle(image, (coords[i][0], coords[i][2]), (coords[i][1], coords[i][3]), (0, 255, 0), 2)
    torch.cuda.empty_cache()
    return coords_car, image


test_images_path = ('data/test/*.jpg')
img_names = list(glob.glob(test_images_path))
i = 0
for image in tqdm(img_names):
    img = np.array(pil_loader(image))
    coords, img = slide_extract(img, (64, 64))
    cv2.imwrite(f'results/aftersvm_№{str(i)}.jpg', img)
    i += 1
