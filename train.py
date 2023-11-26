import glob
from torch.utils.data import DataLoader
import numpy
import sklearn.preprocessing
import torch
from tqdm import tqdm
from PIL import Image
from sklearn.decomposition import PCA
from sklearn import svm
import numpy as np
import pickle
from torchvision.models import resnet50, ResNet50_Weights

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
weights = ResNet50_Weights.DEFAULT
model = resnet50(weights=weights)
model.eval()
model.to(device)
preprocess = weights.transforms().to(device)
pca = PCA(n_components=0.95)
batch_size = 192
degree = 3
c = 1
clf = svm.SVC(C=c, kernel='poly', degree=degree, coef0=1.0)

def pil_loader(path):

    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def build_tensor(img_path):
    img_names = list(glob.glob(img_path))
    img_preprocessed = []
    imgs = []
    for image in tqdm(img_names):
        img = numpy.array(pil_loader(image))
        img = torch.tensor(img)
        img = img.permute(2, 1, 0)
        imgs.append(img)
    tensors = torch.stack(imgs, dim=0)
    tensors.to(device)
    tensors = preprocess(tensors).unsqueeze(0).to(device)
    tensors = tensors.squeeze(0)
    data_loader = DataLoader(tensors, batch_size=batch_size)
    for data in data_loader:
        b = torch.split(model(data).detach(), 1, dim=0)
        for i in b:
            img_preprocessed.append(np.array(i.squeeze(0).to('cpu')))
        torch.cuda.empty_cache()
    return img_preprocessed


def apply_pca(images):
    images_normalized = sklearn.preprocessing.normalize(images)
    images_pcaed = pca.fit_transform(images_normalized)
    return images_pcaed, pca


def svm_fit(X, Y):
    clf.fit(X, Y)
    return clf


def save_models(clf, pca):
    svm_filename = "resnet_svm.pkl"
    pca_filename = "resnet_pca.pkl"
    pickle.dump(clf, open(svm_filename, 'wb'))
    pickle.dump(pca, open(pca_filename, 'wb'))


x = np.array([])
y = []
pos_false = "data/non-vehicles/*.png"
pos_true = "data/vehicles/*.png"
cars = build_tensor(pos_true)
not_cars = build_tensor(pos_false)
names_true = ['Car'] * len(cars)
names_false = ['Not car'] * len(not_cars)
X = np.concatenate((cars, not_cars))
Y = np.concatenate((names_true, names_false))
X_pcaed,pca = apply_pca(X)
clf = svm_fit(X_pcaed, Y)
save_models(clf, pca)
