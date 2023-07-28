import numpy as np
import pandas as pd
from PIL import Image
import io
import zipfile
import torch
from torch.utils.data import Dataset
import pickle
from sklearn.preprocessing import LabelEncoder
from torchvision import transforms
from pathlib import Path
import skimage.io
from skimage.transform import resize


class ZIPSimpsonsDataset(Dataset):
    """
    Датасет с картинками, который паралельно подгружает их из папок
    производит скалирование и превращение в торчевые тензоры
    """
    # разные режимы датасета 
    _DATA_MODES = ['train', 'val', 'test']
    # все изображения будут масштабированы к размеру 224x224 px
    _RESCALE_SIZE = 299
    def __init__(self, archive, files, mode, lab_file = None):
        super().__init__()
        # список файлов для загрузки
        self.archive = archive
        self.files = sorted(files)
        # режим работы
        self.mode = mode

        if self.mode not in self._DATA_MODES:
            print(f"{self.mode} is not correct; correct modes: {self._DATA_MODES}")
            raise NameError

        self.len_ = len(self.files)
     
        self.label_encoder = LabelEncoder()

        if self.mode != 'test':
    
            self.labels = (lab_file[[((path).name) for path in self.files]]).to_numpy()
            self.label_encoder.fit(self.labels)

            with open('label_encoder.pkl', 'wb') as le_dump_file:
                  pickle.dump(self.label_encoder, le_dump_file)

       
                      
    def __len__(self):
        return self.len_

    def load_sample(self, archive, file_name):
          f1 = io.BytesIO(archive.read(str(file_name)))
          image = skimage.io.imread(f1)
          f1.close()

          a = np.array(image)
            #img = Image.fromarray(a, 'RGB')
          return a
  
    def __getitem__(self, index):
        # для преобразования изображений в тензоры PyTorch и нормализации входа
        transform = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
        ])
        x = self.load_sample(self.archive, self.files[index])
        x = self._prepare_sample(x)
        #x = np.array(x / 255, dtype='float32')
        x = transform(x)
        if self.mode == 'test':
            return x
        else:
            label = self.labels[index]
            label_id = self.label_encoder.transform([label])
            y = label_id.item()
            return x, y
        
    def _prepare_sample(self, image):
        image = resize(image, (self._RESCALE_SIZE, self._RESCALE_SIZE), anti_aliasing=False)
        return np.array(image)

