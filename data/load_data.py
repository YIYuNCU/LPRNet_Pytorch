from torch.utils.data import *
from imutils import paths
import numpy as np
import random
import cv2
import os

CHARS = ['京', '沪', '津', '渝', '冀', '晋', '蒙', '辽', '吉', '黑',
         '苏', '浙', '皖', '闽', '赣', '鲁', '豫', '鄂', '湘', '粤',
         '桂', '琼', '川', '贵', '云', '藏', '陕', '甘', '青', '宁',
         '新',
         '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
         'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
         'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
         'W', 'X', 'Y', 'Z', 'I', 'O', '-'
         ]

CHARS_DICT = {char:i for i, char in enumerate(CHARS)}


def read_image(filename):
    image = cv2.imdecode(np.fromfile(filename, dtype=np.uint8), cv2.IMREAD_COLOR)
    return image

class LPRDataLoader(Dataset):
    def __init__(
        self,
        img_dir,
        imgSize,
        lpr_max_len,
        PreprocFun=None,
        augment=False,
        aug_prob=0.7,
        color_jitter=0.2,
        noise_std=6.0,
    ):
        self.img_dir = img_dir
        self.img_paths = []
        for i in range(len(img_dir)):
            self.img_paths += [el for el in paths.list_images(img_dir[i])]
        random.shuffle(self.img_paths)
        self.img_size = imgSize
        self.lpr_max_len = lpr_max_len
        self.augment = augment
        self.aug_prob = aug_prob
        self.color_jitter = color_jitter
        self.noise_std = noise_std
        if PreprocFun is not None:
            self.PreprocFun = PreprocFun
        else:
            self.PreprocFun = self.transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        filename = self.img_paths[index]
        Image = read_image(filename)
        if Image is None:
            raise FileNotFoundError("Failed to read image file: {}".format(filename))
        height, width, _ = Image.shape
        if height != self.img_size[1] or width != self.img_size[0]:
            Image = cv2.resize(Image, self.img_size)
        if self.augment:
            Image = self.augment_image(Image)
        Image = self.PreprocFun(Image)

        basename = os.path.basename(filename)
        imgname, suffix = os.path.splitext(basename)
        imgname = imgname.split("-")[0].split("_")[0]
        label = list()
        for c in imgname:
            # one_hot_base = np.zeros(len(CHARS))
            # one_hot_base[CHARS_DICT[c]] = 1
            label.append(CHARS_DICT[c])

        if len(label) == 8:
            if self.check(label) == False:
                print(imgname)
                assert 0, "Error label ^~^!!!"

        return Image, label, len(label)

    def augment_image(self, img):
        if random.random() > self.aug_prob:
            return img

        h, w = img.shape[:2]

        # Random affine transform to simulate camera jitter.
        if random.random() < 0.5:
            dx = random.uniform(-0.04, 0.04) * w
            dy = random.uniform(-0.08, 0.08) * h
            angle = random.uniform(-3.0, 3.0)
            scale = random.uniform(0.96, 1.04)
            m = cv2.getRotationMatrix2D((w * 0.5, h * 0.5), angle, scale)
            m[0, 2] += dx
            m[1, 2] += dy
            img = cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

        # Brightness and contrast jitter.
        if random.random() < 0.8:
            alpha = 1.0 + random.uniform(-self.color_jitter, self.color_jitter)
            beta = random.uniform(-20.0, 20.0)
            img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

        # Mild blur for motion/defocus robustness.
        if random.random() < 0.3:
            img = cv2.GaussianBlur(img, (3, 3), 0)

        # Gaussian noise for sensor robustness.
        if random.random() < 0.3:
            noise = np.random.normal(0.0, self.noise_std, img.shape).astype(np.float32)
            img = np.clip(img.astype(np.float32) + noise, 0.0, 255.0).astype(np.uint8)

        return img

    def transform(self, img):
        img = img.astype('float32')
        img -= 127.5
        img *= 0.0078125
        img = np.transpose(img, (2, 0, 1))

        return img

    def check(self, label):
        if label[2] != CHARS_DICT['D'] and label[2] != CHARS_DICT['F'] \
                and label[-1] != CHARS_DICT['D'] and label[-1] != CHARS_DICT['F']:
            print("Error label, Please check!")
            return False
        else:
            return True
