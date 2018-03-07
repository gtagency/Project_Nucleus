import imgaug as ia
import numpy as np
from imgaug import augmenters as iaa
import cv2
from PIL import Image



image = np.load('input/x_train.npy')
seq = iaa.Sequential([iaa.Fliplr(0.5), iaa.GaussianBlur(sigma=(0, 3.0))])
images_aug = seq.augment_images(np.copy(image))

# w, h = 512, 512
# data = np.zeros((h, w, 3), dtype=np.uint8)
# data[256, 256] = [255, 0, 0]
img = Image.fromarray(image[100],'RGB')
img.show()

augmented_image = Image.fromarray(images_aug[100], 'RGB')
augmented_image.show()

#seq.show_grid(images_aug[0], cols=8, rows=8)

