import os
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt

from config import Config
import utils
import model as modellib
import visualize
from model import log

from skimage.transform import resize
from skimage.io import imread

from tqdm import tqdm
import pickle

DO_TRAINING = True

# Root directory of the project
ROOT_DIR = os.path.dirname(__file__)

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

class NucleusConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "nucleus"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    USE_MINI_MASK = False

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + 3 shapes


    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 128
    IMAGE_MAX_DIM = 128

    RPN_ANCHOR_SCALES = (4, 8, 16, 32)  # anchor side in pixels

    TRAIN_ROIS_PER_IMAGE = 512

    STEPS_PER_EPOCH = 600

    VALIDATION_STEPS = 70

config = NucleusConfig()
config.display()


def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.

    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax


class NucleusDataset(utils.Dataset):
    """Generates the shapes synthetic dataset. The dataset consists of simple
    shapes (triangles, squares, circles) placed randomly on a blank surface.
    The images are generated on the fly. No file access required.
    """

    X_train = None
    Y_train = None

    def load_images(self, image_id_list):
        """Generate the requested number of synthetic images.
        count: number of images to generate.
        height, width: the size of the generated images.
        """
        # Add classes
        self.add_class("nucleus", 1, "nucleus")

        # Add images
        # Generate random specifications of images (i.e. color and
        # list of shapes sizes and locations). This is more compact than
        # actual images. Images are generated on the fly in load_image().
        TRAIN_PATH = os.path.realpath(os.path.join(ROOT_DIR, '..', 'input', 'stage1_train'))
        # print(os.path.realpath(TRAIN_PATH))
        IMG_HEIGHT = 128
        IMG_WIDTH = 128
        IMG_CHANNELS = 3

        train_ids = next(os.walk(TRAIN_PATH))[1]

        #self.X_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
        self.X_train = []
        #Y_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
        self.Y_train = []

        # else:
        print('Getting and resizing images and masks ... ')
        sys.stdout.flush()
        j = 0
        for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
            if n in image_id_list:
                path = os.path.join(TRAIN_PATH, id_)
                img = imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]
                img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
                self.add_image("nucleus", j, path)
                j += 1
                self.X_train.append(img)

                mask_files = next(os.walk(path + '/masks/'))[2]
                masks = np.zeros((IMG_HEIGHT, IMG_WIDTH, len(mask_files)), dtype=np.bool)
                i = 0
                for mask_file in next(os.walk(path + '/masks/'))[2]:
                    mask_ = imread(path + '/masks/' + mask_file)
                    mask_ = resize(mask_, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
                    if np.sum(mask_) > 0:
                        masks[:,:,i] = mask_
                        i += 1
                self.Y_train.append(masks)
        print('Done!  Saving...')

    def load_image(self, image_id):
        """Generate an image from the specs of the given image ID.
        Typically this function loads the image from a file, but's
        in this case it generates the image on the fly from the
        specs in image_info.
        """
        return self.X_train[image_id]

    def load_mask(self, image_id):
        """Generate instance masks for shapes of the given image ID.
        """
        info = self.image_info[image_id]

        class_ids = []

        masks = self.Y_train[image_id]
        class_ids = np.ones(masks.shape[2])
        # Map class names to class IDs.
        return masks, class_ids.astype(np.int32)

dataset_train = None
if DO_TRAINING:
    # Training dataset
    dataset_train_path = os.path.join(ROOT_DIR, '..', 'input', 'dataset_train.pkl')
    if (os.path.exists(dataset_train_path)):
        with open(dataset_train_path, 'rb') as f:
            dataset_train = pickle.load(f)
    else:
        dataset_train = NucleusDataset()
        dataset_train.load_images(list(range(600)))
        dataset_train.prepare()
        with open(dataset_train_path, 'wb') as f:
            pickle.dump(dataset_train, f)

# Validation dataset
dataset_val_path = os.path.join(ROOT_DIR, '..', 'input', 'dataset_val.pkl')
if (os.path.exists(dataset_val_path)):
    with open(dataset_val_path, 'rb') as f:
        dataset_val = pickle.load(f)
else:
    dataset_val = NucleusDataset()
    dataset_val.load_images(list(range(600, 670)))
    dataset_val.prepare()
    with open(dataset_val_path, 'wb') as f:
        pickle.dump(dataset_val, f)


if DO_TRAINING:
    # Load and display random samples
    image_ids = np.random.choice(dataset_train.image_ids, 4)
    for image_id in image_ids:
        image = dataset_train.load_image(image_id)
        mask, class_ids = dataset_train.load_mask(image_id)
        #visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names, limit=1)


    # Create model in training mode
    model = modellib.MaskRCNN(mode="training", config=config,
                              model_dir=MODEL_DIR)


    # Which weights to start with?
    init_with = "coco"  # imagenet, coco, or last

    if init_with == "imagenet":
        model.load_weights(model.get_imagenet_weights(), by_name=True)
    elif init_with == "coco":
        # Load weights trained on MS COCO, but skip layers that
        # are different due to the different number of classes
        # See README for instructions to download the COCO weights
        model.load_weights(COCO_MODEL_PATH, by_name=True,
                           exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                    "mrcnn_bbox", "mrcnn_mask"])
    elif init_with == "last":
        # Load the last model you trained and continue training
        model.load_weights(model.find_last()[1], by_name=True)


    # Train the head branches
    # Passing layers="heads" freezes all layers except the head
    # layers. You can also pass a regular expression to select
    # which layers to train by name pattern.
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=1,
                layers='heads')


    # Fine tune all layers
    # Passing layers="all" trains all layers. You can also
    # pass a regular expression to select which layers to
    # train by name pattern.
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE / 10,
                epochs=4,
                layers="all")


    # Save weights
    # Typically not needed because callbacks save after every epoch
    # Uncomment to save manually
    # model_path = os.path.join(MODEL_DIR, "mask_rcnn_shapes.h5")
    # model.keras_model.save_weights(model_path)

class InferenceConfig(NucleusConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

inference_config = InferenceConfig()

# Recreate the model in inference mode
model = modellib.MaskRCNN(mode="inference",
                          config=inference_config,
                          model_dir=MODEL_DIR)

# Get path to saved weights
# Either set a specific path or find last trained weights
# model_path = os.path.join(ROOT_DIR, ".h5 file name here")
model_path = model.find_last()[1]

# Load trained weights (fill in path to trained weights here)
assert model_path != "", "Provide path to trained weights"
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)

# Test on a random image
image_id = random.choice(dataset_val.image_ids)
original_image, image_meta, gt_class_id, gt_bbox, gt_mask =\
    modellib.load_image_gt(dataset_val, inference_config,
                           image_id, use_mini_mask=False)

log("original_image", original_image)
log("image_meta", image_meta)
log("gt_class_id", gt_class_id)
log("gt_bbox", gt_bbox)
log("gt_mask", gt_mask)

visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id,
                            dataset_val.class_names, figsize=(8, 8))

visualize.display_images([original_image])

results = model.detect([original_image], verbose=1)

r = results[0]
visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'],
                            dataset_val.class_names, r['scores'], ax=get_ax())



'''
# Compute VOC-Style mAP @ IoU=0.5
# Running on 10 images. Increase for better accuracy.
image_ids = np.random.choice(dataset_val.image_ids, 10)
APs = []
for image_id in image_ids:
    # Load image and ground truth data
    image, image_meta, gt_class_id, gt_bbox, gt_mask =
        modellib.load_image_gt(dataset_val, inference_config,
                               image_id, use_mini_mask=False)
    molded_images = np.expand_dims(modellib.mold_image(image, inference_config), 0)
    # Run object detection
    results = model.detect([image], verbose=0)
    r = results[0]
    # Compute AP
    AP, precisions, recalls, overlaps =\
        utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                         r["rois"], r["class_ids"], r["scores"], r['masks'])
    APs.append(AP)

print("mAP: ", np.mean(APs))
'''
