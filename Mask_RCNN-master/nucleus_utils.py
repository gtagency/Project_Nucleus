import os
import sys
import tqdm

from config import Config
import utils

from skimage.transform import resize
from skimage.io import imread

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

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 256

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 600

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 50

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
        TRAIN_PATH = '../input/st1_train/'
        IMG_HEIGHT = 128
        IMG_WIDTH = 128
        IMG_CHANNELS = 3

        train_ids = next(os.walk(TRAIN_PATH))[1]

        #self.X_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
        self.X_train = []
        #Y_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
        self.Y_train = []

        print('Getting and resizing images and masks ... ')
        sys.stdout.flush()
        j = 0
        for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
            if n in image_id_list:
                path = TRAIN_PATH + id_
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
        print('Done!')

    def load_image(self, image_id):
        """Generate an image from the specs of the given image ID.
        Typically this function loads the image from a file, but
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
