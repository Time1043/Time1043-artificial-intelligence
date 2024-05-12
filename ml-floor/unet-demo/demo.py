import os

WORK_SPACE = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(WORK_SPACE, 'data')
RESULTS_ROOT = os.path.join(WORK_SPACE, 'results')

print("Working dir: {}, \nData dir: {}, \nCurrent contents: {}, \nResults dir: {}, \n"
      .format(WORK_SPACE, DATA_ROOT, os.listdir(DATA_ROOT), RESULTS_ROOT))

TRAIN_IMG_DIR = os.path.join(DATA_ROOT, "train_image")
TRAIN_MASK_DIR = os.path.join(DATA_ROOT, "train_mask")
VAL_IMG_DIR = os.path.join(DATA_ROOT, "val_image")
VAL_MASK_DIR = os.path.join(DATA_ROOT, "val_mask")
print("train image: {}, {}, \ntrain mask: {}, {}, \nval image: {}, {}, \nval mask: {}, {}, \n".format(
    TRAIN_IMG_DIR, len(os.listdir(TRAIN_IMG_DIR)), TRAIN_MASK_DIR, len(os.listdir(TRAIN_MASK_DIR)),
    VAL_IMG_DIR, len(os.listdir(VAL_IMG_DIR)), VAL_MASK_DIR, len(os.listdir(VAL_MASK_DIR))))

print(os.listdir(TRAIN_IMG_DIR)[:5])
print(os.listdir(TRAIN_MASK_DIR)[:5])
