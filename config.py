import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
DATAROOT = 'E:/ML/test_data/pix2pix/test_set_01'
OUTPUT_DIRECTORY = 'E:/ML/test_data/pix2pix/test_output_01'
EXPERIMENT_NAME = 'test01'
DIRECTION = 'BtoA' # ['AtoB', 'BtoA']
LEARNING_RATE = 2e-4
BATCH_SIZE = 1
NUM_WORKERS = 4
IMAGE_SIZE = 256
INPUT_NC = 3
LOAD_SIZE = 512
CROP_SIZE = 256
LAMBDA_L1 = 100.0
LAMDA_SOBEL = 10.0
BETA1 = 0.5
NUM_EPOCHS = 3
LOAD_MODEL = False
SAVE_MODEL = True
MODEL_SAVE_RATE = 1 # Number of epochs between model saves
CHECKPOINT_DISC = "netD.pth.tar"
CHECKPOINT_GEN = "netG.pth.tar"

both_transform = A.Compose(
    [A.Resize(width=512, height=512), A.HorizontalFlip(p=0.5),], additional_targets={"image0": "image"},
)

transform_only_input = A.Compose(
    [
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0),
        ToTensorV2(),
    ]
)

transform_only_target = A.Compose(
    [
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0),
        ToTensorV2(),
    ]
)