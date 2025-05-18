import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
DIRECTION = 'BtoA' # ['AtoB', 'BtoA']
LEARNING_RATE = 2e-4
BATCH_SIZE = 1
NUM_WORKERS = 4
IMAGE_SIZE = 256
CHANNELS_IMG = 3
LAMBDA_L1 = 100.0
LAMDA_SOBEL = 10.0
NUM_EPOCHS = 100
LOAD_MODEL = False
SAVE_MODEL = True
MODEL_SAVE_RATE = 5 # Number of epochs between model saves
CHECKPOINT_DISC = "netD.pth.tar"
CHECKPOINT_GEN = "netG.pth.tar"

both_transform = A.Compose(
    [A.Resize(width=256, height=256), A.HorizontalFlip(p=0.5),], additional_targets={"image0": "image"},
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