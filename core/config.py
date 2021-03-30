import torch
from torch.multiprocessing import set_start_method

# preprocess
MAX_LENGTH = 49
WORD_COUNT_THRESHOLD = 1
NUM_OBJECT = 36
PAD_IDX = 0

IMAGE_MODEL = 'YOLOv5'
# IMAGE_MODEL = 'FasterRCNN'
CAPTION_MODEL = 'Transformer'
# CAPTION_MODEL = 'RL_Transformer'

MOVE_FIRST_IMAGE_FAETURE = True
SPLIT_POSITION = True

MODEL_NAME = 'maxlen49_36obj_1wordCount'
OUTPUT_NAME = 'maxlen49_36obj_1wordCount_128_24b_8h_SplitPosition'
print('OUTPUT_NAME: ', OUTPUT_NAME)

DATA_PATH = f'./data/{MODEL_NAME}'
OUTPUT_PATH = f'./output/{OUTPUT_NAME}'
WORD_TO_IDX_PATH = f'{DATA_PATH}/train/word_index.pkl'

print('Data :', MODEL_NAME)
print('Model : ', OUTPUT_NAME)

if torch.cuda.is_available():
    # set_start_method('spawn', force=True)

    device_name = "cuda:0"
    resnet_device = "cuda:1"
    frcnn_device = "cuda:1"

else:
    device_name = "cpu"
    frcnn_device = "cpu"
    resnet_device = "cpu"

RESNET_DEVICE = torch.device(resnet_device)
FRCNN_DEVICE = torch.device(frcnn_device)
DEVICE = torch.device(device_name)
print(f'Using {resnet_device} for ResNet101 and {frcnn_device} for FasterRCNN\n')
print(f'Using {device_name} for training model.')

# encoder
ENCODE_DIM_FEATURES = 2048

if IMAGE_MODEL == 'YOLOv5':
    ENCODE_DIM_POSITIONS = 84
elif IMAGE_MODEL == 'FasterRCNN':
    ENCODE_DIM_POSITIONS = 95

# solver
NUM_EPOCH = 1000
BATCH_SIZE = 64
DROPOUT = 0.5
LEARNING_RATE = 0.00005
LOG_PATH = f'./logs_{OUTPUT_NAME}/'


if OUTPUT_NAME == 'maxlen49_36obj_1wordCount_128_24b_8h_SplitPosition':
    assert SPLIT_POSITION
    assert NUM_OBJECT == 36
    assert IMAGE_MODEL == 'YOLOv5'
    assert MOVE_FIRST_IMAGE_FAETURE
    assert CAPTION_MODEL == 'Transformer'

    # encoder
    ENCODE_INPUT_SIZE = 64
    ENCODE_Q_K_DIM = 128
    ENCODE_V_DIM = 128
    ENCODE_HIDDEN_SIZE = 128
    ENCODE_NUM_BLOCKS = 2
    ENCODE_NUM_HEADS = 8

    # decoder
    DIM_WORD_EMBEDDING = 256
    DECODE_INPUT_SIZE = 64
    DECODE_Q_K_DIM = 128
    DECODE_V_DIM = 128
    DECODE_HIDDEN_SIZE = 128
    DECODE_NUM_BLOCKS = 4
    DECODE_NUM_HEADS = 8


if OUTPUT_NAME == 'maxlen49_36obj_1wordCount_256_25b_32h_RL':
    assert NUM_OBJECT == 36
    assert IMAGE_MODEL == 'YOLOv5'
    assert MOVE_FIRST_IMAGE_FAETURE
    assert CAPTION_MODEL == 'RL_Transformer'
    
    # encoder
    ENCODE_INPUT_SIZE = 256
    ENCODE_Q_K_DIM = 256
    ENCODE_V_DIM = 256
    ENCODE_HIDDEN_SIZE = 256
    ENCODE_NUM_BLOCKS = 2
    ENCODE_NUM_HEADS = 32

    # decoder
    DIM_WORD_EMBEDDING = 256
    DECODE_INPUT_SIZE = 256
    DECODE_Q_K_DIM = 256
    DECODE_V_DIM = 256
    DECODE_HIDDEN_SIZE = 256
    DECODE_NUM_BLOCKS = 5
    DECODE_NUM_HEADS = 32


if OUTPUT_NAME == 'maxlen49_36obj_1wordCount_256_25b_32h_FocalLoss_SplitPosition' or \
        OUTPUT_NAME == 'maxlen49_36obj_1wordCount_256_25b_32h_SplitPosition':
    assert SPLIT_POSITION
    assert NUM_OBJECT == 36
    assert IMAGE_MODEL == 'YOLOv5'
    assert MOVE_FIRST_IMAGE_FAETURE
    assert CAPTION_MODEL == 'Transformer'

    # encoder
    ENCODE_INPUT_SIZE = 256
    ENCODE_Q_K_DIM = 256
    ENCODE_V_DIM = 256
    ENCODE_HIDDEN_SIZE = 256
    ENCODE_NUM_BLOCKS = 2
    ENCODE_NUM_HEADS = 32

    # decoder
    DIM_WORD_EMBEDDING = 256
    DECODE_INPUT_SIZE = 256
    DECODE_Q_K_DIM = 256
    DECODE_V_DIM = 256
    DECODE_HIDDEN_SIZE = 256
    DECODE_NUM_BLOCKS = 5
    DECODE_NUM_HEADS = 32


if OUTPUT_NAME == 'maxlen49_36obj_1wordCount_256_25b_32h_EncoderMask' or\
        OUTPUT_NAME == 'maxlen49_36obj_1wordCount_256_25b_32h_FocalLoss':
    assert NUM_OBJECT == 36
    assert IMAGE_MODEL == 'YOLOv5'
    assert MOVE_FIRST_IMAGE_FAETURE
    assert CAPTION_MODEL == 'Transformer'
    assert not SPLIT_POSITION
    
    # encoder
    ENCODE_INPUT_SIZE = 256
    ENCODE_Q_K_DIM = 256
    ENCODE_V_DIM = 256
    ENCODE_HIDDEN_SIZE = 256
    ENCODE_NUM_BLOCKS = 2
    ENCODE_NUM_HEADS = 32

    # decoder
    DIM_WORD_EMBEDDING = 256
    DECODE_INPUT_SIZE = 256
    DECODE_Q_K_DIM = 256
    DECODE_V_DIM = 256
    DECODE_HIDDEN_SIZE = 256
    DECODE_NUM_BLOCKS = 5
    DECODE_NUM_HEADS = 32


if OUTPUT_NAME == 'maxlen49_36obj_1wordCount_move_3':
    assert NUM_OBJECT == 36
    assert IMAGE_MODEL == 'YOLOv5'
    assert MOVE_FIRST_IMAGE_FAETURE
    assert CAPTION_MODEL == 'Transformer'
    assert not SPLIT_POSITION
    
    # encoder
    ENCODE_INPUT_SIZE = 256
    ENCODE_Q_K_DIM = 512
    ENCODE_V_DIM = 512
    ENCODE_HIDDEN_SIZE = 1024
    ENCODE_NUM_BLOCKS = 3
    ENCODE_NUM_HEADS = 16

    # decoder
    DIM_WORD_EMBEDDING = 256
    DECODE_INPUT_SIZE = 256
    DECODE_Q_K_DIM = 512
    DECODE_V_DIM = 512
    DECODE_HIDDEN_SIZE = 1024
    DECODE_NUM_BLOCKS = 5
    DECODE_NUM_HEADS = 16


if OUTPUT_NAME == 'maxlen49_36obj_1wordCount_256_25b_32h_move':
    assert NUM_OBJECT == 36
    assert IMAGE_MODEL == 'YOLOv5'
    assert MOVE_FIRST_IMAGE_FAETURE
    assert CAPTION_MODEL == 'Transformer'
    assert not SPLIT_POSITION
    
    # encoder
    ENCODE_INPUT_SIZE = 256
    ENCODE_Q_K_DIM = 256
    ENCODE_V_DIM = 256
    ENCODE_HIDDEN_SIZE = 256
    ENCODE_NUM_BLOCKS = 2
    ENCODE_NUM_HEADS = 32

    # decoder
    DIM_WORD_EMBEDDING = 256
    DECODE_INPUT_SIZE = 256
    DECODE_Q_K_DIM = 256
    DECODE_V_DIM = 256
    DECODE_HIDDEN_SIZE = 256
    DECODE_NUM_BLOCKS = 5
    DECODE_NUM_HEADS = 32


if OUTPUT_NAME == 'maxlen49_36obj_1wordCount_1024_25b_32h_mask':
    assert NUM_OBJECT == 36
    assert IMAGE_MODEL == 'YOLOv5'
    assert not MOVE_FIRST_IMAGE_FAETURE
    assert CAPTION_MODEL == 'Transformer'
    assert not SPLIT_POSITION

    # encoder
    ENCODE_INPUT_SIZE = 1024
    ENCODE_Q_K_DIM = 1024
    ENCODE_V_DIM = 1024
    ENCODE_HIDDEN_SIZE = 2048
    ENCODE_NUM_BLOCKS = 2
    ENCODE_NUM_HEADS = 32

    # decoder
    DIM_WORD_EMBEDDING = 1024
    DECODE_INPUT_SIZE = 1024
    DECODE_Q_K_DIM = 1024
    DECODE_V_DIM = 1024
    DECODE_HIDDEN_SIZE = 2048
    DECODE_NUM_BLOCKS = 5
    DECODE_NUM_HEADS = 32


if OUTPUT_NAME == 'maxlen49_36obj_1wordCount_frcnn_256_25b_32h':
    assert NUM_OBJECT == 36
    assert IMAGE_MODEL == 'FasterRCNN'
    assert not MOVE_FIRST_IMAGE_FAETURE
    assert CAPTION_MODEL == 'Transformer'
    assert not SPLIT_POSITION

    # encoder
    ENCODE_INPUT_SIZE = 256
    ENCODE_Q_K_DIM = 256
    ENCODE_V_DIM = 256
    ENCODE_HIDDEN_SIZE = 256
    ENCODE_NUM_BLOCKS = 2
    ENCODE_NUM_HEADS = 32

    # decoder
    DIM_WORD_EMBEDDING = 256
    DECODE_INPUT_SIZE = 256
    DECODE_Q_K_DIM = 256
    DECODE_V_DIM = 256
    DECODE_HIDDEN_SIZE = 256
    DECODE_NUM_BLOCKS = 5
    DECODE_NUM_HEADS = 32

if OUTPUT_NAME == 'maxlen49_36obj_1wordCount_256_66b_32h':
    assert NUM_OBJECT == 36
    assert IMAGE_MODEL == 'YOLOv5'
    assert not MOVE_FIRST_IMAGE_FAETURE
    assert CAPTION_MODEL == 'Transformer'
    assert not SPLIT_POSITION

    # encoder
    ENCODE_INPUT_SIZE = 256
    ENCODE_Q_K_DIM = 256
    ENCODE_V_DIM = 256
    ENCODE_HIDDEN_SIZE = 256
    ENCODE_NUM_BLOCKS = 6
    ENCODE_NUM_HEADS = 32

    # decoder
    DIM_WORD_EMBEDDING = 256
    DECODE_INPUT_SIZE = 256
    DECODE_Q_K_DIM = 256
    DECODE_V_DIM = 256
    DECODE_HIDDEN_SIZE = 256
    DECODE_NUM_BLOCKS = 6
    DECODE_NUM_HEADS = 32


if OUTPUT_NAME == 'maxlen49_36obj_1wordCount_256_25b_32h_mask' \
        or OUTPUT_NAME == 'maxlen49_36obj_1wordCount_256_25b_32h_NoBias':
    assert NUM_OBJECT == 36
    assert IMAGE_MODEL == 'YOLOv5'
    assert not MOVE_FIRST_IMAGE_FAETURE
    assert CAPTION_MODEL == 'Transformer'
    assert not SPLIT_POSITION

    # encoder
    ENCODE_INPUT_SIZE = 256
    ENCODE_Q_K_DIM = 256
    ENCODE_V_DIM = 256
    ENCODE_HIDDEN_SIZE = 256
    ENCODE_NUM_BLOCKS = 2
    ENCODE_NUM_HEADS = 32

    # decoder
    DIM_WORD_EMBEDDING = 256
    DECODE_INPUT_SIZE = 256
    DECODE_Q_K_DIM = 256
    DECODE_V_DIM = 256
    DECODE_HIDDEN_SIZE = 256
    DECODE_NUM_BLOCKS = 5
    DECODE_NUM_HEADS = 32


if OUTPUT_NAME == 'maxlen49_36obj_1wordCount_128_14b_16h_mask':
    assert NUM_OBJECT == 36
    assert IMAGE_MODEL == 'YOLOv5'
    assert not MOVE_FIRST_IMAGE_FAETURE
    assert CAPTION_MODEL == 'Transformer'
    assert not SPLIT_POSITION

    # encoder
    ENCODE_INPUT_SIZE = 128
    ENCODE_Q_K_DIM = 128
    ENCODE_V_DIM = 128
    ENCODE_HIDDEN_SIZE = 256
    ENCODE_NUM_BLOCKS = 1
    ENCODE_NUM_HEADS = 16

    # decoder
    DIM_WORD_EMBEDDING = 256
    DECODE_INPUT_SIZE = 128
    DECODE_Q_K_DIM = 128
    DECODE_V_DIM = 128
    DECODE_HIDDEN_SIZE = 256
    DECODE_NUM_BLOCKS = 4
    DECODE_NUM_HEADS = 16


if OUTPUT_NAME == 'maxlen49_20obj_128_25b_32h':
    assert NUM_OBJECT == 20
    assert IMAGE_MODEL == 'YOLOv5'
    assert not MOVE_FIRST_IMAGE_FAETURE
    assert CAPTION_MODEL == 'Transformer'
    assert not SPLIT_POSITION

    # encoder
    ENCODE_INPUT_SIZE = 64
    ENCODE_Q_K_DIM = 128
    ENCODE_V_DIM = 128
    ENCODE_HIDDEN_SIZE = 128
    ENCODE_NUM_BLOCKS = 2
    ENCODE_NUM_HEADS = 32

    # decoder
    DIM_WORD_EMBEDDING = 256
    DECODE_INPUT_SIZE = 64
    DECODE_Q_K_DIM = 128
    DECODE_V_DIM = 128
    DECODE_HIDDEN_SIZE = 128
    DECODE_NUM_BLOCKS = 5
    DECODE_NUM_HEADS = 32


if OUTPUT_NAME == 'maxlen49_20obj_128_14b_16h' or \
        OUTPUT_NAME == 'maxlen49_20obj_128_14b_16h_mask' or \
        OUTPUT_NAME == 'maxlen49_20obj_128_14b_16h_mask_slower':
    assert NUM_OBJECT == 20
    assert IMAGE_MODEL == 'YOLOv5'
    assert not MOVE_FIRST_IMAGE_FAETURE
    assert CAPTION_MODEL == 'Transformer'
    assert not SPLIT_POSITION

    # encoder
    ENCODE_INPUT_SIZE = 128
    ENCODE_Q_K_DIM = 128
    ENCODE_V_DIM = 128
    ENCODE_HIDDEN_SIZE = 256
    ENCODE_NUM_BLOCKS = 1
    ENCODE_NUM_HEADS = 16

    # decoder
    DIM_WORD_EMBEDDING = 256
    DECODE_INPUT_SIZE = 128
    DECODE_Q_K_DIM = 128
    DECODE_V_DIM = 128
    DECODE_HIDDEN_SIZE = 256
    DECODE_NUM_BLOCKS = 4
    DECODE_NUM_HEADS = 16


if OUTPUT_NAME == 'maxlen49_64':
    assert NUM_OBJECT == 36
    assert IMAGE_MODEL == 'YOLOv5'
    assert not MOVE_FIRST_IMAGE_FAETURE
    assert CAPTION_MODEL == 'Transformer'
    assert not SPLIT_POSITION

    # encoder
    ENCODE_INPUT_SIZE = 64
    ENCODE_Q_K_DIM = 64
    ENCODE_V_DIM = 64
    ENCODE_HIDDEN_SIZE = 64
    ENCODE_NUM_BLOCKS = 1
    ENCODE_NUM_HEADS = 2

    # decoder
    DIM_WORD_EMBEDDING = 64
    DECODE_INPUT_SIZE = 64
    DECODE_Q_K_DIM = 64
    DECODE_V_DIM = 64
    DECODE_HIDDEN_SIZE = 64
    DECODE_NUM_BLOCKS = 3
    DECODE_NUM_HEADS = 2

if OUTPUT_NAME == 'maxlen49_128':
    assert NUM_OBJECT == 36
    assert IMAGE_MODEL == 'YOLOv5'
    assert not MOVE_FIRST_IMAGE_FAETURE
    assert CAPTION_MODEL == 'Transformer'
    assert not SPLIT_POSITION

    # encoder
    ENCODE_INPUT_SIZE = 64
    ENCODE_Q_K_DIM = 128
    ENCODE_V_DIM = 128
    ENCODE_HIDDEN_SIZE = 128
    ENCODE_NUM_BLOCKS = 2
    ENCODE_NUM_HEADS = 4

    # decoder
    DIM_WORD_EMBEDDING = 128
    DECODE_INPUT_SIZE = 64
    DECODE_Q_K_DIM = 128
    DECODE_V_DIM = 128
    DECODE_HIDDEN_SIZE = 128
    DECODE_NUM_BLOCKS = 4
    DECODE_NUM_HEADS = 4

if OUTPUT_NAME == 'maxlen49_128_14b':
    assert NUM_OBJECT == 36
    assert IMAGE_MODEL == 'YOLOv5'
    assert not MOVE_FIRST_IMAGE_FAETURE
    assert CAPTION_MODEL == 'Transformer'
    assert not SPLIT_POSITION

    # encoder
    ENCODE_INPUT_SIZE = 128
    ENCODE_Q_K_DIM = 128
    ENCODE_V_DIM = 128
    ENCODE_HIDDEN_SIZE = 128
    ENCODE_NUM_BLOCKS = 1
    ENCODE_NUM_HEADS = 4

    # decoder
    DIM_WORD_EMBEDDING = 128
    DECODE_INPUT_SIZE = 128
    DECODE_Q_K_DIM = 128
    DECODE_V_DIM = 128
    DECODE_HIDDEN_SIZE = 128
    DECODE_NUM_BLOCKS = 4
    DECODE_NUM_HEADS = 4

if OUTPUT_NAME == 'maxlen49_256_13b':
    assert NUM_OBJECT == 36
    assert IMAGE_MODEL == 'YOLOv5'
    assert not MOVE_FIRST_IMAGE_FAETURE
    assert CAPTION_MODEL == 'Transformer'
    assert not SPLIT_POSITION

    # encoder
    ENCODE_INPUT_SIZE = 128
    ENCODE_Q_K_DIM = 256
    ENCODE_V_DIM = 256
    ENCODE_HIDDEN_SIZE = 128
    ENCODE_NUM_BLOCKS = 1
    ENCODE_NUM_HEADS = 4

    # decoder
    DIM_WORD_EMBEDDING = 128
    DECODE_INPUT_SIZE = 128
    DECODE_Q_K_DIM = 256
    DECODE_V_DIM = 256
    DECODE_HIDDEN_SIZE = 128
    DECODE_NUM_BLOCKS = 3
    DECODE_NUM_HEADS = 4

if OUTPUT_NAME == 'maxlen49_128_14b_8h':
    assert NUM_OBJECT == 36
    assert IMAGE_MODEL == 'YOLOv5'
    assert not MOVE_FIRST_IMAGE_FAETURE
    assert CAPTION_MODEL == 'Transformer'
    assert not SPLIT_POSITION

    # encoder
    ENCODE_INPUT_SIZE = 128
    ENCODE_Q_K_DIM = 128
    ENCODE_V_DIM = 128
    ENCODE_HIDDEN_SIZE = 256
    ENCODE_NUM_BLOCKS = 1
    ENCODE_NUM_HEADS = 8

    # decoder
    DIM_WORD_EMBEDDING = 256
    DECODE_INPUT_SIZE = 128
    DECODE_Q_K_DIM = 128
    DECODE_V_DIM = 128
    DECODE_HIDDEN_SIZE = 256
    DECODE_NUM_BLOCKS = 4
    DECODE_NUM_HEADS = 8

if OUTPUT_NAME == 'maxlen49_128_14b_16h':
    assert NUM_OBJECT == 36
    assert IMAGE_MODEL == 'YOLOv5'
    assert not MOVE_FIRST_IMAGE_FAETURE
    assert CAPTION_MODEL == 'Transformer'
    assert not SPLIT_POSITION

    # encoder
    ENCODE_INPUT_SIZE = 128
    ENCODE_Q_K_DIM = 128
    ENCODE_V_DIM = 128
    ENCODE_HIDDEN_SIZE = 256
    ENCODE_NUM_BLOCKS = 1
    ENCODE_NUM_HEADS = 16

    # decoder
    DIM_WORD_EMBEDDING = 256
    DECODE_INPUT_SIZE = 128
    DECODE_Q_K_DIM = 128
    DECODE_V_DIM = 128
    DECODE_HIDDEN_SIZE = 256
    DECODE_NUM_BLOCKS = 4
    DECODE_NUM_HEADS = 16
