import hickle
from torch.utils.data import Dataset

from core.utils import load_pickle, load_coco_data
from core.preprocess import image_feature_YOLOv5, image_feature_FasterRCNN
from core.config import MAX_OBJ

class TrainDataset(Dataset):
    def __init__(self, data_path, split):
        self.data = load_coco_data(data_path=data_path, split=split)

    def __getitem__(self, index):
        image_idx = self.data['image_idxs'][index]

        return self.data['features'][image_idx], \
                self.data['positions'][image_idx], \
                self.data['captions'][index], \
                image_idx

    def __len__(self):
        return len(self.data['captions'])

    @property
    def len_image(self):
        return len(self.data['positions'])

    @property
    def data_dict(self):
        return self.data


class TestDataset(Dataset):
    def __init__(self, data_path, split='test'):
        self.data = load_coco_data(data_path=data_path, split=split)

    def __getitem__(self, index):
        image_idx = self.data['image_idxs'][index]

        return self.data['features'][image_idx], \
                self.data['positions'][image_idx], \
                image_idx

    def __len__(self):
        return len(self.data['captions'])

    @property
    def len_image(self):
        return len(self.data['positions'])

    @property
    def data_dict(self):
        return self.data


class ImagePreprocessDataset(Dataset):
    def __init__(self, path_list, model):
        self.path_list = path_list
        self.model = model

    def __getitem__(self, index):
        path = self.path_list[index]

        if self.model == 'YOLOv5':
            features, positions, _ = image_feature_YOLOv5(image_path=path, max_obj=MAX_OBJ)
        elif self.model == 'FasterRCNN':
            features, positions, _ = image_feature_FasterRCNN(image_path=path)

        return features, positions

    def __len__(self):
        return len(self.path_list)
