import hickle
from torch.utils.data import Dataset

from core.utils import load_pickle, load_coco_data

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

        return self.data['features'][index], \
                self.data['positions'][index], \
                image_idx

    def __len__(self):
        return len(self.data['positions'])

    @property
    def len_image(self):
        return len(self.data['positions'])

    @property
    def data_dict(self):
        return self.data