import torch
import torch.utils.data
import torchvision

import cv2
import numpy as np

class Cityscapes(torch.utils.data.Dataset):
    """
        Native PyTorch Cityscapes but with proper data augmentation.
    """
    def __init__(
        self, root='/media/hpc4_Raid/e_burkov/Datasets/Cityscapes/',
        split='train', size=(1024, 512), augmented=False):

        super().__init__()

        self.cityscapes = torchvision.datasets.Cityscapes(
            root=root, split=split, mode='fine', target_type='semantic')
        
        self.size = size
        self.n_classes = 19

        # precomputed mean and stddev per channel
        self.mean = np.float32([0.28470638394356, 0.32577008008957, 0.28766867518425]) * 255.0
        self.std  = np.float32([0.18671783804893, 0.1899059265852,  0.18665011227131]) * 255.0

        # precomputed class frequencies
        class_probs = np.float32([
            0.36869695782661,
            0.060849856585264,
            0.22824048995972,
            0.0065539856441319,
            0.0087727159261703,
            0.012273414991796,
            0.0020779478363693,
            0.0055127013474703,
            0.1592865139246,
            0.011578181758523,
            0.040189824998379,
            0.012189572677016,
            0.0013512192526832,
            0.069945447146893,
            0.0026745572686195,
            0.0023519159294665,
            0.0023290426470339,
            0.00098657899070531,
            0.0041390685364604,
        ])
        # add "void" class and adopt a slightly modified class weighting scheme from ENet
        self.class_weights = np.concatenate(([0], 1.0 / np.log(class_probs + 1.1)))
        self.class_weights = torch.tensor(self.class_weights, dtype=torch.float32)

        self.augmented = augmented

    @staticmethod
    def augment(image, labels):
        """
            image: np.uint8, H x W x 3
            labels: np.uint8, H x W
        """
        flip = bool(np.random.randint(2))
        maybe_flip_matrix = np.eye(3)
        if flip:
            maybe_flip_matrix[0,0] = -1
            maybe_flip_matrix[0,2] = labels.shape[1]
        
        angle = (np.random.rand() * 2 - 1) * 7.5
        scale_factor = (np.random.rand() * 2 - 1) * 0.12 + 1.0
        image_center = (labels.shape[1] / 2, labels.shape[0] / 2)
        rotation_matrix = np.eye(3)
        rotation_matrix[:2] = cv2.getRotationMatrix2D(image_center, angle, scale_factor)
        
        transformation_matrix = (maybe_flip_matrix @ rotation_matrix)[:2]
        
        image_size = (labels.shape[1], labels.shape[0])

        image = cv2.warpAffine(
            image, transformation_matrix, image_size,
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        labels = cv2.warpAffine(
            labels, transformation_matrix, image_size,
            flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)

        return image, labels
        
    def __len__(self):
        return len(self.cityscapes)

    def __getitem__(self, idx):
        image, labels = map(np.array, self.cityscapes[idx])

        if image.shape[:2] != self.size[::-1]:
            image = cv2.resize(image, self.size, interpolation=cv2.INTER_AREA)
        if labels.shape != self.size[::-1]:
            labels = cv2.resize(labels, self.size, interpolation=cv2.INTER_NEAREST)

        def remap_labels(labels):
            """
            Shift labels according to
            https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py#L61
            """
            retval = np.zeros_like(labels)
            class_map = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
            for new_class, old_class in enumerate(class_map, start=1):
                retval[labels == old_class] = new_class
            return retval

        # comment this line if you have already preprocessed the labels this way
        labels = remap_labels(labels)

        if self.augmented:
            image, labels = self.augment(image, labels)

        image = image.transpose((2, 0, 1)).astype(np.float32)
        image -= self.mean.reshape(3, 1, 1)
        image *= 1 / self.std.reshape(3, 1, 1)

        return torch.tensor(image), torch.tensor(labels, dtype=torch.long)


if __name__ == '__main__':
    dataset = Cityscapes('/home/shrubb/Datasets/Cityscapes', augmented=True)
    print(len(dataset))

