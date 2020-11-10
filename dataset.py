import imageio
import numpy as np
import os
import torch
import torchvision
import utils


class EventDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, testset_path):
        self.image_dir = image_dir

        # Read dataset
        self.dataset = utils.read_jsonl(
            testset_path,
            keep_keys={
                "id": "id",
                "wikidata_id": "wikidata_id",
                "leaf_class_idx": "leaf_class_idx",
                "leaf_wd_id": "leaf_wd_id",
            },
        )

        # Build image transformation
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize(size=224),
            torchvision.transforms.CenterCrop(size=224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __getitem__(self, idx):
        x = self.dataset[idx]
        path = os.path.join(self.image_dir, x["id"][0:2], x["id"][2:4], x["id"] + ".jpg")
        if not os.path.exists(path):
            print(path)
            exit()
            return self[(idx + 1) % len(self)]

        img = imageio.imread(path)

        if self.transform:
            img = self.transform(img)

        return {**x, "image": img}

    def __len__(self):
        return len(self.dataset)


class InferDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths):
        self.image_paths = image_paths

        # Build image transformation
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize(size=224),
            torchvision.transforms.CenterCrop(size=224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]

        img = imageio.imread(image_path)
        if len(img.shape) == 2:
            img = np.stack([img] * 3)

        if len(img.shape) == 3:
            if img.shape[-1] > 3:
                img = img[..., 0:3]
            if img.shape[-1] < 3:
                img = np.stack([img[..., 0]] * 3)

        if self.transform:
            img = self.transform(img)

        return {"image_path": image_path, "image": img}

    def __len__(self):
        return len(self.image_paths)
