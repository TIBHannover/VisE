import torch
import torchvision
import logging


class ResNet50(torch.nn.Module):
    def __init__(self, num_classes, model_type, redundancy_removal):
        super(ResNet50, self).__init__()
        self._num_classes = num_classes
        self.model_type = model_type
        self.redundancy_removal = redundancy_removal

        resnet_model = torchvision.models.resnet.resnet50(pretrained=False)
        self._features = torch.nn.Sequential(*list(resnet_model.children())[:-1])
        self._fc = torch.nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self._features(x)
        x = torch.flatten(x, 1)
        features = x
        x = self._fc(x)
        return {"predictions": torch.sigmoid(x), "features": features}

    def load(self, path, device="cpu"):

        state_dict = torch.load(path, map_location=device)
        try:
            self.load_state_dict(state_dict["model"])
        except RuntimeError:
            logging.warn("Trainer: Save DataParallel model without using module")
            map_dict = {}
            for key, value in state_dict["model"].items():
                if key.split(".")[0] == "module":
                    map_dict[".".join(key.split(".")[1:])] = value
                else:
                    map_dict["module." + key] = value
            self.load_state_dict(map_dict)
