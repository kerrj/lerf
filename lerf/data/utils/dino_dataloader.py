import typing

import torch
from lerf.data.utils.dino_extractor import ViTExtractor
from lerf.data.utils.feature_dataloader import FeatureDataloader
from tqdm import tqdm
import torchvision.transforms as transforms


class DinoDataloader(FeatureDataloader):
    dino_model_type = "dino_vits8"
    dino_stride = 8
    # # For dinov2, use this:
    # dino_model_type = "dinov2_vitb14"
    # dino_stride = 14
    dino_load_size = 500
    dino_layer = 11
    dino_facet = "key"
    dino_bin = False

    def __init__(
        self,
        cfg: dict,
        device: torch.device,
        image_list: torch.Tensor,
        cache_path: str = None,
    ):
        assert "image_shape" in cfg
        super().__init__(cfg, device, image_list, cache_path)
        # Do distillation-preprocessing as noted in N3F:
        # The features are then L2-normalized and reduced with PCA to 64 dimensions before distillation.
        data_shape = self.data.shape
        self.data = self.data / self.data.norm(dim=-1, keepdim=True) 
        self.data = torch.pca_lowrank(self.data.reshape(-1, data_shape[-1]), q=64)[0].reshape((*data_shape[:-1], 64))

    def create(self, image_list):
        extractor = ViTExtractor(self.dino_model_type, self.dino_stride)
        preproc_image_lst = extractor.preprocess(image_list, self.dino_load_size)[0].to(self.device)

        dino_embeds = []
        for image in tqdm(preproc_image_lst, desc="dino", total=len(image_list), leave=False):
            # image nees to be resized s.t. H, W are divisible by dino_stride
            if "dinov2" in self.dino_model_type:
                image = transforms.Resize((
                    (image.shape[1]//self.dino_stride)*self.dino_stride,
                    (image.shape[2]//self.dino_stride)*self.dino_stride,
                    ))(image)
            with torch.no_grad():
                descriptors = extractor.extract_descriptors(
                    image.unsqueeze(0),
                    [self.dino_layer],
                    self.dino_facet,
                    self.dino_bin,
                )
            descriptors = descriptors.reshape(extractor.num_patches[0], extractor.num_patches[1], -1)
            dino_embeds.append(descriptors.cpu().detach())

        self.data = torch.stack(dino_embeds, dim=0)

    def __call__(self, img_points):
        # img_points: (B, 3) # (img_ind, x, y)
        img_scale = (
            self.data.shape[1] / self.cfg["image_shape"][0],
            self.data.shape[2] / self.cfg["image_shape"][1],
        )
        x_ind, y_ind = (img_points[:, 1] * img_scale[0]).long(), (img_points[:, 2] * img_scale[1]).long()
        return (self.data[img_points[:, 0].long(), x_ind, y_ind]).to(self.device)
