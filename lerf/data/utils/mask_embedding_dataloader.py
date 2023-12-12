import json

import numpy as np
import torch
from lerf.data.utils.feature_dataloader import FeatureDataloader
from lerf.encoders.image_encoder import BaseImageEncoder
from tqdm import tqdm

import matplotlib.pyplot as plt 
import torchvision.transforms as T
import pdb


class MaskEmbeddingDataloader(FeatureDataloader):
    def __init__(
        self,
        cfg: dict,
        device: torch.device,
        model: BaseImageEncoder,
        image_list: torch.Tensor = None,
        cache_path: str = None,
    ):
        assert "tile_ratio" in cfg
        assert "stride_ratio" in cfg
        assert "image_shape" in cfg
        assert "model_name" in cfg

        self.tile_ratio = cfg["tile_ratio"]
        self.kernel_size = int(cfg["image_shape"][0] * self.tile_ratio)
        self.stride = int(self.kernel_size * cfg["stride_ratio"])
        self.padding = self.kernel_size // 2
        self.center_x = (
            (self.kernel_size - 1) / 2
            - self.padding
            + self.stride
            * np.arange(
                np.floor((cfg["image_shape"][0] + 2 * self.padding - (self.kernel_size - 1) - 1) / self.stride + 1)
            )
        )
        self.center_y = (
            (self.kernel_size - 1) / 2
            - self.padding
            + self.stride
            * np.arange(
                np.floor((cfg["image_shape"][1] + 2 * self.padding - (self.kernel_size - 1) - 1) / self.stride + 1)
            )
        )
        self.center_x = torch.from_numpy(self.center_x).half()
        self.center_y = torch.from_numpy(self.center_y).half()
        self.start_x = self.center_x[0].float()
        self.start_y = self.center_y[0].float()

        self.model = model
        self.embed_size = self.model.embedding_dim
        super().__init__(cfg, device, image_list, cache_path)

    def load(self):
        cache_info_path = self.cache_path.with_suffix(".info")
        if not cache_info_path.exists():
            raise FileNotFoundError
        with open(cache_info_path, "r") as f:
            cfg = json.loads(f.read())
        if cfg != self.cfg:
            raise ValueError("Config mismatch")
        self.data = torch.from_numpy(np.load(self.cache_path)).half()

    def create(self, image_list):
        assert self.model is not None, "model must be provided to generate features"
        assert image_list is not None, "image_list must be provided to generate features"

        unfold_func = torch.nn.Unfold(
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
        ).to(self.device)

        img_embeds = []
        for img in tqdm(image_list, desc="Embedding images", leave=False):
            img_embeds.append(self._embed_clip_tiles(img.unsqueeze(0), unfold_func))
        self.data = torch.from_numpy(np.stack(img_embeds)).half()

    def __call__(self, img_points):
        # img_points: (B, 3) # (img_ind, x, y) (img_ind, row, col)
        # return: (B, 512)
        img_points = img_points.cpu()
        img_ind, img_points_x, img_points_y = img_points[:, 0], img_points[:, 1], img_points[:, 2]

        x_ind = torch.floor((img_points_x - (self.start_x)) / self.stride).long()
        y_ind = torch.floor((img_points_y - (self.start_y)) / self.stride).long()
        return self._interp_inds(img_ind, x_ind, y_ind, img_points_x, img_points_y)

    def _interp_inds(self, img_ind, x_ind, y_ind, img_points_x, img_points_y):
        img_ind = img_ind.to(self.data.device)  # self.data is on cpu to save gpu memory, hence this line
        topleft = self.data[img_ind, x_ind, y_ind].to(self.device)
        topright = self.data[img_ind, x_ind + 1, y_ind].to(self.device)
        botleft = self.data[img_ind, x_ind, y_ind + 1].to(self.device)
        botright = self.data[img_ind, x_ind + 1, y_ind + 1].to(self.device)

        x_stride = self.stride
        y_stride = self.stride
        right_w = ((img_points_x - (self.center_x[x_ind])) / x_stride).to(self.device)  # .half()
        top = torch.lerp(topleft, topright, right_w[:, None])
        bot = torch.lerp(botleft, botright, right_w[:, None])

        bot_w = ((img_points_y - (self.center_y[y_ind])) / y_stride).to(self.device)  # .half()
        return torch.lerp(top, bot, bot_w[:, None])
    
    def _create_binary_masks_vectorized(self, image_shape):
        """
        Create a binary mask for each tile to indicate its occupancy in the original image.
        Vectorized for performance.
        """
        image_h, image_w = image_shape
        kernel_size = self.kernel_size
        stride = self.stride
        padding = self.padding
        # Calculate padded image dimensions
        padded_h = image_h + 2 * padding
        padded_w = image_w + 2 * padding

        # Generate a grid of coordinates for the padded image
        grid_y, grid_x = torch.meshgrid(torch.arange(padded_h), torch.arange(padded_w), indexing='ij')

        # Calculate the coordinates of the top-left corner of each window
        window_tops_y = torch.arange(0, padded_h - kernel_size + 1, stride)
        window_tops_x = torch.arange(0, padded_w - kernel_size + 1, stride)

        # Create a list to hold the masks
        masks = []

        # Determine which grid points fall within each window
        for y in window_tops_y:
            for x in window_tops_x:
                mask = (grid_y >= y) & (grid_y < y + kernel_size) & (grid_x >= x) & (grid_x < x + kernel_size).int()

                # Remove padding from the mask
                mask = mask[padding:-padding, padding:-padding] if padding > 0 else mask
                masks.append(mask)

        # Stack the masks into a single tensor
        return torch.stack(masks)

    def _embed_clip_tiles(self, image, unfold_func):
        # image augmentation: slow-ish (0.02s for 600x800 image per augmentation)
        # aug_imgs = torch.cat([image])

        # tiles = unfold_func(aug_imgs).permute(2, 0, 1).reshape(-1, 3, self.kernel_size, self.kernel_size).to("cuda")

        # breakpoint()

        binary_masks_of_tiles = self._create_binary_masks_vectorized(image.shape[-2:])

        #display_image = image[0].permute(1, 2, 0).cpu().numpy() # Adjust if needed


        # for i in range(tiles.shape[0]):
        #     current_tile = tiles[i].cpu().numpy().transpose(1, 2, 0)  # Adjust if needed
        #     current_mask = binary_masks_of_tiles[i].cpu().numpy()

        #     # Plotting
        #     plt.figure(figsize=(12, 4))

        #     # Plot original image
        #     plt.subplot(1, 3, 1)
        #     plt.imshow(display_image)
        #     plt.title('Original Image')
        #     plt.axis('off')

        #     # Plot current tile
        #     plt.subplot(1, 3, 2)
        #     plt.imshow(current_tile)
        #     plt.title(f'Tile {i+1}')
        #     plt.axis('off')

        #     # Plot current binary mask
        #     plt.subplot(1, 3, 3)
        #     plt.imshow(current_mask, cmap='gray')  # Only the single channel
        #     plt.title(f'Binary Mask {i+1}')
        #     plt.axis('off')

        #     # Save the plot to a file
        #     save_path = '/home/ahojel/project/plot.png'  # Change this to your desired path
        #     plt.savefig(save_path, bbox_inches='tight')
        #     plt.close()

        #     breakpoint()

        with torch.no_grad():
            processed_image = self.model.process(T.ToPILImage()(image[0])).half().to(self.device)
            clip_embeds = self.model.encode_image(processed_image, binary_masks_of_tiles.numpy().astype(float))
        clip_embeds /= clip_embeds.norm(dim=-1, keepdim=True)

        clip_embeds = clip_embeds.reshape((self.center_x.shape[0], self.center_y.shape[0], -1))
        clip_embeds = torch.concat((clip_embeds, clip_embeds[:, [-1], :]), dim=1)
        clip_embeds = torch.concat((clip_embeds, clip_embeds[[-1], :, :]), dim=0)
        return clip_embeds.detach().cpu().numpy()
