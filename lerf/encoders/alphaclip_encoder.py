from dataclasses import dataclass, field
from typing import Tuple, Type

import torch
import torchvision

try:
    import alpha_clip
except ImportError:
    assert False, "alpha_clip is not installed, install it with https://github.com/SunzeY/AlphaCLIP"

from lerf.encoders.image_encoder import (BaseImageEncoder, BaseImageEncoderConfig)
from nerfstudio.viewer.server.viewer_elements import ViewerText

@dataclass
class AlphaCLIPNetworkConfig(BaseImageEncoderConfig):
    _target: Type = field(default_factory=lambda: AlphaCLIPNetwork)
    clip_model_type: str = "ViT-B/16"
    clip_n_dims: int = 512
    negatives: Tuple[str] = ("object", "things", "stuff", "texture")


class AlphaCLIPNetwork(BaseImageEncoder):
    def __init__(self, config: AlphaCLIPNetworkConfig):
        super().__init__()
        self.config = config

        model, preprocess = alpha_clip.load("ViT-B/16", alpha_vision_ckpt_pth="/home/ahojel/project/lerf/alpha_clip/clip_b16_grit+mim_fultune_4xe.pth", lora_adapt=False, rank=-1)
 
        self.process = preprocess

        model.eval()
        self.tokenizer = alpha_clip.tokenize
        self.model = model.to("cuda")
        self.clip_n_dims = self.config.clip_n_dims

        self.positive_input = ViewerText("LERF Positives", "", cb_hook=self.gui_cb)

        self.positives = self.positive_input.value.split(";")
        self.negatives = self.config.negatives
        with torch.no_grad():
            tok_phrases = torch.cat([self.tokenizer(phrase) for phrase in self.positives]).to("cuda")
            self.pos_embeds = model.encode_text(tok_phrases)
            tok_phrases = torch.cat([self.tokenizer(phrase) for phrase in self.negatives]).to("cuda")
            self.neg_embeds = model.encode_text(tok_phrases)
        self.pos_embeds /= self.pos_embeds.norm(dim=-1, keepdim=True)
        self.neg_embeds /= self.neg_embeds.norm(dim=-1, keepdim=True)

        assert (
            self.pos_embeds.shape[1] == self.neg_embeds.shape[1]
        ), "Positive and negative embeddings must have the same dimensionality"
        assert (
            self.pos_embeds.shape[1] == self.clip_n_dims
        ), "Embedding dimensionality must match the model dimensionality"

    @property
    def name(self) -> str:
        return "alphaclip_{}".format(self.config.clip_model_type)

    @property
    def embedding_dim(self) -> int:
        return self.config.clip_n_dims

    def gui_cb(self,element):
        self.set_positives(element.value.split(";"))

    def set_positives(self, text_list):
        self.positives = text_list
        with torch.no_grad():
            tok_phrases = torch.cat([self.tokenizer(phrase) for phrase in self.positives]).to("cuda")
            self.pos_embeds = self.model.encode_text(tok_phrases)
        self.pos_embeds /= self.pos_embeds.norm(dim=-1, keepdim=True)

    def get_relevancy(self, embed: torch.Tensor, positive_id: int) -> torch.Tensor:
        phrases_embeds = torch.cat([self.pos_embeds, self.neg_embeds], dim=0)
        p = phrases_embeds.to(embed.dtype)  # phrases x 512
        output = torch.mm(embed, p.T)  # rays x phrases
        positive_vals = output[..., positive_id : positive_id + 1]  # rays x 1
        negative_vals = output[..., len(self.positives) :]  # rays x N_phrase
        repeated_pos = positive_vals.repeat(1, len(self.negatives))  # rays x N_phrase

        sims = torch.stack((repeated_pos, negative_vals), dim=-1)  # rays x N-phrase x 2
        softmax = torch.softmax(10 * sims, dim=-1)  # rays x n-phrase x 2
        best_id = softmax[..., 0].argmin(dim=1)  # rays x 2
        return torch.gather(softmax, 1, best_id[..., None, None].expand(best_id.shape[0], len(self.negatives), 2))[
            :, 0, :
        ]

    def encode_image(self, input_image, input_binary_mask):
        # Transformation for the masks
        mask_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(), 
            torchvision.transforms.Resize((224, 224)), 
            torchvision.transforms.Normalize(0.5, 0.26)
        ])

        # Apply the transformation to the entire batch of masks
        # Stack the masks to create a batch and multiply by 255
        masks_batch = torch.stack([mask_transform(mask * 255) for mask in input_binary_mask])

        # Unsqueeze the input_image to add the batch dimension and repeat it for each mask
        input_image_batch = input_image.unsqueeze(0).repeat(masks_batch.size(0), 1, 1, 1)

        # Send the batched images and masks to the model
        # Assuming the model can handle batched inputs
        output = self.model.visual(input_image_batch.to("cuda"), masks_batch.to("cuda").half())

        return output
