"""
LERF configuration file.
"""

from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig, RAdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.plugins.types import MethodSpecification

from lerf.data.lerf_datamanager import LERFDataManagerConfig
from lerf.data.alpha_lerf_datamanager import AlphaLERFDataManagerConfig
from lerf.lerf import LERFModelConfig
from lerf.lerf_pipeline import LERFPipelineConfig

"""
Swap out the network config to use OpenCLIP or CLIP here.
"""
from lerf.encoders.alphaclip_encoder import AlphaCLIPNetworkConfig
from lerf.encoders.clip_encoder import CLIPNetworkConfig
from lerf.encoders.openclip_encoder import OpenCLIPNetworkConfig

lerf_method = MethodSpecification(
    config=TrainerConfig(
        method_name="lerf",
        steps_per_eval_batch=500,
        steps_per_save=2000,
        max_num_iterations=30000,
        mixed_precision=True,
        pipeline=LERFPipelineConfig(
            datamanager=LERFDataManagerConfig(
                dataparser=NerfstudioDataParserConfig(train_split_fraction=0.99),
                train_num_rays_per_batch=4096,
                eval_num_rays_per_batch=4096,
                camera_optimizer=CameraOptimizerConfig(
                    mode="SO3xR3", optimizer=AdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-2)
                ),
            ),
            model=LERFModelConfig(
                eval_num_rays_per_chunk=1 << 15,
                # NOTE: exceeding 16 layers per hashgrid causes a segfault within Tiny CUDA NN, so instead we compose multiple hashgrids together
                hashgrid_sizes=(19, 19),
                hashgrid_layers=(12, 12),
                hashgrid_resolutions=((16, 128), (128, 512)),
                num_lerf_samples=24,
            ),
            network=OpenCLIPNetworkConfig(
               clip_model_type="ViT-B-16", clip_model_pretrained="laion2b_s34b_b88k", clip_n_dims=512
            ),
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": None,
            },
            "fields": {
                "optimizer": RAdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-3, max_steps=30000),
            },
            "lerf": {
                "optimizer": RAdamOptimizerConfig(lr=1e-2, eps=1e-15, weight_decay=1e-9),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-3, max_steps=4000),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="Base config for LERF",
)
lerf_method_alpha = MethodSpecification(
    config=TrainerConfig(
        method_name="lerf-alpha",
        steps_per_eval_batch=500,
        steps_per_save=2000,
        max_num_iterations=30000,
        mixed_precision=True,
        pipeline=LERFPipelineConfig(
            datamanager=AlphaLERFDataManagerConfig(
                dataparser=NerfstudioDataParserConfig(train_split_fraction=0.99),
                train_num_rays_per_batch=4096,
                eval_num_rays_per_batch=4096,
                camera_optimizer=CameraOptimizerConfig(
                    mode="SO3xR3", optimizer=AdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-2)
                ),
            ),
            model=LERFModelConfig(
                eval_num_rays_per_chunk=1 << 15,
                # NOTE: exceeding 16 layers per hashgrid causes a segfault within Tiny CUDA NN, so instead we compose multiple hashgrids together
                hashgrid_sizes=(19, 19),
                hashgrid_layers=(12, 12),
                hashgrid_resolutions=((16, 128), (128, 512)),
                num_lerf_samples=24,
            ),
            network=AlphaCLIPNetworkConfig(
                clip_n_dims=512
            )
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": None,
            },
            "fields": {
                "optimizer": RAdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-3, max_steps=30000),
            },
            "lerf": {
                "optimizer": RAdamOptimizerConfig(lr=1e-2, eps=1e-15, weight_decay=1e-9),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-3, max_steps=4000),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="LERF with Alpha CLIP",
)
lerf_method_big = MethodSpecification(
    config=TrainerConfig(
        method_name="lerf-big",
        steps_per_eval_batch=500,
        steps_per_save=2000,
        max_num_iterations=30000,
        mixed_precision=True,
        pipeline=LERFPipelineConfig(
            datamanager=AlphaLERFDataManagerConfig(
                dataparser=NerfstudioDataParserConfig(train_split_fraction=0.99),
                train_num_rays_per_batch=4096,
                eval_num_rays_per_batch=4096,
                camera_optimizer=CameraOptimizerConfig(
                    mode="SO3xR3", optimizer=AdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-2)
                ),
            ),
            model=LERFModelConfig(
                eval_num_rays_per_chunk=1 << 15,
                # NOTE: exceeding 16 layers per hashgrid causes a segfault within Tiny CUDA NN, so instead we compose multiple hashgrids together
                hashgrid_sizes=(19, 19),
                hashgrid_layers=(12, 12),
                hashgrid_resolutions=((16, 128), (128, 512)),
                num_lerf_samples=24,
            ),
            network=AlphaCLIPNetworkConfig(
                clip_n_dims=512
            )
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": None,
            },
            "fields": {
                "optimizer": RAdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-3, max_steps=30000),
            },
            "lerf": {
                "optimizer": RAdamOptimizerConfig(lr=1e-2, eps=1e-15, weight_decay=1e-9),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-3, max_steps=4000),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="LERF with Alpha CLIP",
)


lerf_method_lite = MethodSpecification(
    config=TrainerConfig(
        method_name="lerf-clip",
        steps_per_eval_batch=500,
        steps_per_save=2000,
        max_num_iterations=30000,
        mixed_precision=True,
        pipeline=LERFPipelineConfig(
            datamanager=LERFDataManagerConfig(
                dataparser=NerfstudioDataParserConfig(train_split_fraction=0.99),
                train_num_rays_per_batch=4096,
                eval_num_rays_per_batch=4096,
                camera_optimizer=CameraOptimizerConfig(
                    mode="SO3xR3", optimizer=AdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-2)
                ),
            ),
            model=LERFModelConfig(
                eval_num_rays_per_chunk=1 << 15,
                # NOTE: exceeding 16 layers per hashgrid causes a segfault within Tiny CUDA NN, so instead we compose multiple hashgrids together
                hashgrid_sizes=(19, 19),
                hashgrid_layers=(12, 12),
                hashgrid_resolutions=((16, 128), (128, 512)),
                num_lerf_samples=24,
            ),

            network=CLIPNetworkConfig(
                 clip_model_type="ViT-B/16", clip_n_dims=512
            )
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": None,
            },
            "fields": {
                "optimizer": RAdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-3, max_steps=30000),
            },
            "lerf": {
                "optimizer": RAdamOptimizerConfig(lr=1e-2, eps=1e-15, weight_decay=1e-9),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-3, max_steps=4000),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="Original OAI CLIP, represented under lerf-clip",
)