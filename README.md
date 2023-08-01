# LERF: Language Embedded Radiance Fields
This is the official implementation for [LERF](https://lerf.io).


<div align='center'>
<img src="https://www.lerf.io/data/nerf_render.svg" height="230px">
</div>

# Installation
LERF follows the integration guidelines described [here](https://docs.nerf.studio/en/latest/developer_guides/new_methods.html) for custom methods within Nerfstudio. 
### 0. Install Nerfstudio dependencies
[Follow these instructions](https://docs.nerf.studio/en/latest/quickstart/installation.html) up to and including "tinycudann" to install dependencies and create an environment
### 1. Clone this repo
`git clone https://github.com/kerrj/lerf`
### 2. Install this repo as a python package
Navigate to this folder and run `python -m pip install -e .`

### 3. Run `ns-install-cli`

### Checking the install
Run `ns-train -h`: you should see a list of "subcommands" with lerf, lerf-big, and lerf-lite included among them.

# Using LERF
Now that LERF is installed you can play with it! 

- Launch training with `ns-train lerf --data <data_folder>`. This specifies a data folder to use. For more details, see [Nerfstudio documentation](https://docs.nerf.studio/en/latest/quickstart/first_nerf.html). 
- Connect to the viewer by forwarding the viewer port (we use VSCode to do this), and click the link to `viewer.nerf.studio` provided in the output of the train script
- Within the viewer, you can type text into the textbox, then select the `relevancy_0` output type to visualize relevancy maps.

## Relevancy Map Normalization
By default, the viewer shows **raw** relevancy scaled with the turbo colormap. As values lower than 0.5 correspond to irrelevant regions, **we recommend setting the `range` parameter to (-1.0, 1.0)**. To match the visualization from the paper, check the `Normalize` tick-box, which stretches the values to use the full colormap.

The images below show the rgb, raw, centered, and normalized output views for the query "Lily".


<div align='center'>
<img src="readme_images/lily_rgb.jpg" width="150px">
<img src="readme_images/lily_raw.jpg" width="150px">
<img src="readme_images/lily_centered.jpg" width="150px">
<img src="readme_images/lily_normalized.jpg" width="150px">
</div>


## Resolution
The Nerfstudio viewer dynamically changes resolution to achieve a desired training throughput.

**To increase resolution, pause training**. Rendering at high resolution (512 or above) can take a second or two, so we recommend rendering at 256px
## `lerf-big` and `lerf-lite`
If your GPU is struggling on memory, we provide a `lerf-lite` implementation that reduces the LERF network capacity and number of samples along rays. If you find you still need to reduce memory footprint, the most impactful parameters for memory are `num_lerf_samples`, hashgrid levels, and hashgrid size.

`lerf-big` provides a larger model that uses ViT-L/14 instead of ViT-B/16 for those with large memory GPUs.

# Extending LERF
Be mindful that code for visualization will change as more features are integrated into Nerfstudio, so if you fork this repo and build off of it, check back regularly for extra changes.
### Issues
Please open Github issues for any installation/usage problems you run into. We've tried to support as broad a range of GPUs as possible with `lerf-lite`, but it might be necessary to provide even more low-footprint versions. Thank you!
#### Known TODOs
- [ ] Integrate into `ns-render` commands to render videos from the command line with custom prompts
### Using custom image encoders
We've designed the code to modularly accept any image encoder that implements the interface in `BaseImageEncoder` (`image_encoder.py`). An example of different encoder implementations can be seen in `clip_encoder.py` vs `openclip_encoder.py`, which implement OpenAI's CLIP and OpenCLIP respectively.
### Code structure
(TODO expand this section)
The main file to look at for editing and building off LERF is `lerf.py`, which extends the Nerfacto model from Nerfstudio, adds an additional language field, losses, and visualization. The CLIP and DINO pre-processing are carried out by `pyramid_interpolator.py` and `dino_dataloader.py`.

## Bibtex
If you find this useful, please cite the paper!
<pre id="codecell0">@inproceedings{lerf2023,
&nbsp;author = {Kerr, Justin and Kim, Chung Min and Goldberg, Ken and Kanazawa, Angjoo and Tancik, Matthew},
&nbsp;title = {LERF: Language Embedded Radiance Fields},
&nbsp;booktitle = {International Conference on Computer Vision (ICCV)},
&nbsp;year = {2023},
} </pre>
