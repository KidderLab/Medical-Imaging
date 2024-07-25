# Medical Imaging


**Dreambooth Stable Diffusion Notebook:**
[Notebook](https://github.com/ShivamShrirao/diffusers/blob/main/examples/dreambooth/DreamBooth_Stable_Diffusion.ipynb).
This jupyter [notebook](https://github.com/ShivamShrirao/diffusers/blob/main/examples/dreambooth/DreamBooth_Stable_Diffusion.ipynb) can be run in [colab](https://colab.research.google.com/) to perform training and synthesis of medical images. The code can also be found below.

# Dreambooth Stable Diffusion Training

**Check GPU Availability**

```bash
!nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
```

**Download Training Scripts**

```bash
!wget -q https://github.com/ShivamShrirao/diffusers/raw/main/examples/dreambooth/train_dreambooth.py
!wget -q https://github.com/ShivamShrirao/diffusers/raw/main/scripts/convert_diffusers_to_original_stable_diffusion.py
```

**Installation**

```bash
%pip install -qq git+https://github.com/ShivamShrirao/diffusers
%pip install -q -U --pre triton
%pip install -q accelerate transformers ftfy bitsandbytes==0.35.0 gradio natsort safetensors xformers
```

**Hugging Face Login**

```bash
!mkdir -p ~/.huggingface
HUGGINGFACE_TOKEN = "<your_token_here>"
!echo -n "$HUGGINGFACE_TOKEN" > ~/.huggingface/token
```

**Set Up Directories**

```bash
save_to_gdrive = False
if save_to_gdrive:
    from google.colab import drive
    drive.mount('/content/drive')

MODEL_NAME = "runwayml/stable-diffusion-v1-5"
OUTPUT_DIR = "/content/stable_diffusion_weights/Brain_pituitary_tumor_cross_MRI_kaggle_04192023"
if save_to_gdrive:
    OUTPUT_DIR = "/content/drive/MyDrive/" + OUTPUT_DIR
```

**Start Training**

```
# You can also add multiple concepts here. Try tweaking `--max_train_steps` accordingly.
concepts_list = [
    {
        "instance_prompt":      "photo of zwx dog",
        "class_prompt":         "photo of a dog",
        "instance_data_dir":    "/content/data/zwx",
        "class_data_dir":       "/content/data/dog"
    },

# `class_data_dir` contains regularization images
import json
import os
for c in concepts_list:
    os.makedirs(c["instance_data_dir"], exist_ok=True)

with open("concepts_list.json", "w") as f:
    json.dump(concepts_list, f, indent=4)
```

```python
#Upload your images by running this cell.
#OR
#You can use the file manager on the left panel to upload (drag and drop) to each `instance_data_dir` (it uploads faster). You can also upload your own class images in `class_data_dir` if u don't wanna generate with SD.

import os
from google.colab import files
import shutil

for c in concepts_list:
    print(f"Uploading instance images for `{c['instance_prompt']}`")
    uploaded = files.upload()
    for filename in uploaded.keys():
        dst_path = os.path.join(c['instance_data_dir'], filename)
        shutil.move(filename, dst_path)
```

```bash
!python3 train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --pretrained_vae_name_or_path="stabilityai/sd-vae-ft-mse" \
  --output_dir=$OUTPUT_DIR \
  --revision="fp16" \
  --with_prior_preservation --prior_loss_weight=1.0 \
  --seed=1337 \
  --resolution=512 \
  --train_batch_size=1 \
  --train_text_encoder \
  --mixed_precision="fp16" \
  --use_8bit_adam \
  --gradient_accumulation_steps=1 \
  --learning_rate=1e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_class_images=50 \
  --sample_batch_size=4 \
  --max_train_steps=800 \
  --save_interval=10000 \
  --save_sample_prompt="photo of zwx dog" \
  --concepts_list="concepts_list.json"

# Reduce the `--save_interval` to lower than `--max_train_steps` to save weights from intermediate steps.
# `--save_sample_prompt` can be same as `--instance_prompt` to generate intermediate samples (saved along with weights in samples dire
```


```python
#Specify the weights directory to use (leave blank for latest)
WEIGHTS_DIR = "" #@param {type:"string"}
if WEIGHTS_DIR == "":
    from natsort import natsorted
    from glob import glob
    import os
    WEIGHTS_DIR = natsorted(glob(OUTPUT_DIR + os.sep + "*"))[-1]
print(f"[*] WEIGHTS_DIR={WEIGHTS_DIR}")
```

```python
#Run to generate a grid of preview images from the last saved weights.
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

weights_folder = OUTPUT_DIR
folders = sorted([f for f in os.listdir(weights_folder) if f != "0"], key=lambda x: int(x))

row = len(folders)
col = len(os.listdir(os.path.join(weights_folder, folders[0], "samples")))
scale = 4
fig, axes = plt.subplots(row, col, figsize=(col*scale, row*scale), gridspec_kw={'hspace': 0, 'wspace': 0})

for i, folder in enumerate(folders):
    folder_path = os.path.join(weights_folder, folder)
    image_folder = os.path.join(folder_path, "samples")
    images = [f for f in os.listdir(image_folder)]
    for j, image in enumerate(images):
        if row == 1:
            currAxes = axes[j]
        else:
            currAxes = axes[i, j]
        if i == 0:
            currAxes.set_title(f"Image {j}")
        if j == 0:
            currAxes.text(-0.1, 0.5, folder, rotation=0, va='center', ha='center', transform=currAxes.transAxes)
        image_path = os.path.join(image_folder, image)
        img = mpimg.imread(image_path)
        currAxes.imshow(img, cmap='gray')
        currAxes.axis('off')
        
plt.tight_layout()
plt.savefig('grid.png', dpi=72)
```


**Convert weights to ckpt**

```
#Run conversion.
ckpt_path = WEIGHTS_DIR + "/model.ckpt"

half_arg = ""
#@markdown  Whether to convert to fp16, takes half the space (2GB).
fp16 = True #@param {type: "boolean"}
if fp16:
    half_arg = "--half"
!python convert_diffusers_to_original_stable_diffusion.py --model_path $WEIGHTS_DIR  --checkpoint_path $ckpt_path $half_arg
print(f"[*] Converted ckpt saved at {ckpt_path}")
```



**Inference Code**

```python
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline, DDIMScheduler
from IPython.display import display

model_path = WEIGHTS_DIR             # If you want to use previously trained model saved in gdrive, replace this with the full path of model in gdrive

pipe = StableDiffusionPipeline.from_pretrained(model_path, safety_checker=None, torch_dtype=torch.float16).to("cuda")
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe.enable_xformers_memory_efficient_attention()
g_cuda = None
```

```python
#Can set random seed here for reproducibility.
g_cuda = torch.Generator(device='cuda')
seed = 52362 #@param {type:"number"}
g_cuda.manual_seed(seed)
```

```
#Run for generating images.

prompt = "photo of zwx dog in a bucket" #@param {type:"string"}
negative_prompt = "" #@param {type:"string"}
num_samples = 4 #@param {type:"number"}
guidance_scale = 7.5 #@param {type:"number"}
num_inference_steps = 24 #@param {type:"number"}
height = 512 #@param {type:"number"}
width = 512 #@param {type:"number"}

with autocast("cuda"), torch.inference_mode():
    images = pipe(
        prompt,
        height=height,
        width=width,
        negative_prompt=negative_prompt,
        num_images_per_prompt=num_samples,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=g_cuda
    ).images

for img in images:
    display(img)
```


# Huggingface Models

For accessing Stable Diffusion model checkpoints, visit the Hugging Face Models repository: 

[Brain_normal_horizontal_MRI](https://huggingface.co/KidderLab/Brain_normal_horizontal_MRI)

[Brain_normal_saggital_MRI](https://huggingface.co/KidderLab/Brain_normal_saggital_MRI)

[Brain_pituitary_tumor_horizontal_MRI](https://huggingface.co/KidderLab/Brain_pituitary_tumor_horizontal_MRI)

[Brain_pituitary_tumor_saggital](https://huggingface.co/KidderLab/Brain_pituitary_tumor_saggital)

[Chest X-ray](https://huggingface.co/KidderLab/Chest_X-ray)

[Glioma_saggital_MRI](https://huggingface.co/KidderLab/Glioma_saggital_MRI)

[low_grade_glioma_lgg_horizontal](https://huggingface.co/KidderLab/low_grade_glioma_lgg_horizontal)

[Mammography_CESM](https://huggingface.co/KidderLab/Mammography_CESM)

[meningioma_tumor](https://huggingface.co/KidderLab/meningioma_tumor)



# Compute FID Scores 

You may access the PyTorch-FID code for computing FID scores at this link: [pytorch-fid](https://github.com/mseitzer/pytorch-fid/).

**Installation**

```bash
git clone https://github.com/mseitzer/pytorch-fid.git
pip install pytorch-fid
```

**Requirements:**

* python3
* pytorch
* torchvision
* pillow
* numpy
* scipy


**Usage**

```bash
python -m pytorch_fid path/to/dataset1 path/to/dataset2
```

# Citing

If you use this repository in your research, consider citing it using:

```
@article {Kidder2023.08.18.553859,
	author = {Kidder, Benjamin L.},
	title = {Advanced image generation for cancer using diffusion models},
	elocation-id = {2023.08.18.553859},
	year = {2023},
	doi = {10.1101/2023.08.18.553859},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2023/08/21/2023.08.18.553859},
	eprint = {https://www.biorxiv.org/content/early/2023/08/21/2023.08.18.553859.full.pdf},
	journal = {bioRxiv}
}
```


