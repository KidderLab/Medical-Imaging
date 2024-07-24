# Medical_Imaging


**Dreambooth Stable Diffusion Notebook:**
[Notebook](https://github.com/ShivamShrirao/diffusers/blob/main/examples/dreambooth/DreamBooth_Stable_Diffusion.ipynb).
This jupyter notebook can be run in [colab](https://colab.research.google.com/) to perform training and synthesis of medical images.

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

```
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

```
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

**Inference Code**

```python
from diffusers import StableDiffusionPipeline
pipe = StableDiffusionPipeline.from_pretrained("./path_to_trained_model", safety_checker=None, torch_dtype=torch.float16).to("cuda")
prompt = "your_prompt_here"
with torch.autocast("cuda"), torch.inference_mode():
    images = pipe(prompt=prompt, num_inference_steps=50).images
for img in images:
    img.show()
```

# Huggingface Models

[Chest X-ray](https://huggingface.co/KidderLab/Chest_X-ray)

[Brain_normal_saggital_MRI](https://huggingface.co/KidderLab/Brain_normal_saggital_MRI)

[Brain_pituitary_tumor_cross_sectio](https://huggingface.co/KidderLab/Brain_pituitary_tumor_cross_section)

[Brain_pituitary_tumor_horizontal_MRI](https://huggingface.co/KidderLab/Brain_pituitary_tumor_horizontal_MRI)

[Glioma_saggital_MRI](https://huggingface.co/KidderLab/Glioma_saggital_MRI)

[Mammography_CESM](https://huggingface.co/KidderLab/Mammography_CESM)

[meningioma_tumor](https://huggingface.co/KidderLab/meningioma_tumor)

[Brain_pituitary_tumor_saggital](https://huggingface.co/KidderLab/Brain_pituitary_tumor_saggital)

[low_grade_glioma_lgg_horizontal](https://huggingface.co/KidderLab/low_grade_glioma_lgg_horizontal)




