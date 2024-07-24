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

```bash
!python3 train_dreambooth.py --pretrained_model_name_or_path=$MODEL_NAME --output_dir=$OUTPUT_DIR --train_batch_size=1 --max_train_steps=800 --learning_rate=1e-6
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




