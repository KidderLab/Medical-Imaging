# Medical_Imaging


**Dreambooth Stable Diffusion Notebook:**
[DreamBooth Stable Diffusion Notebook](https://github.com/ShivamShrirao/diffusers/blob/main/examples/dreambooth/DreamBooth_Stable_Diffusion.ipynb).
This jupyter notebook was used for training and synthesis of medical images.

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

**Gradio Web UI**

```python
import gradio as gr
def generate_images(prompt):
    return pipe(prompt=prompt, num_inference_steps=50).images
gr.Interface(fn=generate_images, inputs="text", outputs="image").launch()
```
