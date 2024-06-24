from diffusers import StableDiffusionPipeline
import torch
import clip

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PIPELINE = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)
# When you fine-tune using LORA, you can load the weights here
# PIPELINE.load_lora_weights("lora weights/unique_artist.safetensors", weight_name="pytorch_lora_weights.safetensors")
PIPELINE = PIPELINE.to(DEVICE)

def get_prompt(random):
    prompts = [
        f"Create an image of a working on a tour plan in a",
        f"Create an image of a brainstorming new ideas in a",
        f"Create an image of a actively working on a project in a",
        f"Create an image of a reflecting on their work in a",
        f"Create an image of a collaborating with colleagues in a",
        f"Create an image of a teaching or presenting in a",
        f"Create an image of a conducting research in a",
        f"Create an image of a creating an art piece in a",
        f"Create an image of a solving a complex problem in a",
        f"Create an image of a giving a speech or a lecture in a",
        f"Create an image of a experimenting with new techniques in a"
        f"Create an image of a designing a new invention in a",
        f"Create an image of a leading a team meeting in a",
        f"Create an image of a analyzing data on a computer in a",
        f"Create an image of a writing a book in a",
        f"Create an image of a gardening in a",
        f"Create an image of a playing a musical instrument in a",
        f"Create an image of a practicing yoga in a",
        f"Create an image of a cooking in a gourmet kitchen in a",
        f"Create an image of a building a robot in a",
        f"Create an image of a exploring a historic site in a"
    ]
    return prompts[random]

DESCRIPTIONS = ['unique', 'distinctive', 'cool']  
PROFESSIONS = ['scientist', 'artist', 'professor']  
SETTINGS = ['corporate office', 'research center', 'classroom'] 

def create_prompt(a,b,c, prompt):
    sentence = prompt[:21] + DESCRIPTIONS[a] + " " + PROFESSIONS[b] + " " + prompt[21:] + " " + SETTINGS[c] + "."
    return sentence

NUM_ACTIONS = 3
CLIP_MODEL, CLIP_PREPROCESS = clip.load("ViT-B/32", device=DEVICE)