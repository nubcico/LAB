from IPython.display import Image
import torch
import clip
from main import load_vqgan_model, CLIP_DIM, clamp_with_grad, synth
import torchvision  

model_path = "model.th"
device = "cuda" if torch.cuda.is_available() else "cpu"
net = torch.load(model_path, map_location="cpu").to(device)
config = net.config
vqgan_config = config.vqgan_config 
vqgan_checkpoint = config.vqgan_checkpoint
clip_model = config.clip_model
clip_dim = CLIP_DIM
perceptor = clip.load(clip_model, jit=False)[0].eval().requires_grad_(False).to(device)
model = load_vqgan_model(vqgan_config, vqgan_checkpoint).to(device)
z_min = model.quantize.embedding.weight.min(dim=0).values[None, :, None, None]
z_max = model.quantize.embedding.weight.max(dim=0).values[None, :, None, None]

texts = [
  "Little Red Riding Hood with her bright red hood and innocent face wanders through the dark and eerie woods unaware of the danger lurking nearby",
  "The wolf with his sinister grin and cunning eyes stands outside the grandmother's house ready to deceive Little Red Riding Hood and carry out his evil plan",
  "The image of Little Red Riding Hood standing at her grandmother's door, unknowingly facing the wolf disguised as her beloved grandmother, sends chills down the spine as the inevitable danger looms near"
]
print("|".join(texts))
toks = clip.tokenize(texts, truncate=True)
H = perceptor.encode_text(toks.to(device)).float()
with torch.no_grad():
    z = net(H)
    z = clamp_with_grad(z, z_min.min(), z_max.max())
    xr = synth(model, z)
grid = torchvision.utils.make_grid(xr.cpu(), nrow=len(xr))
out_path = "gen.png"
torchvision.transforms.functional.to_pil_image(grid).save(out_path)
sz = 256
Image("gen.png", width=sz*len(texts), height=sz)