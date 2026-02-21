import torch
import pandas as pd
from torchvision.transforms import transforms
from PIL import Image

from mem3test import model


def load_and_preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize(1024),
        transforms.CenterCrop(1024),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img = Image.open(image_path).convert('RGB')
    return transform(img).unsqueeze(0).expand(1, -1, -1, -1)

def measure_layer_memory(target_model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cpu':
        print(" CUDA is not available.")
        return

    model = target_model(weights=None).to(device).eval()
    input_image = load_and_preprocess_image("../pic/Cat03.jpg").to(device)
    layer_stats = []

    def get_memory_usage(layer, x):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        initial_mem = torch.cuda.memory_allocated() / (1024 ** 2)

        with torch.no_grad():
            output = layer(x)

        peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)
        return output, peak_mem - initial_mem

    x = input_image

    x, mem = get_memory_usage(model.conv_proj, x)
    layer_stats.append({"Layer": "patch_embedding", "Peak_Memory_MB": mem})

    n, c, h, w = x.shape
    x = x.reshape(n, c, h * w).transpose(1, 2)
    batch_class_token = model.class_token.expand(n, -1, -1)
    x = torch.cat([batch_class_token, x], dim=1)

    pos_embed = model.encoder.pos_embedding
    if x.shape[1] != pos_embed.shape[1]:
        print(f"Interpolating position embedding from {pos_embed.shape[1]} to {x.shape[1]}")
        if x.shape[1] > pos_embed.shape[1]:
            padding = torch.zeros((1, x.shape[1] - pos_embed.shape[1], pos_embed.shape[2]), device=device)
            pos_embed = torch.cat([pos_embed, padding], dim=1)
        else:
            pos_embed = pos_embed[:, :x.shape[1], :]

    x = x + pos_embed
    layer_stats.append({"Layer": "pos_embedding_addition", "Peak_Memory_MB": 0.1})

    for i, block in enumerate(model.encoder.layers):
        x, mem = get_memory_usage(block, x)
        layer_stats.append({"Layer": f"transformer_block_{i}", "Peak_Memory_MB": mem})
        if i % 4 == 0:
            print(f"Measured Block {i}/31: {mem:.2f} MB")

    x, mem = get_memory_usage(model.encoder.ln, x)
    layer_stats.append({"Layer": "final_layer_norm", "Peak_Memory_MB": mem})

    x = x[:, 0]
    x, mem = get_memory_usage(model.heads, x)
    layer_stats.append({"Layer": "heads_fc", "Peak_Memory_MB": mem})

    df = pd.DataFrame(layer_stats)
    df.to_csv("xxx_1024.csv", index=False)


if __name__ == "__main__":
    model=""
    measure_layer_memory(model)