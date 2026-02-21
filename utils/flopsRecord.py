import torch
import pandas as pd

def measure_layer_flops(target_model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = target_model(weights=None).to(device).eval()

    # 分辨率
    input_res = 1024
    input_tensor = torch.randn(1, 3, input_res, input_res).to(device)

    layer_stats = []

    # Patch Embedding
    flops_conv = FlopCountAnalysis(model.conv_proj, input_tensor).total()
    layer_stats.append({"Layer": "patch_embedding", "FLOPs_G": flops_conv / 1e9})

    with torch.no_grad():
        x = model.conv_proj(input_tensor)
        n, c, h, w = x.shape
        x = x.reshape(n, c, h * w).transpose(1, 2)
        batch_class_token = model.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        pos_embed = model.encoder.pos_embedding
        if x.shape[1] != pos_embed.shape[1]:
            if x.shape[1] > pos_embed.shape[1]:
                padding = torch.zeros((1, x.shape[1] - pos_embed.shape[1], pos_embed.shape[2]), device=device)
                pos_embed = torch.cat([pos_embed, padding], dim=1)
            else:
                pos_embed = pos_embed[:, :x.shape[1], :]

        x = x + pos_embed
    # Blocks
    for i, block in enumerate(model.encoder.layers):
        block_flops = FlopCountAnalysis(block, x).total()
        layer_stats.append({"Layer": f"transformer_block_{i}", "FLOPs_G": block_flops / 1e9})

        with torch.no_grad():
            x = block(x)

        if i % 4 == 0:
            print(f"Layer {i}/31 finished. Current FLOPs: {block_flops / 1e9:.2f} G")

    # Final Layers
    with torch.no_grad():
        x = model.encoder.ln(x)
        x = x[:, 0]
    flops_head = FlopCountAnalysis(model.heads, x).total()
    layer_stats.append({"Layer": "heads_fc", "FLOPs_G": flops_head / 1e9})

    df = pd.DataFrame(layer_stats)
    df.to_csv("xxx_1024.csv", index=False)



if __name__ == "__main__":
    target_model=""
    measure_layer_flops(target_model)