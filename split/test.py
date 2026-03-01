import os
import torch
import torch.nn as nn

from PIL import Image
import torchvision.transforms as transforms
import torch.nn.functional as F
from thop import profile



def load_and_preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    img = Image.open(image_path).convert('RGB')
    return transform(img).unsqueeze(0).expand(16, -1, -1, -1)


class ViTSegment1(nn.Module):
    def __init__(self, original_model, end_layer_idx):
        super(ViTSegment1, self).__init__()
        self.patch_embedding = original_model.conv_proj
        self.class_token = original_model.class_token
        self.positional_embedding = original_model.encoder.pos_embedding
        self.encoder_layers = nn.Sequential(*list(original_model.encoder.layers.children())[:end_layer_idx])

    def forward(self, x):
        x = self.patch_embedding(x)
        n, c, h, w = x.shape
        x = x.permute(0, 2, 3, 1).reshape(n, h * w, c)
        class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat((class_token, x), dim=1)
        x = x + self.positional_embedding
        x = self.encoder_layers(x)
        return x



def calculate_flops_and_params(model, input_shape, device='cuda'):
    model = model.to(device)
    model.eval()

    if len(input_shape) == 4:
        dummy_input = torch.randn(input_shape).to(device)
    else:
        dummy_input = torch.randn(input_shape).to(device)

    with torch.no_grad():
        flops, params = profile(model, inputs=(dummy_input,), verbose=False)

    return flops, params


def split_vit(model_path=None, save_dir="../pth/vit_b16/", pretrained=False):
    weights = 'DEFAULT' if pretrained else None
    model = ""

    if model_path:
        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict)

    model.eval()
    os.makedirs(save_dir, exist_ok=True)


    segments = {

    }

    flops1, params1 = calculate_flops_and_params(
        segments['segment1'],
        input_shape=(1, 3, 224, 224)
    )


    flops2, params2 = calculate_flops_and_params(
        segments['segment2'],
        input_shape=(1, 197, 768)
    )


    flops3, params3 = calculate_flops_and_params(
        segments['segment3'],
        input_shape=(1, 197, 768)
    )


    total_params = params1 + params2 + params3
    total_flops = flops1 + flops2 + flops3


    return segments


def distributed_inference(segments, input_image, warmup=1, iterations=1):
    device = 'cuda:0'
    seg1 = segments['segment1'].eval().to(device)
    seg2 = segments['segment2'].eval().to(device)
    seg3 = segments['segment3'].eval().to(device)


    with torch.no_grad():

        flops1, _ = profile(seg1, inputs=(input_image.to(device),), verbose=False)

        intermediate1 = seg1(input_image.to(device))
        flops2, _ = profile(seg2, inputs=(intermediate1,), verbose=False)

        intermediate2 = seg2(intermediate1)
        flops3, _ = profile(seg3, inputs=(intermediate2,), verbose=False)


    timings = {'seg1': [], 'seg2': [], 'seg3': []}
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)

    with torch.no_grad():

        for _ in range(warmup):
            _ = seg1(input_image.to(device))
            _ = seg2(intermediate1)
            _ = seg3(intermediate2)
        torch.cuda.synchronize()

        for _ in range(iterations):
            starter.record()
            intermediate1 = seg1(input_image.to(device))
            ender.record()
            torch.cuda.synchronize()
            timings['seg1'].append(starter.elapsed_time(ender))

            starter.record()
            intermediate2 = seg2(intermediate1)
            ender.record()
            torch.cuda.synchronize()
            timings['seg2'].append(starter.elapsed_time(ender))

            starter.record()
            output = seg3(intermediate2)
            ender.record()
            torch.cuda.synchronize()
            timings['seg3'].append(starter.elapsed_time(ender))

    return output, timings


def main():
    segments = split_vit(pretrained=True)
    image_path = '../pic/Cat03.jpg'

    try:
        input_image = load_and_preprocess_image(image_path)
        print(f"成功加载图片: {image_path}")
    except Exception as e:
        print(f"图片加载失败: {e}")


    output, timings = distributed_inference(segments, input_image)
    averages = {key: sum(value) / len(value) for key, value in timings.items()}
    for seg, time in averages.items():
        print(f"{seg}: {time:.2f} ms")

    classes = get_imagenet_classes()
    prob = F.softmax(output, dim=1)
    top5_prob, top5_idx = prob.topk(5)

    for i in range(5):
        print(f"{classes[top5_idx[0][i]]:>20s}: {top5_prob[0][i]:.3f}")


