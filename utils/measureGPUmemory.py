import torch
from thop import profile
from torchvision import transforms
from PIL import Image


def measure_segment(model, input_tensor, name):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    device = next(model.parameters()).device
    input_on_device = input_tensor.to(device)

    input_bytes = input_on_device.element_size() * input_on_device.nelement()
    print(f"[{name}] 输入张量大小: {input_bytes / 1024 ** 2:.2f} MB")

    model.eval()
    with torch.no_grad():
        output = model(input_on_device)

    peak_memory = torch.cuda.max_memory_allocated() / 1024 ** 2
    output_bytes = output.element_size() * output.nelement()

    print(f"[{name}] 显存峰值: {peak_memory:.2f} MB")
    print(f"[{name}] 输出张量大小: {output_bytes / 1024 ** 2:.2f} MB")

    output_cpu = output.cpu()
    del output
    torch.cuda.empty_cache()
    return output_cpu


if __name__ == '__main__':
    target_model=""

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA不可用")

    device = torch.device('cuda')
    torch.cuda.empty_cache()


    weight_paths = [
        "","",...,""
    ]

    image_path = "../pic/Cat03.jpg"
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    input_tensor = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
    base_model = target_model(weights=False)

    segment_config = [
        "", "", ..., ""
    ]
    current_output = input_tensor
    for idx, (segment_class, segment_name, kwargs) in enumerate(segment_config):
        segment = segment_class(base_model, **kwargs).to(device)
        segment.eval()

        state_dict = torch.load(weight_paths[idx], map_location=device)
        segment.load_state_dict(state_dict)

        current_output = measure_segment(segment, current_output, segment_name)

        num_params = sum(p.numel() for p in segment.parameters())
        with torch.no_grad():
            flops, _ = profile(segment, inputs=(current_output,), verbose=False)
        print(f"Model Parameters: {num_params / 1e6:.2f} M")
        print(f"Model FLOPs: {flops / 1e9:.2f} G")
        del segment
        torch.cuda.empty_cache()

    probabilities = torch.nn.functional.softmax(current_output, dim=1)
    top_prob, top_class = probabilities.topk(1, dim=1)
    # print(f"\n预测结果: 类别 {top_class.item()} (置信度: {top_prob.item():.2%})")