import os
import timm

def download_timm_pretrained(model_name):
    cache_dir = os.path.expanduser('~/.cache/torch/hub/checkpoints')
    os.makedirs(cache_dir, exist_ok=True)
    print(f'正在下载 {model_name} 的预训练权重...')
    # pretrained=True会自动下载权重到默认目录
    timm.create_model(model_name, pretrained=True)
    print(f'{model_name} 权重下载完成！')

if __name__ == '__main__':
    model_list = [
        #'tf_efficientnet_b3',
        #'resnet50',
        #'convnext_base',
        'swin_base_patch4_window12_384'
    ]
    for model_name in model_list:
        download_timm_pretrained(model_name)
    print('所有模型权重下载完成。')