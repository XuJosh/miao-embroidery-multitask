"""
命令行单张图片推理工具。
用法示例：
    python predict_cli.py --checkpoint checkpoints/checkpoint_best_pruned_amount0.3.pth.gz \
                          --image ../data/guizhou_embroidery_correct/images/test/test_00000_a0.jpg
"""

import argparse
import json

from model_loader import predict_from_path


def main():
    parser = argparse.ArgumentParser(description='苗绣多任务识别命令行工具')
    parser.add_argument('--checkpoint', type=str,
                        default='checkpoints/checkpoint_best_pruned_amount0.3.pth.gz',
                        help='模型 checkpoint 路径')
    parser.add_argument('--image', type=str, required=True, help='待识别图片路径')
    parser.add_argument('--device', type=str, default='auto', help='计算设备：auto / cuda / cpu')
    parser.add_argument('--input_size', type=int, default=224, help='输入尺寸')
    args = parser.parse_args()

    result = predict_from_path(args.checkpoint, args.image, args.device, args.input_size)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
