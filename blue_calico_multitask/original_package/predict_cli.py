"""
原网络命令行单张图片推理工具。
逻辑：先鉴别真伪；仅当真品时才输出纹样和疵点结果。
"""

import argparse
import json

from model_loader import predict_from_path


def format_result(result):
    auth = result['auth']
    lines = [
        f"真伪：{auth['name']}（置信度 {auth['prob']:.2%}）",
        f"  真品 {auth['probs']['真品 / 手工']:.2%} | 伪作 {auth['probs']['伪作 / 机绣']:.2%}",
    ]
    if auth['label'] == 1:
        pat = result['pattern']
        defect = result['defect']
        lines += [
            '',
            f"纹样：{pat['name']}（置信度 {pat['prob']:.2%}）",
            f"  辫绣 {pat['probs']['辫绣']:.2%} | 堆绣 {pat['probs']['堆绣']:.2%} | "
            f"马尾绣 {pat['probs']['马尾绣']:.2%} | 其他 {pat['probs']['其他']:.2%} | "
            f"数纱马尾绣 {pat['probs']['数纱马尾绣']:.2%}",
            '',
            f"疵点：{defect['name']}（置信度 {defect['prob']:.2%}）",
            f"  无疵点 {defect['probs']['无疵点']:.2%} | 有疵点 {defect['probs']['有疵点']:.2%}",
        ]
    else:
        lines += ['', '伪品：跳过纹样分类与疵点检测。']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description='苗绣多任务识别命令行工具（原网络版）')
    parser.add_argument('--checkpoint', type=str,
                        default='checkpoints/embroidery_cls_best.pth',
                        help='原网络 checkpoint 路径')
    parser.add_argument('--image', type=str, required=True, help='待识别图片路径')
    parser.add_argument('--device', type=str, default='auto', help='计算设备：auto / cuda / cpu')
    parser.add_argument('--input_size', type=int, default=224, help='输入尺寸')
    parser.add_argument('--json', action='store_true', help='以 JSON 格式输出完整结果')
    args = parser.parse_args()

    result = predict_from_path(args.checkpoint, args.image, args.device, args.input_size)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(format_result(result))


if __name__ == '__main__':
    main()
