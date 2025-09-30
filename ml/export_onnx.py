"""
Export PyTorch model to ONNX format for TypeScript/JavaScript inference

Usage (in Colab):
    python export_onnx.py --model /path/to/best_model.pt --output /path/to/best_model.onnx
"""

import argparse
import torch
import torch.onnx
from model import create_model

def export_to_onnx(
    model_path: str,
    output_path: str,
    model_type: str = "resnet",
    num_channels: int = 128,
    num_res_blocks: int = 6,
    hidden_size: int = 512
):
    """
    Export PyTorch model to ONNX format

    Args:
        model_path: Path to .pt checkpoint file
        output_path: Path to save .onnx file
        model_type: "simple" or "resnet"
        num_channels: Number of channels (for resnet)
        num_res_blocks: Number of residual blocks (for resnet)
        hidden_size: Hidden size (for simple model)
    """
    print(f"Loading model from {model_path}...")

    # Create model architecture
    if model_type == "simple":
        model = create_model("simple", hidden_size=hidden_size)
    else:
        model = create_model("resnet", num_channels=num_channels, num_res_blocks=num_res_blocks)

    # Load weights
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print(f"Model loaded on {device}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create dummy input (batch_size=1, channels=6, squares=32)
    dummy_input = torch.randn(1, 6, 32).to(device)

    # Export to ONNX
    print(f"\nExporting to ONNX...")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=['state'],
        output_names=['policy', 'value'],
        dynamic_axes={
            'state': {0: 'batch_size'},
            'policy': {0: 'batch_size'},
            'value': {0: 'batch_size'}
        },
        opset_version=14,
        do_constant_folding=True,
        verbose=False
    )

    print(f"✓ ONNX model saved to {output_path}")

    # Verify the exported model
    print("\nVerifying ONNX model...")
    import onnx
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("✓ ONNX model is valid!")

    # Test inference
    print("\nTesting ONNX inference...")
    import onnxruntime as ort

    ort_session = ort.InferenceSession(output_path)

    # Run inference
    test_input = dummy_input.cpu().numpy()
    ort_inputs = {ort_session.get_inputs()[0].name: test_input}
    ort_outputs = ort_session.run(None, ort_inputs)

    policy_shape = ort_outputs[0].shape
    value_shape = ort_outputs[1].shape

    print(f"  Policy output shape: {policy_shape}")
    print(f"  Value output shape: {value_shape}")
    print("✓ ONNX inference successful!")

    print(f"\n{'='*60}")
    print("EXPORT COMPLETE!")
    print(f"{'='*60}")
    print(f"ONNX model: {output_path}")
    print(f"\nNext steps:")
    print(f"1. Download {output_path}")
    print(f"2. Place in project: ml/best_model.onnx")
    print(f"3. Install: npm install onnxruntime-node")
    print(f"4. Use TypeScript inference wrapper")

def main():
    parser = argparse.ArgumentParser(description="Export PyTorch model to ONNX")

    parser.add_argument("--model", type=str, required=True, help="Path to .pt model file")
    parser.add_argument("--output", type=str, required=True, help="Output path for .onnx file")
    parser.add_argument("--model_type", type=str, default="resnet", choices=["simple", "resnet"])
    parser.add_argument("--num_channels", type=int, default=128, help="Number of channels (resnet)")
    parser.add_argument("--num_res_blocks", type=int, default=6, help="Number of residual blocks (resnet)")
    parser.add_argument("--hidden_size", type=int, default=512, help="Hidden size (simple)")

    args = parser.parse_args()

    export_to_onnx(
        model_path=args.model,
        output_path=args.output,
        model_type=args.model_type,
        num_channels=args.num_channels,
        num_res_blocks=args.num_res_blocks,
        hidden_size=args.hidden_size
    )

if __name__ == "__main__":
    main()