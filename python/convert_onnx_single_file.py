#!/usr/bin/env python3
"""
Convert ONNX model with external data to single-file format
"""

import onnx
import sys

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage: python convert_onnx_single_file.py input.onnx output.onnx')
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    print(f'Loading model from: {input_path}')
    model = onnx.load(input_path, load_external_data=True)

    print(f'Saving single-file model to: {output_path}')
    onnx.save(model, output_path)

    print('Done!')
    print(f'File size: {onnx.shape_inference.infer_shapes(model).__sizeof__()} bytes')
