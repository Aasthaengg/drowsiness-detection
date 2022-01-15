import os
import onnx
import torch
import tensorflow as tf
from onnx_tf.backend import prepare

if __name__ == '__main__':
    PYTORCH_MODEL = "./models/eyes_resnet18_128x128.pt"
    ONNX_PATH = PYTORCH_MODEL.replace(".pt", ".onnx")
    TF_DIR = './models/ONNX_TO_TF'
    TF_PATH = PYTORCH_MODEL.replace(".pt", ".pb")
    TFLITE_PATH = PYTORCH_MODEL.replace(".pt", ".tflite")
    
    # Set device and load pytorch model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch_model = torch.load(PYTORCH_MODEL, map_location=device)
    torch_model.eval()

    # Dummy input for ONNX export
    x = torch.randn(1, 3, 128, 128, requires_grad=True).to(device)
    torch_out = torch_model(x)
    
    # Export ONNX model
    torch.onnx.export(
        torch_model,
        x,                         
        ONNX_PATH,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names = ['input'],
        output_names = ['output']
    )

    print('Converted to onnx..')

    # Validate onnx model
    onnx_model = onnx.load(ONNX_PATH)
    onnx.checker.check_model(onnx_model)

    print('Checked onnx model..')

    # Convert onnx model to tensorflow
    tf_rep = prepare(onnx_model)
    tf_rep.export_graph(TF_DIR)

    print('Converted to tf..')

    # Convert tf model to tflite
    converter = tf.lite.TFLiteConverter.from_saved_model(TF_DIR)

    converter.experimental_new_converter = True
    converter.optimizations = [tf.compat.v1.lite.Optimize.DEFAULT]

    converter.target_spec.supported_ops = [tf.compat.v1.lite.OpsSet.TFLITE_BUILTINS,
                                       tf.compat.v1.lite.OpsSet.SELECT_TF_OPS]

    tflite_model = converter.convert()

    # Save the model
    with open(TFLITE_PATH, 'wb') as f:
        f.write(tflite_model)
    f.close()

    print('Converted to tflite..')