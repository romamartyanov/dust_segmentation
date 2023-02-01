import os
from pathlib import Path

import torch
from config import CFG


class ModelOptimization:
    def __init__(self, pytorch_model, pytorch_path) -> None:
        self.pytorch_model = pytorch_model
        self.pytorch_path = pytorch_path

        self.dirname = os.path.dirname(pytorch_path)
        self.model_name = os.path.splitext(os.path.basename(pytorch_path))[0]
        self.model_path = os.path.join(self.dirname, self.model_name)

        self.onnx_path = Path(self.model_path).with_suffix(".onnx").as_posix()
        self.ir_path = Path(self.model_path).with_suffix(".xml").as_posix()

        self.c, self.h, self.w = 3, CFG.img_size[0], CFG.img_size[1]

        self.piece_size = CFG.img_size[0]
        self.num_pieces_hor = CFG.big_img_size[0] // self.piece_size
        self.num_pieces_ver = CFG.big_img_size[0] // self.piece_size

        self.bs = self.num_pieces_hor

    def pytorch2onnx(self):
        self.pytorch_model.eval()
        x = torch.randn(self.bs, self.c, self.h, self.w, requires_grad=True)
        _ = self.pytorch_model(x)

        torch.onnx.export(self.pytorch_model,           # model being run
                          x,                            # model input (or a tuple for multiple inputs)
                          self.onnx_path,               # where to save the model (can be a file or file-like object)
                          export_params=True,           # store the trained parameter weights inside the model file
                          opset_version=11,             # the ONNX version to export the model to
                          do_constant_folding=True,     # whether to execute constant folding for optimization
                          input_names=['input'],        # the model's input names
                          output_names=['output'],      # the model's output names
                          dynamic_axes={
                              # variable length axes
                              'input': {0: 'batch_size'},
                              'output': {0: 'batch_size'}
                          })
        return self.onnx_path

    def onnx2openvino(self, mode="FP16"):
        if not os.path.exists(self.onnx_path):
            self.pytorch2onnx()

        # Construct the command for Model Optimizer.
        mo_command = f"""mo
                        --input_model "{self.onnx_path}"
                        --input_shape "[{self.bs},{self.c},{self.h},{self.w}]"
                        --data_type {mode}
                        --output_dir "{self.dirname}"
                        """
        mo_command = " ".join(mo_command.split())
        print("Model Optimizer command to convert the ONNX model to OpenVINO:")
        print(f"`{mo_command}`")

        print("Exporting ONNX model to IR... This may take a few minutes.")
        os.system(mo_command)
        return self.ir_path

    def openvino_fp16_to_int8(self):
        raise NotImplementedError()
