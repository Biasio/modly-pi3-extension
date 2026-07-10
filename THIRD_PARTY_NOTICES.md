# Third-party notices

This extension vendors Pi3 source code under `pi3_vendor/pi3`.

## Pi3 upstream

- Project: Pi3
- Source: https://github.com/yyfz/Pi3
- Vendored commit: `9fa3ddb3f8d53041f8b2738df404f62223bbaa7b`
- Top-level upstream license: BSD-3-Clause; see `pi3_vendor/PI3_LICENSE`.

## Additional file-level notices

The vendored upstream tree includes files with their own notices, including:

- Apache-2.0 notices from Meta/DINO-related source files.
- CC BY-NC-SA 4.0 notices in Naver-derived positional embedding utilities.

Keep those file headers intact when modifying vendored code.

## Model weights

- Hugging Face repo: https://huggingface.co/yyfz233/Pi3
- Required file: `model.safetensors`
- Hugging Face model-card metadata says BSD-2-Clause. Its prose describes academic use and asks users to contact the authors for commercial use. No standalone license file for the weights was found. Review the current upstream model-card terms before use; this notice does not replace them.

The extension does not redistribute weights and does not download them in `setup.py` or `generator.py`; Modly's model-download UI manages the runtime copy under `models/pi3/generate/model.safetensors`.

## Python dependencies

Runtime dependencies are declared in `requirements.txt` and include PyTorch, torchvision, NumPy, Pillow, OpenCV, plyfile, huggingface_hub, and safetensors. Each dependency remains under its own license.
