# Third-party notices

This extension vendors Pi3/Pi3X source code under `pi3_vendor/pi3`.

## Pi3 upstream code

- Project: Pi3
- Source: https://github.com/yyfz/Pi3
- Vendored commit: `9fa3ddb3f8d53041f8b2738df404f62223bbaa7b`
- Top-level upstream license: BSD-3-Clause; see `pi3_vendor/PI3_LICENSE`.

The vendored tree retains file-level Apache-2.0 notices from Meta/DINO-derived files and CC-BY-NC-SA-4.0 notices in Naver-derived positional embedding utilities. Keep those headers intact.

## Pi3X upstream code

- Upstream code: https://github.com/yyfz/Pi3
- Code license: BSD-3-Clause.

## Pi3 and Pi3X model weights

- Pi3 Hugging Face repository: https://huggingface.co/yyfz233/Pi3
- Pi3 runtime path: `models/pi3/generate/model.safetensors`
- Pi3X Hugging Face repository: https://huggingface.co/yyfz233/Pi3X
- Pi3X runtime path: `models/pi3/pi3x/model.safetensors`
- Required checkpoint filename: `model.safetensors`
- Weight license for both models: CC-BY-NC-4.0; strictly noncommercial.

The current upstream `yyfz/Pi3` README explicitly applies these weight terms to both Pi3 and Pi3X. Older Pi3 Hugging Face metadata may still report BSD-2, but this extension follows the current upstream repository's explicit weight terms. The checkpoints are not redistributed, and no standalone weight-license file is fabricated or bundled; Modly's model-download UI manages each runtime copy independently.

## Python dependencies

Runtime dependencies are declared in `requirements.txt` and include PyTorch, torchvision, NumPy, Pillow, OpenCV, plyfile, huggingface_hub, and safetensors. Each dependency remains under its own license.

The extension does not download either checkpoint from `setup.py` or `generator.py`. The canonical extension-wrapper [LICENSE](./LICENSE) remains MIT and is separate from the BSD-3-Clause upstream source-code terms and CC-BY-NC-4.0 weight terms above.
