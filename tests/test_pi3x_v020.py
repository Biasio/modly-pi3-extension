import io
import json
import os
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

import generator
import setup

try:
    import numpy as np
    from PIL import Image
except ImportError:  # pragma: no cover
    np = None
    Image = None


ROOT = Path(__file__).resolve().parents[1]


class FlashAttentionOptionalFallbackTests(unittest.TestCase):
    @staticmethod
    def _probe(*, missing=(), flash_status="missing", flash_error=None):
        imports = {
            package: package not in set(missing)
            for package, _module in setup.DEPENDENCY_IMPORTS
        }
        flash = {"status": flash_status}
        if flash_error:
            flash["error"] = flash_error
        return {
            "python": "/fake/venv/python",
            "imports": imports,
            "torch": {
                "version": "2.7.0+cu128",
                "cuda_version": "12.8",
                "cuda_available": True,
            },
            "torchvision": {"version": "0.22.0+cu128"},
            "flash_attn": flash,
        }

    @staticmethod
    def _pip_result(ok, tail):
        return {
            "command": ["/fake/venv/python", "-m", "pip"],
            "returncode": 0 if ok else 1,
            "ok": ok,
            "stdout_tail": tail,
            "env": {},
        }

    @staticmethod
    def _payload(ext, **extra):
        payload = {
            "python_exe": "/provided/python",
            "ext_dir": str(ext),
            "gpu_sm": 121,
            "cuda_version": 128,
            "allow_flash_attn_source_build": True,
        }
        payload.update(extra)
        return json.dumps(payload)

    def test_normal_setup_source_build_failure_uses_sdpa_and_preserves_diagnostics(self):
        with tempfile.TemporaryDirectory() as temp:
            ext = Path(temp) / "pi3"
            probe = self._probe()
            stderr = io.StringIO()
            pip_results = [
                self._pip_result(False, "no compatible binary wheel"),
                self._pip_result(True, "build tools ready"),
                self._pip_result(False, "nvcc compilation failed"),
            ]
            with mock.patch.object(setup, "ensure_extension_venv", return_value=ext / "venv/bin/python"), mock.patch.object(
                setup, "run_python_probe", return_value=probe
            ), mock.patch.object(setup, "pip_command", side_effect=pip_results), mock.patch.object(
                setup,
                "run_cuda_smoke_probe",
                return_value={"cuda_available": True, "smoke_ok": True, "smoke_error": None},
            ), mock.patch.object(setup.sys, "stderr", stderr):
                returncode = setup.main([self._payload(ext)])

            payload = json.loads((ext / setup.STATUS_RELATIVE_PATH).read_text())
            self.assertEqual(returncode, 0)
            self.assertEqual(payload["status"], "needs_weights")
            self.assertTrue(payload["dependencies_ok"])
            self.assertEqual(payload["flash_attn_status"], "fallback-sdpa")
            self.assertEqual(payload["flash_attn_code"], "flash-attn-source-build-failed")
            self.assertEqual(payload["flash_attn_results"][-1]["result"]["stdout_tail"], "nvcc compilation failed")
            self.assertEqual(payload["details"]["flash_attn"]["fallback_backend"], "pytorch-sdpa")
            warning = stderr.getvalue()
            self.assertIn("OPTIONAL acceleration failure", warning)
            self.assertIn("continuing with PyTorch SDPA", warning)
            self.assertIn("slower", warning)
            self.assertIn("more VRAM", warning)

    def test_post_install_import_failure_is_nonfatal_and_keeps_error(self):
        with tempfile.TemporaryDirectory() as temp:
            ext = Path(temp) / "pi3"
            before = self._probe()
            after = self._probe(
                flash_status="error",
                flash_error="ImportError: undefined symbol: flash_attn_cuda",
            )
            with mock.patch.object(setup, "ensure_extension_venv", return_value=ext / "venv/bin/python"), mock.patch.object(
                setup, "run_python_probe", side_effect=[before, before, after]
            ), mock.patch.object(
                setup, "pip_command", return_value=self._pip_result(True, "binary wheel installed")
            ), mock.patch.object(
                setup,
                "run_cuda_smoke_probe",
                return_value={"cuda_available": True, "smoke_ok": True, "smoke_error": None},
            ), mock.patch.object(setup.sys, "stderr", io.StringIO()):
                returncode = setup.main([self._payload(ext)])

            payload = json.loads((ext / setup.STATUS_RELATIVE_PATH).read_text())
            self.assertEqual(returncode, 0)
            self.assertEqual(payload["flash_attn_status"], "fallback-sdpa")
            self.assertEqual(payload["flash_attn_code"], "flash-attn-post-install-import-failed")
            self.assertIn("undefined symbol", payload["flash_attn_error"])
            self.assertEqual(payload["details"]["flash_attn"]["status_before_fallback"], "installed")

    def test_missing_required_dependency_remains_fatal(self):
        with tempfile.TemporaryDirectory() as temp:
            ext = Path(temp) / "pi3"
            probe = self._probe(missing={"numpy"}, flash_status="ok")
            smoke = mock.Mock()
            with mock.patch.object(setup, "ensure_extension_venv", return_value=ext / "venv/bin/python"), mock.patch.object(
                setup, "run_python_probe", return_value=probe
            ), mock.patch.object(setup, "install_base_requirements"), mock.patch.object(
                setup, "pip_command", side_effect=AssertionError("unexpected pip call")
            ), mock.patch.object(setup, "run_cuda_smoke_probe", smoke), mock.patch.object(
                setup.sys, "stderr", io.StringIO()
            ):
                returncode = setup.main([self._payload(ext)])

            payload = json.loads((ext / setup.STATUS_RELATIVE_PATH).read_text())
            self.assertEqual(returncode, 1)
            self.assertEqual(payload["status"], "needs_dependencies")
            self.assertFalse(payload["dependencies_ok"])
            smoke.assert_not_called()

    def test_failed_cuda_smoke_remains_fatal(self):
        with tempfile.TemporaryDirectory() as temp:
            ext = Path(temp) / "pi3"
            probe = self._probe(flash_status="ok")
            with mock.patch.object(setup, "ensure_extension_venv", return_value=ext / "venv/bin/python"), mock.patch.object(
                setup, "run_python_probe", return_value=probe
            ), mock.patch.object(
                setup, "pip_command", side_effect=AssertionError("unexpected pip call")
            ), mock.patch.object(
                setup,
                "run_cuda_smoke_probe",
                return_value={"cuda_available": True, "smoke_ok": False, "smoke_error": "CUDA launch failed"},
            ), mock.patch.object(setup.sys, "stderr", io.StringIO()):
                returncode = setup.main([self._payload(ext)])

            payload = json.loads((ext / setup.STATUS_RELATIVE_PATH).read_text())
            self.assertEqual(returncode, 1)
            self.assertEqual(payload["status"], "needs_dependencies")
            self.assertFalse(payload["dependencies_ok"])
            self.assertEqual(payload["torch_cuda_smoke_error"], "CUDA launch failed")

    def test_explicit_flash_attention_wheel_build_failure_remains_fatal(self):
        with tempfile.TemporaryDirectory() as temp:
            ext = Path(temp) / "pi3"
            probe = self._probe()
            with mock.patch.object(setup, "ensure_extension_venv", return_value=ext / "venv/bin/python"), mock.patch.object(
                setup, "run_python_probe", return_value=probe
            ), mock.patch.object(
                setup, "pip_command", return_value=self._pip_result(False, "build tools failed")
            ), mock.patch.object(setup.sys, "stderr", io.StringIO()):
                returncode = setup.main(
                    ["--build-flash-attn-wheel", self._payload(ext)]
                )

            payload = json.loads((ext / setup.STATUS_RELATIVE_PATH).read_text())
            self.assertEqual(returncode, 1)
            self.assertEqual(payload["status"], "failed")
            self.assertEqual(payload["flash_attn_status"], "failed")
            self.assertEqual(payload["flash_attn_code"], "flash-attn-build-tools-failed")


class Pi3RuntimeParamsSchemaTests(unittest.TestCase):
    @staticmethod
    def _schema_by_id():
        return {item["id"]: item for item in generator.Pi3Generator.params_schema()}

    def test_pi3x_model_id_selects_pi3x_runtime_schema(self):
        with mock.patch.dict(
            os.environ,
            {"MODEL_ID": "pi3/pi3x", "MODEL_DIR": "/models/pi3/generate"},
            clear=True,
        ):
            schema = self._schema_by_id()

        self.assertTrue(
            {"left_image_path", "back_image_path", "right_image_path"}.isdisjoint(schema)
        )
        self.assertEqual(schema["output_name"]["default"], "pi3x_point_cloud")

    def test_generate_model_id_selects_pi3_runtime_schema(self):
        with mock.patch.dict(
            os.environ,
            {"MODEL_ID": "pi3/generate", "MODEL_DIR": "/models/pi3/pi3x"},
            clear=True,
        ):
            schema = self._schema_by_id()

        self.assertTrue(
            {"left_image_path", "back_image_path", "right_image_path"}.isdisjoint(schema)
        )
        self.assertEqual(schema["output_name"]["default"], "pi3_point_cloud")

    def test_invalid_model_id_suppresses_pi3x_model_dir(self):
        for model_id in ("pi3", "foreign/pi3x", "pi3/unknown", "pi3/pi3x/extra"):
            with self.subTest(model_id=model_id), mock.patch.dict(
                os.environ,
                {"MODEL_ID": model_id, "MODEL_DIR": "/models/pi3/pi3x"},
                clear=True,
            ):
                schema = self._schema_by_id()
                self.assertNotIn("left_image_path", schema)
                self.assertEqual(schema["output_name"]["default"], "pi3_point_cloud")

    def test_model_dir_fallback_accepts_only_known_pi3_owners(self):
        self.assertEqual(
            generator._runtime_schema_node_id({"MODEL_DIR": "/models/pi3/pi3x"}),
            "pi3x",
        )
        self.assertEqual(
            generator._runtime_schema_node_id(
                {"MODEL_DIR": "/models/pi3/pi3x/model.safetensors"}
            ),
            "pi3x",
        )
        self.assertEqual(
            generator._runtime_schema_node_id(
                {"MODEL_ID": "", "MODEL_DIR": "/models/pi3/pi3x"}
            ),
            "pi3x",
        )
        self.assertEqual(
            generator._runtime_schema_node_id({"MODEL_DIR": "/models/foreign/pi3x"}),
            "generate",
        )
        self.assertEqual(
            generator._runtime_schema_node_id({"MODEL_DIR": "/models/pi3/unknown"}),
            "generate",
        )

    def test_runtime_schema_environment_is_restored(self):
        before = {name: os.environ.get(name) for name in ("MODEL_ID", "MODEL_DIR")}

        with mock.patch.dict(
            os.environ,
            {"MODEL_ID": "pi3/pi3x", "MODEL_DIR": "/models/pi3/pi3x"},
            clear=False,
        ):
            self.assertEqual(
                self._schema_by_id()["output_name"]["default"],
                "pi3x_point_cloud",
            )

        self.assertEqual(
            {name: os.environ.get(name) for name in ("MODEL_ID", "MODEL_DIR")},
            before,
        )


@unittest.skipIf(np is None or Image is None, "NumPy and Pillow are required for sidecar tests")
class Pi3ExtensionV023Tests(unittest.TestCase):
    def _image_bytes(self, color=(20, 40, 60)):
        stream = io.BytesIO()
        Image.new("RGB", (8, 6), color).save(stream, format="PNG")
        return stream.getvalue()

    @staticmethod
    def _write_file(path, payload):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)

    def test_manifest_separates_canonical_pi3_and_pi3x(self):
        manifest = json.loads((ROOT / "manifest.json").read_text())
        self.assertEqual(manifest["version"], "0.2.3")
        nodes = {node["id"]: node for node in manifest["nodes"]}
        self.assertEqual(set(nodes), {"generate", "pi3x"})
        self.assertEqual(nodes["generate"]["input"], "image")
        self.assertNotIn("io_contract", nodes["generate"])
        self.assertNotIn("input_ports", nodes["generate"])
        self.assertEqual(nodes["generate"]["weight_owner_id"], "generate")
        self.assertEqual(nodes["generate"]["hf_repo"], "yyfz233/Pi3")
        self.assertEqual(nodes["generate"]["download_check"], "model.safetensors")
        self.assertEqual(nodes["pi3x"]["weight_owner_id"], "pi3x")
        self.assertEqual(nodes["pi3x"]["hf_repo"], "yyfz233/Pi3X")
        self.assertEqual(nodes["pi3x"]["download_check"], "model.safetensors")
        self.assertEqual(nodes["pi3x"]["input"], "image")
        self.assertEqual(nodes["pi3x"]["output"], "mesh")
        self.assertEqual(nodes["pi3x"]["io_contract"], "named-v1")
        self.assertNotIn("inputs", nodes["pi3x"])
        self.assertEqual(
            nodes["pi3x"]["input_ports"],
            [
                {"name": "front", "label": "Front RGB Image", "type": "image", "required": True},
                {"name": "left", "label": "Left RGB Image", "type": "image", "required": False},
                {"name": "back", "label": "Back RGB Image", "type": "image", "required": False},
                {"name": "right", "label": "Right RGB Image", "type": "image", "required": False},
            ],
        )
        self.assertTrue(
            all(
                set(port) == {"name", "type", "label", "required"}
                for port in nodes["pi3x"]["input_ports"]
            )
        )
        assets = {asset["id"]: asset for asset in manifest["asset_requirements"]}
        self.assertEqual(assets["pi3-weights"]["repo"], "yyfz233/Pi3")
        self.assertEqual(assets["pi3x-weights"]["repo"], "yyfz233/Pi3X")
        self.assertEqual(assets["pi3-weights"]["target"], "models/pi3/generate/model.safetensors")
        self.assertEqual(assets["pi3x-weights"]["target"], "models/pi3/pi3x/model.safetensors")
        self.assertEqual(assets["pi3x-weights"]["storage_gb"], 5.44)
        self.assertEqual(assets["pi3-weights"]["license"], "CC-BY-NC-4.0; strictly noncommercial")
        self.assertEqual(assets["pi3x-weights"]["license"], "CC-BY-NC-4.0; strictly noncommercial")
        self.assertEqual(manifest["generator_class"], "Pi3Generator")

    def test_pi3_regression_defaults_and_pi3x_bounds(self):
        manifest = json.loads((ROOT / "manifest.json").read_text())
        nodes = {node["id"]: node for node in manifest["nodes"]}
        generate = {item["id"]: item for item in nodes["generate"]["params_schema"]}
        pi3x = {item["id"]: item for item in nodes["pi3x"]["params_schema"]}
        self.assertEqual(generate["output_name"]["default"], "pi3_point_cloud")
        self.assertEqual(generate["pixel_limit"]["default"], 255000)
        self.assertEqual(generate["confidence_threshold"]["default"], 0.1)
        self.assertEqual(generate["edge_filter"]["default"], "true")
        self.assertEqual(generate["edge_rtol"]["default"], 0.03)
        self.assertEqual(pi3x["output_name"]["default"], "pi3x_point_cloud")
        self.assertTrue(
            {"left_image_path", "back_image_path", "right_image_path"}.isdisjoint(pi3x)
        )
        for name in ("pixel_limit", "confidence_threshold", "edge_filter", "edge_rtol"):
            for field in ("type", "default", "min", "max", "step", "options"):
                if field in generate[name]:
                    self.assertEqual(pi3x[name][field], generate[name][field])

    def test_node_config_and_readiness_use_selected_owner(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            generate_dir = root / "models" / "pi3" / "generate"
            pi3x_dir = generate_dir.parent / "pi3x"
            generate_dir.mkdir(parents=True)
            pi3x_dir.mkdir()
            (generate_dir / "model.safetensors").write_bytes(b"pi3")
            gen = generator.Pi3Generator(model_dir=generate_dir, outputs_dir=root / "out")
            with mock.patch.object(generator, "_missing_dependencies", return_value=[]), mock.patch.object(
                generator, "_dependency_status", return_value={}
            ):
                self.assertTrue(gen.readiness_status()["ok"])
                gen.node_id = "pi3x"
                status = gen.readiness_status()
                self.assertFalse(status["ok"])
                self.assertEqual(status["details"]["model_id"], "pi3/pi3x")
                self.assertEqual(Path(status["details"]["expected_weights_path"]).parent, pi3x_dir.resolve())
                (pi3x_dir / "model.safetensors").write_bytes(b"pi3x")
                self.assertTrue(gen.readiness_status()["ok"])
                gen.node_id = "unknown"
                with self.assertRaisesRegex(ValueError, "Unknown Pi3 node"):
                    gen.readiness_status()

    def test_direct_checkpoint_paths_cannot_cross_owner_boundaries(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "models" / "pi3"
            generate_weights = root / "generate" / "model.safetensors"
            pi3x_weights = root / "pi3x" / "model.safetensors"
            generate_weights.parent.mkdir(parents=True)
            pi3x_weights.parent.mkdir(parents=True)
            generate_weights.write_bytes(b"pi3")

            gen = generator.Pi3Generator(model_dir=generate_weights, outputs_dir=Path(temp) / "out")
            self.assertEqual(gen._find_weights_path(), generate_weights.resolve())
            gen.node_id = "pi3x"
            self.assertIsNone(gen._find_weights_path())

            generate_weights.unlink()
            pi3x_weights.write_bytes(b"pi3x")
            gen = generator.Pi3Generator(model_dir=pi3x_weights, outputs_dir=Path(temp) / "out")
            gen.node_id = "pi3x"
            self.assertEqual(gen._find_weights_path(), pi3x_weights.resolve())
            gen.node_id = "generate"
            self.assertIsNone(gen._find_weights_path())

    def test_setup_reports_both_node_paths_and_legacy_fields(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            ext = root / "extensions" / "pi3"
            ext.mkdir(parents=True)
            config = setup.SetupConfig(python_exe="python", ext_dir=ext)
            diagnostics = setup.node_weight_diagnostics(config)
            self.assertEqual(set(diagnostics), {"generate", "pi3x"})
            self.assertTrue(diagnostics["generate"]["expected_weights_path"].endswith("models/pi3/generate/model.safetensors"))
            self.assertTrue(diagnostics["pi3x"]["expected_weights_path"].endswith("models/pi3/pi3x/model.safetensors"))
            self.assertEqual(
                setup.overall_setup_status(dependencies_ok=False, node_diagnostics=diagnostics),
                "needs_dependencies",
            )
            self.assertEqual(
                setup.overall_setup_status(dependencies_ok=True, node_diagnostics=diagnostics),
                "needs_weights",
            )
            generate_path = Path(diagnostics["generate"]["expected_weights_path"])
            generate_path.parent.mkdir(parents=True)
            generate_path.write_bytes(b"pi3")
            diagnostics = setup.node_weight_diagnostics(config)
            self.assertTrue(diagnostics["generate"]["weights_present"])
            self.assertFalse(diagnostics["pi3x"]["weights_present"])
            top_level_status = setup.overall_setup_status(
                dependencies_ok=True, node_diagnostics=diagnostics
            )
            self.assertEqual(top_level_status, "ready")
            status_path = setup.write_status(config, top_level_status, {})
            payload = json.loads(status_path.read_text())
            self.assertEqual(payload["schema"], "modly.setup-status.v1")
            self.assertEqual(payload["node_id"], "generate")
            self.assertEqual(payload["hf_repo"], "yyfz233/Pi3")
            self.assertEqual(payload["status"], "ready")
            self.assertTrue(payload["weights_present"])
            self.assertEqual(set(payload["nodes"]), {"generate", "pi3x"})
            self.assertEqual(payload["nodes"]["pi3x"]["status"], "needs_weights")

    def test_setup_explicit_owner_directories_are_isolated(self):
        for owner_id in ("generate", "pi3x"):
            with self.subTest(owner_id=owner_id), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                owner_dir = root / "models" / "pi3" / owner_id
                weights_path = owner_dir / "model.safetensors"
                owner_dir.mkdir(parents=True)
                weights_path.write_bytes(owner_id.encode("ascii"))
                config = setup.SetupConfig(
                    python_exe="python",
                    ext_dir=root / "extension",
                    model_dir=owner_dir,
                )

                diagnostics = setup.node_weight_diagnostics(config)
                other_id = "pi3x" if owner_id == "generate" else "generate"
                expected_status = "ready" if owner_id == "generate" else "needs_weights"
                self.assertTrue(diagnostics[owner_id]["weights_present"])
                self.assertFalse(diagnostics[other_id]["weights_present"])
                self.assertEqual(Path(diagnostics[owner_id]["expected_weights_path"]), weights_path)
                self.assertEqual(
                    setup.overall_setup_status(dependencies_ok=True, node_diagnostics=diagnostics),
                    expected_status,
                )

                status_path = setup.write_status(config, expected_status, {})
                payload = json.loads(status_path.read_text())
                self.assertEqual(payload["status"], expected_status)
                self.assertEqual(payload["weights_present"], owner_id == "generate")
                self.assertEqual(payload["nodes"][owner_id]["status"], "ready")
                self.assertEqual(payload["nodes"][other_id]["status"], "needs_weights")

    def test_setup_explicit_checkpoint_files_are_isolated(self):
        for owner_id in ("generate", "pi3x"):
            with self.subTest(owner_id=owner_id), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                weights_path = root / "models" / "pi3" / owner_id / "model.safetensors"
                weights_path.parent.mkdir(parents=True)
                weights_path.write_bytes(owner_id.encode("ascii"))
                config = setup.SetupConfig(
                    python_exe="python",
                    ext_dir=root / "extension",
                    model_dir=weights_path,
                )

                diagnostics = setup.node_weight_diagnostics(config)
                other_id = "pi3x" if owner_id == "generate" else "generate"
                self.assertTrue(diagnostics[owner_id]["weights_present"])
                self.assertFalse(diagnostics[other_id]["weights_present"])
                self.assertEqual(Path(diagnostics[owner_id]["expected_weights_path"]), weights_path)
                self.assertEqual(
                    setup.overall_setup_status(dependencies_ok=True, node_diagnostics=diagnostics),
                    "ready" if owner_id == "generate" else "needs_weights",
                )

    def test_named_views_canonicalize_real_png_bytes_in_port_order(self):
        with tempfile.TemporaryDirectory() as temp:
            input_dir = Path(temp) / "input"
            names = generator._prepare_pi3x_named_views(
                {
                    "right": self._image_bytes((90, 80, 70)),
                    "front": self._image_bytes((10, 20, 30)),
                    "back": self._image_bytes((60, 50, 40)),
                    "left": self._image_bytes((30, 40, 50)),
                },
                input_dir,
            )

            self.assertEqual(names, ["front", "left", "back", "right"])
            self.assertEqual(
                [path.name for path in sorted(input_dir.iterdir())],
                ["back.png", "front.png", "left.png", "right.png"],
            )
            for view_name in names:
                with Image.open(input_dir / f"{view_name}.png") as image:
                    self.assertEqual(image.format, "PNG")
                    self.assertEqual(image.mode, "RGB")
                    self.assertEqual(image.size, (8, 6))

            loaded_names = []

            def fake_loader(path, **_kwargs):
                loaded_names.extend(item.name for item in sorted(Path(path).iterdir()))
                return object()

            generator._load_prepared_views(
                fake_loader,
                input_dir,
                names,
                pixel_limit=255000,
            )
            self.assertEqual(
                loaded_names,
                ["00_front.png", "01_left.png", "02_back.png", "03_right.png"],
            )

    def test_named_views_reject_invalid_payloads(self):
        with tempfile.TemporaryDirectory() as temp:
            input_dir = Path(temp) / "input"
            with self.assertRaisesRegex(ValueError, "front image"):
                generator._prepare_pi3x_named_views({}, input_dir)
            with self.assertRaisesRegex(ValueError, "front image"):
                generator._prepare_pi3x_named_views({"front": b""}, input_dir)
            with self.assertRaisesRegex(ValueError, "unknown image ports"):
                generator._prepare_pi3x_named_views(
                    {"front": self._image_bytes(), "top": self._image_bytes()},
                    input_dir,
                )
            with self.assertRaisesRegex(TypeError, "must be bytes"):
                generator._prepare_pi3x_named_views(
                    {"front": self._image_bytes(), "left": "not-bytes"},
                    input_dir,
                )

    def test_generate_v2_rejects_pi3_and_delegates_pi3x_prepared_views(self):
        with tempfile.TemporaryDirectory() as temp:
            outputs = Path(temp) / "Workflows"
            gen = generator.Pi3Generator(outputs_dir=outputs)
            with self.assertRaisesRegex(ValueError, "only for pi3/pi3x"):
                gen.generate_v2({"front": self._image_bytes()}, {})
            self.assertFalse(outputs.exists())

            gen.node_id = "pi3x"
            sentinel = outputs / "sentinel.glb"
            captured = {}

            def fake_core(run, view_names, params, progress_cb, cancel_evt):
                captured["view_names"] = list(view_names)
                captured["params"] = params
                captured["files"] = {
                    path.relative_to(run.staging_dir).as_posix()
                    for path in run.staging_dir.rglob("*")
                    if path.is_file()
                }
                return sentinel

            with mock.patch.object(gen, "_generate_pi3x_in_run_dir", side_effect=fake_core) as core, mock.patch.object(
                gen, "load", side_effect=AssertionError("generate_v2 should not load before shared core")
            ) as load:
                result = gen.generate_v2(
                    {
                        "right": self._image_bytes((3, 3, 3)),
                        "front": self._image_bytes((1, 1, 1)),
                        "left": self._image_bytes((2, 2, 2)),
                    },
                    {"output_name": "named"},
                )

            self.assertEqual(result, sentinel)
            core.assert_called_once()
            load.assert_not_called()
            self.assertEqual(captured["view_names"], ["front", "left", "right"])
            self.assertEqual(captured["params"], {"output_name": "named"})
            self.assertEqual(
                captured["files"],
                {"input/front.png", "input/left.png", "input/right.png"},
            )
            self.assertEqual(list(outputs.iterdir()), [])

    def test_legacy_generate_routes_remain_present(self):
        image = self._image_bytes()
        params = {"output_name": "legacy"}
        gen = generator.Pi3Generator(outputs_dir=Path(tempfile.gettempdir()) / "unused")

        with mock.patch.object(gen, "_generate_pi3", return_value=Path("/tmp/pi3.glb")) as pi3:
            self.assertEqual(gen.generate(image, params), Path("/tmp/pi3.glb"))
            pi3.assert_called_once_with(image, params, None, None)

        gen.node_id = "pi3x"
        with mock.patch.object(gen, "_generate_pi3x", return_value=Path("/tmp/pi3x.glb")) as pi3x:
            self.assertEqual(gen.generate(image, params), Path("/tmp/pi3x.glb"))
            pi3x.assert_called_once_with(image, params, None, None)

    def test_side_images_are_secure_and_ordered(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            workspace = root / "workspace"
            input_dir = root / "input"
            workspace.mkdir()
            for name, color in (("left.webp", "red"), ("back.jpg", "green"), ("right.png", "blue")):
                Image.new("RGB", (4, 4), color).save(workspace / name)
            params = {
                "left_image_path": "left.webp",
                "back_image_path": "back.jpg",
                "right_image_path": "right.png",
            }
            with mock.patch.dict(os.environ, {"WORKSPACE_DIR": str(workspace)}, clear=False):
                names = generator._prepare_pi3x_views(self._image_bytes(), params, input_dir)
                self.assertEqual(names, ["front", "left", "back", "right"])
                self.assertEqual(
                    [path.name for path in sorted(input_dir.iterdir())],
                    ["back.png", "front.png", "left.png", "right.png"],
                )
                loaded_names = []

                def fake_loader(path, **_kwargs):
                    loaded_names.extend(item.name for item in sorted(Path(path).iterdir()))
                    return object()

                generator._load_prepared_views(
                    fake_loader,
                    input_dir,
                    names,
                    pixel_limit=255000,
                )
                self.assertEqual(
                    loaded_names,
                    ["00_front.png", "01_left.png", "02_back.png", "03_right.png"],
                )
                self.assertFalse(list(root.glob(".ordered-input.*.tmp")))
                with self.assertRaisesRegex(ValueError, "inside WORKSPACE_DIR"):
                    generator._resolve_side_image_path("../escape.png", workspace, "left_image_path")
                bad = workspace / "bad.txt"
                bad.write_text("not an image")
                with self.assertRaisesRegex(ValueError, "unsupported image extension"):
                    generator._resolve_side_image_path("bad.txt", workspace, "left_image_path")

            outside = root / "outside.png"
            Image.new("RGB", (2, 2)).save(outside)
            link = workspace / "link.png"
            try:
                link.symlink_to(outside)
            except OSError:
                self.skipTest("symlinks unavailable")
            with self.assertRaisesRegex(ValueError, "inside WORKSPACE_DIR"):
                generator._resolve_side_image_path(link, workspace, "right_image_path")

    def test_pi3x_state_key_validation(self):
        generator._validate_pi3x_state_keys(
            ["depth_encoder.patch.weight", "depth_emb"],
            ["ray_embed.proj.weight", "pose_inject_blk.0.weight"],
        )
        with self.assertRaisesRegex(RuntimeError, "Missing core keys"):
            generator._validate_pi3x_state_keys(["encoder.patch_embed.weight"], [])
        with self.assertRaisesRegex(RuntimeError, "unexpected non-multimodal"):
            generator._validate_pi3x_state_keys([], ["decoder.bad.weight"])

    def test_pi3_success_publishes_one_nested_complete_run(self):
        with tempfile.TemporaryDirectory() as temp:
            outputs = Path(temp) / "Workflows"
            run = generator._create_run_directory(outputs, "pi3")
            base = generator._sanitize_output_base("../../outside/cloud.glb")
            self.assertEqual(base, "cloud")
            self._write_file(run.staging_dir / "input" / "front.png", self._image_bytes())
            self._write_file(run.staging_dir / f"{base}.glb", b"glb")
            self._write_file(run.staging_dir / f"{base}.ply", b"ply")

            result = generator._publish_run_directory(
                run,
                ["input/front.png", f"{base}.glb", f"{base}.ply"],
                f"{base}.glb",
            )

            self.assertRegex(result.parent.name, r"^pi3_[0-9a-f]{12}$")
            self.assertEqual(result, result.parent / "cloud.glb")
            self.assertEqual(
                {path.relative_to(result.parent).as_posix() for path in result.parent.rglob("*") if path.is_file()},
                {"input/front.png", "cloud.glb", "cloud.ply"},
            )
            self.assertEqual(list(outputs.iterdir()), [result.parent])
            self.assertFalse(list(outputs.glob("*.glb")))
            self.assertFalse(list(outputs.glob(".*")))
            self.assertFalse((Path(temp) / "outside" / "cloud.glb").exists())

    def test_pi3x_success_publishes_inputs_and_every_sidecar_together(self):
        with tempfile.TemporaryDirectory() as temp:
            outputs = Path(temp) / "Workflows"
            run = generator._create_run_directory(outputs, "pi3x")
            view_names = ["front", "left", "back", "right"]
            base = "bundle"
            for view_name in view_names:
                self._write_file(
                    run.staging_dir / "input" / f"{view_name}.png",
                    self._image_bytes(),
                )
            self._write_file(run.staging_dir / f"{base}.glb", b"glb")
            self._write_file(run.staging_dir / f"{base}.ply", b"ply")
            filenames = {
                "glb": f"{base}.glb",
                "ply": f"{base}.ply",
                "npz": f"{base}_pi3x.npz",
                "metadata": f"{base}_metadata.json",
                "depth_previews": [f"{base}_depth_{name}.png" for name in view_names],
                "confidence_previews": [f"{base}_confidence_{name}.png" for name in view_names],
            }
            generator._write_pi3x_sidecars(
                run.staging_dir,
                base,
                arrays=self._valid_sidecar_arrays(n=4),
                metadata={"filenames": filenames},
                view_names=view_names,
            )
            bundle_names = generator._pi3x_bundle_names(base, view_names)
            expected = [f"input/{name}.png" for name in view_names] + bundle_names

            result = generator._publish_run_directory(
                run,
                expected,
                f"{base}.glb",
            )

            self.assertRegex(result.parent.name, r"^pi3x_[0-9a-f]{12}$")
            self.assertEqual(result, result.parent / "bundle.glb")
            self.assertEqual(
                {path.relative_to(result.parent).as_posix() for path in result.parent.rglob("*") if path.is_file()},
                set(expected),
            )
            metadata = json.loads((result.parent / filenames["metadata"]).read_text())
            for value in metadata["filenames"].values():
                for name in value if isinstance(value, list) else [value]:
                    self.assertEqual(Path(name).name, name)
                    self.assertTrue((result.parent / name).is_file())
            self.assertEqual(list(outputs.iterdir()), [result.parent])
            self.assertFalse(list(outputs.glob("*.glb")))
            self.assertFalse(list(outputs.glob(".*")))

    def test_concurrent_run_publications_are_distinct_and_complete(self):
        with tempfile.TemporaryDirectory() as temp:
            outputs = Path(temp) / "Workflows"
            barrier = threading.Barrier(2)
            results = []
            failures = []

            def publish():
                try:
                    run = generator._create_run_directory(outputs, "pi3")
                    self._write_file(run.staging_dir / "input" / "front.png", b"front")
                    self._write_file(run.staging_dir / "cloud.glb", b"glb")
                    self._write_file(run.staging_dir / "cloud.ply", b"ply")
                    barrier.wait(timeout=2)
                    results.append(
                        generator._publish_run_directory(
                            run,
                            ["input/front.png", "cloud.glb", "cloud.ply"],
                            "cloud.glb",
                        )
                    )
                except Exception as exc:  # pragma: no cover - diagnostic path
                    failures.append(exc)

            threads = [threading.Thread(target=publish) for _ in range(2)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=3)

            self.assertFalse(failures)
            self.assertEqual(len(results), 2)
            self.assertEqual(len({path.parent for path in results}), 2)
            for result in results:
                self.assertEqual(
                    {path.relative_to(result.parent).as_posix() for path in result.parent.rglob("*") if path.is_file()},
                    {"input/front.png", "cloud.glb", "cloud.ply"},
                )
            self.assertFalse(list(outputs.glob(".*")))

    def test_incomplete_output_cleans_hidden_stage_and_publishes_nothing(self):
        with tempfile.TemporaryDirectory() as temp:
            outputs = Path(temp) / "Workflows"
            with self.assertRaisesRegex(RuntimeError, "escaped or is missing"):
                with generator._staged_run(outputs, "pi3") as run:
                    self._write_file(run.staging_dir / "input" / "front.png", b"front")
                    self._write_file(run.staging_dir / "cloud.ply", b"ply")
                    generator._publish_run_directory(
                        run,
                        ["input/front.png", "cloud.glb", "cloud.ply"],
                        "cloud.glb",
                    )
            self.assertEqual(list(outputs.iterdir()), [])

    def test_publish_error_cleans_hidden_stage_and_publishes_nothing(self):
        with tempfile.TemporaryDirectory() as temp:
            outputs = Path(temp) / "Workflows"
            with self.assertRaisesRegex(OSError, "synthetic publish failure"):
                with generator._staged_run(outputs, "pi3") as run:
                    self._write_file(run.staging_dir / "input" / "front.png", b"front")
                    self._write_file(run.staging_dir / "cloud.glb", b"glb")
                    self._write_file(run.staging_dir / "cloud.ply", b"ply")
                    with mock.patch.object(
                        generator.os,
                        "replace",
                        side_effect=OSError("synthetic publish failure"),
                    ):
                        generator._publish_run_directory(
                            run,
                            ["input/front.png", "cloud.glb", "cloud.ply"],
                            "cloud.glb",
                        )
            self.assertEqual(list(outputs.iterdir()), [])

    def test_cancellation_cleans_hidden_stage_and_publishes_nothing(self):
        with tempfile.TemporaryDirectory() as temp:
            outputs = Path(temp) / "Workflows"
            cancel_evt = threading.Event()
            cancel_evt.set()
            with self.assertRaises(generator.GenerationCancelled):
                with generator._staged_run(outputs, "pi3x") as run:
                    self._write_file(run.staging_dir / "input" / "front.png", b"front")
                    self._write_file(run.staging_dir / "bundle.glb", b"glb")
                    self._write_file(run.staging_dir / "bundle.ply", b"ply")
                    generator._publish_run_directory(
                        run,
                        ["input/front.png", "bundle.glb", "bundle.ply"],
                        "bundle.glb",
                        cancel_evt,
                    )
            self.assertEqual(list(outputs.iterdir()), [])

    def test_glb_writer_rejects_non_finite_values_without_file(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "bad.glb"
            points = np.asarray([[0.0, np.nan, 1.0]], dtype=np.float32)
            colors = np.asarray([[1.0, 0.0, np.inf]], dtype=np.float32)
            with self.assertRaisesRegex(RuntimeError, "NaN or infinite"):
                generator._write_point_cloud_glb(points, np.zeros_like(points), output)
            self.assertFalse(output.exists())
            with self.assertRaisesRegex(RuntimeError, "NaN or infinite"):
                generator._write_point_cloud_glb(np.zeros_like(points), colors, output)
            self.assertFalse(output.exists())

    def test_non_finite_metadata_and_bundle_arrays_publish_nothing(self):
        with tempfile.TemporaryDirectory() as temp:
            stage = Path(temp)
            arrays = self._valid_sidecar_arrays()
            metadata_path = stage / "metadata.json"
            with self.assertRaises(ValueError):
                generator._atomic_write_json(metadata_path, {"metric": float("nan")})
            self.assertFalse(metadata_path.exists())
            arrays["rays"][0, 0, 0, 0] = np.inf
            with self.assertRaisesRegex(RuntimeError, "rays contains NaN or infinite"):
                generator._write_pi3x_sidecars(
                    stage,
                    "bundle",
                    arrays=arrays,
                    metadata={"filenames": {}},
                    view_names=["front", "left"],
                )
            self.assertEqual(list(stage.iterdir()), [])

            arrays = self._valid_sidecar_arrays()
            arrays["metric"] = np.asarray(0.0, np.float32)
            with self.assertRaisesRegex(RuntimeError, "metric must be finite and positive"):
                generator._write_pi3x_sidecars(
                    stage,
                    "bundle",
                    arrays=arrays,
                    metadata={"filenames": {}},
                    view_names=["front", "left"],
                )
            self.assertEqual(list(stage.iterdir()), [])

    def _valid_sidecar_arrays(self, n=2):
        h, w = 4, 5
        depth = np.linspace(1.0, 4.0, n * h * w, dtype=np.float32).reshape(n, h, w)
        return {
            "points": np.zeros((n, h, w, 3), np.float32),
            "local_points": np.zeros((n, h, w, 3), np.float32),
            "rays": np.zeros((n, h, w, 3), np.float32),
            "depth": depth,
            "confidence_logits": np.zeros((n, h, w), np.float32),
            "confidence": np.full((n, h, w), 0.5, np.float32),
            "valid_mask": np.ones((n, h, w), bool),
            "camera_poses": np.repeat(np.eye(4, dtype=np.float32)[None], n, axis=0),
            "intrinsics": np.repeat(np.eye(3, dtype=np.float32)[None], n, axis=0),
            "metric": np.asarray(1.25, np.float32),
            "colors": np.zeros((n, h, w, 3), np.float32),
        }

    def test_sidecar_export_writes_npz_json_and_png(self):
        with tempfile.TemporaryDirectory() as temp:
            stage = Path(temp)
            arrays = self._valid_sidecar_arrays()
            metadata = {"filenames": {}}
            written = generator._write_pi3x_sidecars(
                stage,
                "bundle",
                arrays=arrays,
                metadata=metadata,
                view_names=["front", "left"],
            )
            self.assertEqual(len(written), 6)
            with np.load(stage / "bundle_pi3x.npz") as archive:
                self.assertEqual(archive["view_names"].tolist(), ["front", "left"])
                self.assertEqual(archive["depth"].dtype, np.float32)
                self.assertIn("valid_mask", archive.files)
            metadata_payload = json.loads((stage / "bundle_metadata.json").read_text())
            self.assertEqual(set(metadata_payload["depth_preview_normalization"]), {"front", "left"})
            with Image.open(stage / "bundle_depth_front.png") as depth_preview:
                self.assertIn(depth_preview.mode, {"I", "I;16"})
                self.assertGreater(depth_preview.getextrema()[1], 255)
            with Image.open(stage / "bundle_confidence_left.png") as confidence_preview:
                self.assertEqual(confidence_preview.mode, "L")

    def test_failed_pi3x_generation_cleans_run_directory(self):
        with tempfile.TemporaryDirectory() as temp:
            outputs = Path(temp)
            gen = generator.Pi3Generator(outputs_dir=outputs)
            gen.node_id = "pi3x"
            with mock.patch.object(generator, "_missing_dependencies", return_value=[]), mock.patch.object(
                gen, "load", side_effect=RuntimeError("synthetic load failure")
            ):
                with self.assertRaisesRegex(RuntimeError, "synthetic load failure"):
                    gen.generate(self._image_bytes(), {})
            self.assertEqual(list(outputs.iterdir()), [])

    def test_failed_legacy_pi3_generation_cleans_run_directory(self):
        with tempfile.TemporaryDirectory() as temp:
            outputs = Path(temp)
            gen = generator.Pi3Generator(outputs_dir=outputs)
            with mock.patch.object(generator, "_missing_dependencies", return_value=["torch"]):
                with self.assertRaisesRegex(RuntimeError, "runtime dependencies are missing"):
                    gen.generate(self._image_bytes(), {})
            self.assertEqual(list(outputs.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
