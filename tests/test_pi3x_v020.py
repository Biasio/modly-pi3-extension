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


@unittest.skipIf(np is None or Image is None, "NumPy and Pillow are required for sidecar tests")
class Pi3ExtensionV020Tests(unittest.TestCase):
    def _image_bytes(self, color=(20, 40, 60)):
        stream = io.BytesIO()
        Image.new("RGB", (8, 6), color).save(stream, format="PNG")
        return stream.getvalue()

    def test_manifest_separates_canonical_pi3_and_pi3x(self):
        manifest = json.loads((ROOT / "manifest.json").read_text())
        self.assertEqual(manifest["version"], "0.2.0")
        nodes = {node["id"]: node for node in manifest["nodes"]}
        self.assertEqual(set(nodes), {"generate", "pi3x"})
        self.assertEqual(nodes["generate"]["weight_owner_id"], "generate")
        self.assertEqual(nodes["generate"]["hf_repo"], "yyfz233/Pi3")
        self.assertEqual(nodes["pi3x"]["weight_owner_id"], "pi3x")
        self.assertEqual(nodes["pi3x"]["hf_repo"], "yyfz233/Pi3X")
        assets = {asset["id"]: asset for asset in manifest["asset_requirements"]}
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
                    ["00_front.png", "01_left.png", "02_back.png", "03_right.png"],
                )
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

    def test_atomic_reservation_checks_existing_sidecars(self):
        with tempfile.TemporaryDirectory() as temp:
            outputs = Path(temp)
            (outputs / "cloud_metadata.json").write_text("{}")
            reservation = generator._reserve_output_base(
                outputs,
                "cloud",
                lambda base: generator._pi3x_bundle_names(base, ["front"]),
            )
            try:
                self.assertNotEqual(reservation.base_name, "cloud")
            finally:
                reservation.release()

    def test_atomic_reservation_gives_concurrent_callers_distinct_bases(self):
        with tempfile.TemporaryDirectory() as temp:
            outputs = Path(temp)
            barrier = threading.Barrier(2)
            reservations = []
            failures = []

            def reserve():
                try:
                    reservation = generator._reserve_output_base(
                        outputs,
                        "cloud",
                        lambda base: [f"{base}.glb", f"{base}.ply"],
                    )
                    reservations.append(reservation)
                    barrier.wait(timeout=2)
                except Exception as exc:  # pragma: no cover - diagnostic path
                    failures.append(exc)

            threads = [threading.Thread(target=reserve) for _ in range(2)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=3)
            try:
                self.assertFalse(failures)
                self.assertEqual(len(reservations), 2)
                self.assertEqual(len({item.base_name for item in reservations}), 2)
            finally:
                for reservation in reservations:
                    reservation.release()
            self.assertFalse(list(outputs.glob(".*.pi3-reservation")))

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

    def _valid_sidecar_arrays(self):
        n, h, w = 2, 4, 5
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
            self.assertFalse(any(path.name.startswith("pi3x_") for path in outputs.iterdir()))

    def test_failed_legacy_pi3_generation_cleans_run_directory(self):
        with tempfile.TemporaryDirectory() as temp:
            outputs = Path(temp)
            gen = generator.Pi3Generator(outputs_dir=outputs)
            with mock.patch.object(generator, "_missing_dependencies", return_value=["torch"]):
                with self.assertRaisesRegex(RuntimeError, "runtime dependencies are missing"):
                    gen.generate(self._image_bytes(), {})
            self.assertFalse(any(path.name.startswith("pi3_") for path in outputs.iterdir()))


if __name__ == "__main__":
    unittest.main()
