import importlib.util

import pytest

from model_smoke_helpers import TORCH_MODEL_SPECS, import_or_skip, smoke_params


def _skip_if_gpu_tests_disabled(enabled):
    if not enabled:
        pytest.skip("GPU tests require --run-gpu-tests")


def test_torch_cuda_tensor_operations_run_on_gpu(gpu_tests_enabled):
    _skip_if_gpu_tests_disabled(gpu_tests_enabled)
    torch = pytest.importorskip("torch")

    assert torch.cuda.is_available(), "PyTorch cannot see a CUDA GPU"
    tensor = torch.ones(4, device="cuda")
    result = tensor * 2

    assert result.is_cuda
    assert result.cpu().tolist() == [2, 2, 2, 2]


@pytest.mark.parametrize(
    "spec",
    TORCH_MODEL_SPECS,
    ids=lambda spec: f"{spec.module_name}.{spec.class_name}",
)
def test_torch_model_networks_are_placed_on_cuda_when_available(spec, gpu_tests_enabled):
    _skip_if_gpu_tests_disabled(gpu_tests_enabled)
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.fail("PyTorch cannot see a CUDA GPU")

    cls = import_or_skip(spec)
    params = smoke_params(spec, epochs=0)
    params["device"] = "cuda"
    model = cls(params)
    if spec.module_name in {
        "nilmtk_contrib.torch.msdc",
        "nilmtk_contrib.torch.msdc_without_crf",
    }:
        network = model.return_network("fridge")
    else:
        network = model.return_network()

    networks = network if isinstance(network, (tuple, list)) else (network,)
    for item in networks:
        first_parameter = next(item.parameters(), None)
        assert first_parameter is not None
        assert first_parameter.device.type == "cuda"


def test_tensorflow_reports_gpu_devices_when_available(gpu_tests_enabled):
    _skip_if_gpu_tests_disabled(gpu_tests_enabled)
    if importlib.util.find_spec("tensorflow") is None:
        pytest.skip("tensorflow is not installed")

    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        pytest.skip("TensorFlow is installed but cannot see GPU devices")

    with tf.device("/GPU:0"):
        result = tf.constant([1.0, 2.0]) + tf.constant([3.0, 4.0])

    assert result.numpy().tolist() == [4.0, 6.0]
