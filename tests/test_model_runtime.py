import pytest

from nilmtk_contrib.utils.model import checkpoint_path, initialize_runtime


class RuntimeOnlyModel:
    pass


class PersistentModel:
    def save_model(self):
        return "saved"

    def load_model(self):
        return "loaded"


def test_initialize_runtime_adds_clear_persistence_fallbacks():
    model = RuntimeOnlyModel()
    model.MODEL_NAME = "RuntimeOnly"

    initialize_runtime(model, {"seed": 123, "verbose": False}, backends=("python",))

    assert model.seed == 123
    assert model.verbose is False
    with pytest.raises(NotImplementedError, match="RuntimeOnly"):
        model.save_model()
    with pytest.raises(NotImplementedError, match="RuntimeOnly"):
        model.load_model()


def test_initialize_runtime_preserves_real_persistence_methods():
    model = PersistentModel()

    initialize_runtime(model, {}, backends=("python",))

    assert model.save_model() == "saved"
    assert model.load_model() == "loaded"


def test_initialize_runtime_replaces_nilmtk_abstract_persistence_methods():
    nilmtk = pytest.importorskip("nilmtk.disaggregate")

    class ModelWithoutPersistence(nilmtk.Disaggregator):
        MODEL_NAME = "NoPersistence"

    model = ModelWithoutPersistence()
    initialize_runtime(model, {}, backends=("python",))

    with pytest.raises(NotImplementedError, match="NoPersistence"):
        model.save_model()
    with pytest.raises(NotImplementedError, match="NoPersistence"):
        model.load_model()


def test_checkpoint_path_returns_string_for_keras_compatibility():
    path = checkpoint_path(".h5")

    assert isinstance(path, str)
    assert path.endswith("checkpoint.h5")
