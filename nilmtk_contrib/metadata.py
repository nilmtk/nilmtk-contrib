"""Public model catalog metadata."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelCatalogEntry:
    name: str
    backend: str
    module_path: str
    class_name: str
    exported_from: str | None = None


MODEL_CATALOG = (
    ModelCatalogEntry("AFHMM", "classical", "nilmtk_contrib.disaggregate.afhmm", "AFHMM", "nilmtk_contrib.disaggregate"),
    ModelCatalogEntry("AFHMM_SAC", "classical", "nilmtk_contrib.disaggregate.afhmm_sac", "AFHMM_SAC", "nilmtk_contrib.disaggregate"),
    ModelCatalogEntry("DSC", "classical", "nilmtk_contrib.disaggregate.dsc", "DSC", "nilmtk_contrib.disaggregate"),
    ModelCatalogEntry("BERT", "tensorflow", "nilmtk_contrib.disaggregate.bert", "BERT", "nilmtk_contrib.disaggregate"),
    ModelCatalogEntry("DAE", "tensorflow", "nilmtk_contrib.disaggregate.dae", "DAE", "nilmtk_contrib.disaggregate"),
    ModelCatalogEntry("RNN", "tensorflow", "nilmtk_contrib.disaggregate.rnn", "RNN", "nilmtk_contrib.disaggregate"),
    ModelCatalogEntry("RNN_attention", "tensorflow", "nilmtk_contrib.disaggregate.rnn_attention", "RNN_attention", "nilmtk_contrib.disaggregate"),
    ModelCatalogEntry("RNN_attention_classification", "tensorflow", "nilmtk_contrib.disaggregate.rnn_attention_classification", "RNN_attention_classification", "nilmtk_contrib.disaggregate"),
    ModelCatalogEntry("ResNet", "tensorflow", "nilmtk_contrib.disaggregate.resnet", "ResNet", "nilmtk_contrib.disaggregate"),
    ModelCatalogEntry("ResNet_classification", "tensorflow", "nilmtk_contrib.disaggregate.resnet_classification", "ResNet_classification", "nilmtk_contrib.disaggregate"),
    ModelCatalogEntry("Seq2Point", "tensorflow", "nilmtk_contrib.disaggregate.seq2point", "Seq2Point", "nilmtk_contrib.disaggregate"),
    ModelCatalogEntry("Seq2Seq", "tensorflow", "nilmtk_contrib.disaggregate.seq2seq", "Seq2Seq", "nilmtk_contrib.disaggregate"),
    ModelCatalogEntry("WindowGRU", "tensorflow", "nilmtk_contrib.disaggregate.WindowGRU", "WindowGRU", "nilmtk_contrib.disaggregate"),
    ModelCatalogEntry("BERT", "torch", "nilmtk_contrib.torch.bert", "BERT", "nilmtk_contrib.torch"),
    ModelCatalogEntry("ConvLSTM", "torch", "nilmtk_contrib.torch.conv_lstm", "ConvLSTM", "nilmtk_contrib.torch"),
    ModelCatalogEntry("DAE", "torch", "nilmtk_contrib.torch.dae", "DAE", "nilmtk_contrib.torch"),
    ModelCatalogEntry("MSDC", "torch", "nilmtk_contrib.torch.msdc", "MSDC", "nilmtk_contrib.torch"),
    ModelCatalogEntry("MSDC without CRF", "torch", "nilmtk_contrib.torch.msdc_without_crf", "MSDC", None),
    ModelCatalogEntry("NILMFormer", "torch", "nilmtk_contrib.torch.nilmformer", "NILMFormer", "nilmtk_contrib.torch"),
    ModelCatalogEntry("Reformer", "torch", "nilmtk_contrib.torch.reformer", "Reformer", "nilmtk_contrib.torch"),
    ModelCatalogEntry("ResNet", "torch", "nilmtk_contrib.torch.resnet", "ResNet", "nilmtk_contrib.torch"),
    ModelCatalogEntry("ResNet_classification", "torch", "nilmtk_contrib.torch.resnet_classification", "ResNet_classification", "nilmtk_contrib.torch"),
    ModelCatalogEntry("RNN", "torch", "nilmtk_contrib.torch.rnn", "RNN", "nilmtk_contrib.torch"),
    ModelCatalogEntry("RNN_attention", "torch", "nilmtk_contrib.torch.rnn_attention", "RNN_attention", "nilmtk_contrib.torch"),
    ModelCatalogEntry("RNN_attention_classification", "torch", "nilmtk_contrib.torch.rnn_attention_classification", "RNN_attention_classification", "nilmtk_contrib.torch"),
    ModelCatalogEntry("Seq2PointTorch", "torch", "nilmtk_contrib.torch.seq2point", "Seq2PointTorch", "nilmtk_contrib.torch"),
    ModelCatalogEntry("Seq2Seq", "torch", "nilmtk_contrib.torch.seq2seq", "Seq2Seq", "nilmtk_contrib.torch"),
    ModelCatalogEntry("TCN", "torch", "nilmtk_contrib.torch.TCN", "TCN", "nilmtk_contrib.torch"),
    ModelCatalogEntry("WindowGRU", "torch", "nilmtk_contrib.torch.WindowGRU", "WindowGRU", "nilmtk_contrib.torch"),
)


def model_catalog_by_module():
    """Return model catalog entries keyed by implementation module path."""
    return {entry.module_path: entry for entry in MODEL_CATALOG}


__all__ = ["MODEL_CATALOG", "ModelCatalogEntry", "model_catalog_by_module"]
