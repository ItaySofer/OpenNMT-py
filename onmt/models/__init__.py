"""Module defining models."""
from onmt.models.model_saver import build_model_saver, ModelSaver
from onmt.models.model import NMTModel
from onmt.models.multi_decoders_model import MultiDecodersNMTModel

__all__ = ["build_model_saver", "ModelSaver",
           "NMTModel", "MultiDecodersNMTModel", "check_sru_requirement"]
