import torch.nn as nn
from onmt.models import NMTModel


def _get_level(levels):
    levels_list = levels.tolist()
    assert (len(set(levels_list)) == 1)  # assert all examples in the same level
    return levels_list[0]


class MultiDecodersNMTModel(nn.Module):
    def __init__(self, encoder, decoders):
        super(MultiDecodersNMTModel, self).__init__()
        self.decoders = decoders
        self._nmt_model = NMTModel(encoder, list(decoders.values())[0])  # arbitrary selection of decoder, will be replaced in action

    def encoder(self):
        return self._nmt_model.encoder

    def decoder(self):
        return self._nmt_model.decoder

    def set_level(self, level):
        self._nmt_model.decoder = self.decoders[str(level)]

    def forward(self, src, tgt, levels, lengths, bptt=False):
        level = _get_level(levels)
        self._nmt_model.decoder = self.decoders[str(level)]

        return self._nmt_model(src, tgt, lengths, bptt)
