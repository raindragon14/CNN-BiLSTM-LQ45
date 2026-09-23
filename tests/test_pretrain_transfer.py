"""Encoder transfer tests; run directly: python3 tests/test_pretrain_transfer.py.

The pre-trained encoder state must load into the supervised CNN-BiLSTM
without shape errors (the old 7-channel vs 8-feature mismatch crashed
`walk-forward-pretrain`). The data here is synthetic and only validates
the pipeline, not results.
"""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lq45.models.cnn_bilstm import CNNBiLSTM
from lq45.models.decoder import MAEDecoder
from lq45.models.encoder import CNNBiLSTMEncoder
from lq45.models.pretrain import load_pretrained_encoder

N_FEATURES = 8
LOOKBACK = 60


def _encoder() -> CNNBiLSTMEncoder:
    return CNNBiLSTMEncoder(
        n_features=N_FEATURES,
        filters=[32, 64],
        kernel_size=3,
        pooling=2,
        units=64,
        layers=2,
        batchnorm=True,
        dropout_cnn=0.2,
        dropout_lstm=0.2,
        bidirectional=True,
    )


def test_encoder_state_loads_into_model() -> None:
    torch.manual_seed(0)
    encoder = _encoder()
    model = CNNBiLSTM(
        n_features=N_FEATURES,
        filters=[32, 64],
        kernel_size=3,
        pooling=2,
        units=64,
        layers=2,
        batchnorm=True,
        dropout_cnn=0.2,
        dropout_lstm=0.2,
        dropout_dense=0.2,
        bidirectional=True,
        activation="relu",
    )
    load_pretrained_encoder(model.encoder, encoder.state_dict())
    out = model(torch.zeros(4, N_FEATURES, LOOKBACK))
    assert out.shape == (4,)


def test_decoder_reconstructs_all_channels() -> None:
    torch.manual_seed(0)
    encoder = _encoder()
    decoder = MAEDecoder(
        latent_dim=encoder.output_dim,
        n_channels=N_FEATURES,
        lookback=LOOKBACK,
        pooling=2,
        hidden=32,
        layers=2,
    )
    z = encoder(torch.rand(3, N_FEATURES, LOOKBACK), return_sequence=True)
    recon = decoder(z)
    assert recon.shape == (3, N_FEATURES, LOOKBACK)


def main() -> int:
    test_encoder_state_loads_into_model()
    test_decoder_reconstructs_all_channels()
    print("test_pretrain_transfer.py: 2 tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
