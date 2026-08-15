import torch
import torch.nn as nn


class NaiveCompressor(nn.Module):
    """
    A very naive compression that only compresses channels.

    The codec boundary is exposed without changing the original
    forward computation:

        encoder
        -> optional FP16-to-FP32 round trip
        -> decoder
    """

    def __init__(self, input_dim, compress_raito):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(
                input_dim,
                input_dim // compress_raito,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(
                input_dim // compress_raito,
                eps=1e-3,
                momentum=0.01,
            ),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(
                input_dim // compress_raito,
                input_dim,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(
                input_dim,
                eps=1e-3,
                momentum=0.01,
            ),
            nn.ReLU(),
            nn.Conv2d(
                input_dim,
                input_dim,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(
                input_dim,
                eps=1e-3,
                momentum=0.01,
            ),
            nn.ReLU(),
        )

    def encode_for_wire(self, x, use_fp16=False):
        """
        Run the original encoder and expose its output as the
        sender-side communication tensor.

        When use_fp16=True, the returned tensor remains FP16 so it
        can be serialized directly. The legacy forward previously
        performed FP16 -> FP32 immediately before the decoder.
        """
        encoded = self.encoder(x)

        if use_fp16:
            encoded = encoded.to(torch.float16)

        return encoded

    def decode_from_wire(self, encoded, use_fp16=False):
        """
        Restore the original decoder input and run the decoder.

        Converting FP16 back to FP32 here exactly preserves the
        legacy inference sequence:
            encoder -> FP16 -> FP32 -> decoder
        """
        if use_fp16:
            encoded = encoded.to(torch.float32)

        return self.decoder(encoded)

    def forward(self, x, use_fp16=False):
        encoded = self.encode_for_wire(
            x,
            use_fp16=use_fp16,
        )

        return self.decode_from_wire(
            encoded,
            use_fp16=use_fp16,
        )


class ImprovedCompressor(nn.Module):
    """
    Compress both spatial dimensions and channels while restoring
    the original output dimensions.
    """

    def __init__(self, input_dim, compress_ratio, stride=4):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(
                input_dim,
                input_dim // compress_ratio,
                kernel_size=3,
                stride=stride,
                padding=1,
            ),
            nn.BatchNorm2d(
                input_dim // compress_ratio,
                eps=1e-3,
                momentum=0.01,
            ),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                input_dim // compress_ratio,
                input_dim,
                kernel_size=3,
                stride=stride,
                padding=1,
                output_padding=stride - 1,
            ),
            nn.BatchNorm2d(
                input_dim,
                eps=1e-3,
                momentum=0.01,
            ),
            nn.ReLU(),
            nn.Conv2d(
                input_dim,
                input_dim,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(
                input_dim,
                eps=1e-3,
                momentum=0.01,
            ),
            nn.ReLU(),
        )

    def encode_for_wire(self, x, use_fp16=False):
        encoded = self.encoder(x)

        if use_fp16:
            encoded = encoded.to(torch.float16)

        return encoded

    def decode_from_wire(self, encoded, use_fp16=False):
        if use_fp16:
            encoded = encoded.to(torch.float32)

        return self.decoder(encoded)

    def forward(self, x, use_fp16=False):
        encoded = self.encode_for_wire(
            x,
            use_fp16=use_fp16,
        )

        return self.decode_from_wire(
            encoded,
            use_fp16=use_fp16,
        )
