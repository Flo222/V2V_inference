# V2X-ViT

Baseline-native responsibility: backbone, shrink/native compressor, priors/spatial correction and V2XTransformer fusion. `V2XViTNativePayloadAdapter` exposes the post-native-compressor tensor as the real communication payload; max-CAV padding and HxW-repeated priors are created after communication.
