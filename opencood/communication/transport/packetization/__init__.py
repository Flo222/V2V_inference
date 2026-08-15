"""
Feature packetization package for ARCE communication simulation.

This subpackage contains packet-side utilities used by the ARCE
communication layer in OPV2V / V2X-ViT experiments.

Planned modules:

1. packetizer.py
   Split an intermediate BEV feature tensor [C, H, W] into spatial packets.

2. size_estimator.py
   Estimate raw bytes, compressed bytes, parity bytes, transmitted bytes,
   and received bytes.

Design note:
    Keep this __init__.py lightweight.

    Do not eagerly import concrete classes such as FeaturePacketizer here.
    The packetizer may depend on torch, and importing it here can introduce
    unnecessary dependencies or circular imports while the project is still
    under development.

    Concrete modules should be imported directly where they are used:

        from opencood.communication.transport.packetization.packetizer import FeaturePacketizer
        from opencood.communication.transport.packetization.size_estimator import estimate_feature_bytes
"""

PACKET_MODE_GRID = "grid"
PACKET_MODE_BLOCK = "block"
PACKET_MODE_PATCH_MAX_SIZE = "patch_max_size"

VALID_PACKET_MODES = (
    PACKET_MODE_GRID,
    PACKET_MODE_BLOCK,
    PACKET_MODE_PATCH_MAX_SIZE,
)


DEFAULT_PACKET_MODE = PACKET_MODE_GRID
DEFAULT_GRID_SIZE = (10, 10)

# Packet mask convention used across ARCE:
#     True  -> packet is lost / missing / invalid
#     False -> packet is received / available / valid
#
# Receive masks use the opposite convention:
#     True  -> packet is received
#     False -> packet is lost
LOSS_MASK_TRUE_MEANS_LOST = True


def normalize_packet_mode(mode):
    """
    Normalize packetization mode.

    Parameters
    ----------
    mode : str
        Packetization mode. Expected values:
            "grid"
            "block"

    Returns
    -------
    str
        Normalized lower-case mode.

    Raises
    ------
    ValueError
        If mode is not supported.
    """
    if mode is None:
        return DEFAULT_PACKET_MODE

    mode = str(mode).strip().lower()

    if mode not in VALID_PACKET_MODES:
        raise ValueError(
            f"Invalid packetization mode: {mode}. "
            f"Expected one of: {VALID_PACKET_MODES}."
        )

    return mode


def is_valid_packet_mode(mode):
    """
    Check whether a packetization mode is valid.

    Parameters
    ----------
    mode : str
        Input packetization mode.

    Returns
    -------
    bool
        True if valid, otherwise False.
    """
    try:
        normalize_packet_mode(mode)
        return True
    except ValueError:
        return False


def normalize_pair(value, name="value", default=None, min_value=1):
    """
    Normalize a scalar or length-2 list/tuple into an integer pair.

    Examples
    --------
    normalize_pair(10) -> (10, 10)
    normalize_pair([10, 8]) -> (10, 8)

    Parameters
    ----------
    value : int, list, tuple, or None
        Input value.

    name : str
        Name used in error messages.

    default : int, list, tuple, or None
        Default value used when value is None.

    min_value : int
        Minimum allowed value for each element.

    Returns
    -------
    tuple
        A tuple of two integers.

    Raises
    ------
    ValueError
        If the value cannot be converted to a valid positive integer pair.
    """
    if value is None:
        value = default

    if value is None:
        raise ValueError(f"{name} is None and no default is provided.")

    if isinstance(value, int):
        pair = (value, value)
    elif isinstance(value, (list, tuple)):
        if len(value) != 2:
            raise ValueError(
                f"{name} should have length 2, got {value}."
            )
        pair = (value[0], value[1])
    else:
        raise ValueError(
            f"{name} should be an int, list, or tuple, got {type(value)}."
        )

    try:
        pair = (int(pair[0]), int(pair[1]))
    except (TypeError, ValueError):
        raise ValueError(
            f"{name} should contain integer-like values, got {value}."
        )

    if pair[0] < min_value or pair[1] < min_value:
        raise ValueError(
            f"{name} values should be >= {min_value}, got {pair}."
        )

    return pair


def normalize_grid_size(grid_size=None):
    """
    Normalize grid size for grid packetization.

    Parameters
    ----------
    grid_size : int, list, tuple, or None
        Grid size. If None, use DEFAULT_GRID_SIZE.

    Returns
    -------
    tuple
        (grid_h, grid_w)
    """
    return normalize_pair(
        grid_size,
        name="grid_size",
        default=DEFAULT_GRID_SIZE,
        min_value=1,
    )


def normalize_block_size(block_size):
    """
    Normalize block size for block packetization.

    Parameters
    ----------
    block_size : int, list, or tuple
        Spatial block size.

    Returns
    -------
    tuple
        (block_h, block_w)
    """
    return normalize_pair(
        block_size,
        name="block_size",
        default=None,
        min_value=1,
    )


def infer_num_packets_from_grid(grid_size=None):
    """
    Infer packet count from a grid size.

    Parameters
    ----------
    grid_size : int, list, tuple, or None
        Grid size.

    Returns
    -------
    int
        Number of spatial packets.
    """
    grid_h, grid_w = normalize_grid_size(grid_size)
    return int(grid_h * grid_w)


def loss_mask_to_receive_mask(loss_mask):
    """
    Convert a loss mask to a receive mask.

    Convention:
        loss_mask[i] == True means packet i is lost.
        receive_mask[i] == True means packet i is received.

    Parameters
    ----------
    loss_mask : tensor-like or bool-like array
        Packet loss mask.

    Returns
    -------
    object
        Inverted mask. For torch tensors / numpy arrays, this uses ~loss_mask.
    """
    return ~loss_mask


def receive_mask_to_loss_mask(receive_mask):
    """
    Convert a receive mask to a loss mask.

    Convention:
        receive_mask[i] == True means packet i is received.
        loss_mask[i] == True means packet i is lost.

    Parameters
    ----------
    receive_mask : tensor-like or bool-like array
        Packet receive mask.

    Returns
    -------
    object
        Inverted mask. For torch tensors / numpy arrays, this uses ~receive_mask.
    """
    return ~receive_mask

def normalize_patch_max_size(patch_max_size, default=64):
    try:
        value = int(patch_max_size if patch_max_size is not None else default)
    except (TypeError, ValueError):
        raise ValueError(
            f"patch_max_size should be convertible to int, got {patch_max_size}."
        )
    if value <= 0:
        raise ValueError(f"patch_max_size should be positive, got {value}.")
    return value


__all__ = [
    "PACKET_MODE_GRID",
    "PACKET_MODE_BLOCK",
    "VALID_PACKET_MODES",
    "DEFAULT_PACKET_MODE",
    "DEFAULT_GRID_SIZE",
    "LOSS_MASK_TRUE_MEANS_LOST",
    "normalize_packet_mode",
    "is_valid_packet_mode",
    "normalize_pair",
    "normalize_grid_size",
    "normalize_block_size",
    "infer_num_packets_from_grid",
    "loss_mask_to_receive_mask",
    "receive_mask_to_loss_mask",
    "PACKET_MODE_PATCH_MAX_SIZE",
]