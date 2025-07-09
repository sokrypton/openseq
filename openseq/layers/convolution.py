import jax
import jax.numpy as jnp
from ..utils.random import get_random_key # Assuming get_random_key will be in openseq.utils.random

# Based on Conv1D_custom from network_functions.py and laxy.Conv1D

def conv1d_init_params(in_dims: int, out_dims: int, window_size: int, use_bias: bool = True, key=None, seed: int = None):
    """
    Initialize parameters for a 1D convolutional layer.

    Args:
        in_dims (int): Number of input channels/dimensions.
        out_dims (int): Number of output channels/filters.
        window_size (int): Size of the convolution window.
        use_bias (bool): Whether to include a bias term. Defaults to True.
        key (jax.random.PRNGKey, optional): JAX PRNG key. If None, a new key is generated using seed.
        seed (int, optional): Seed for key generation if key is None.

    Returns:
        dict: A dictionary containing initialized parameters ('w' for weights, 'b' for bias if use_bias).
    """
    if key is None:
        key = get_random_key(seed)

    key_w, key_b = jax.random.split(key)

    # Glorot/Xavier normal initialization for weights
    # Shape for Conv1D weights in JAX: (out_channels, in_channels, window_size)
    # Or (window_size, in_channels, out_channels) if using lax.conv_general_dilated with ('OIH')
    # jax.lax.conv expects kernel shape (output_channels, input_channels, kernel_spatial_dims...)
    # So, (out_dims, in_dims, window_size)

    # laxy.Conv1D used: params = {"w":jax.nn.initializers.glorot_normal()(key,(out_dims,in_dims,win))}
    # network_functions.Conv1D_custom used: {"w":jnp.zeros((out_dims,in_dims,win))} (and added noise later)
    # Let's use Glorot normal as it's a common good default.

    stddev = jnp.sqrt(2.0 / ((in_dims + out_dims) * window_size)) # Approximation for Glorot normal
    weights = jax.random.normal(key_w, (out_dims, in_dims, window_size)) * stddev

    params = {"w": weights}
    if use_bias:
        params["b"] = jnp.zeros(out_dims)
    return params

def conv1d_apply(params: dict, x: jnp.ndarray, stride: int = 1, padding: str = "SAME",
                 add_noise_key=None, noise_scale: float = 0.1):
    """
    Apply a 1D convolutional layer.

    Args:
        params (dict): Dictionary of layer parameters ('w', optional 'b').
        x (jnp.ndarray): Input data of shape (batch_size, length, in_dims).
        stride (int): Convolution stride. Defaults to 1.
        padding (str): Padding mode ("SAME" or "VALID"). Defaults to "SAME".
        add_noise_key (jax.random.PRNGKey, optional): Key for adding noise to weights during application (from Conv1D_custom).
                                                     Defaults to None (no noise).
        noise_scale (float): Scale of the noise to add if add_noise_key is provided.

    Returns:
        jnp.ndarray: Output of the convolution, shape (batch_size, new_length, out_dims).
    """
    weights = params["w"]
    if add_noise_key is not None:
        weights += noise_scale * jax.random.normal(add_noise_key, shape=weights.shape)

    # JAX conv expects input (N, H, W, C) for 2D or (N, W, C) for 1D, where W is spatial dim.
    # Our input is (N, L, C_in). We need to transpose it to (N, C_in, L) for some conv ops,
    # or specify dimension numbers for lax.conv_general_dilated.
    # The original laxy.Conv1D and network_functions.Conv1D_custom both did:
    # x = x.transpose([0,2,1]) # (N, C_in, L)
    # y = jax.lax.conv(x, weights, (stride,), padding=padding) # weights (C_out, C_in, W)
    # y = y.transpose([0,2,1]) # (N, L_out, C_out)

    x_transposed = x.transpose([0, 2, 1]) # (batch_size, in_dims, length)

    # jax.lax.conv dimension numbers: (batch_dim, feature_dim, spatial_dims...)
    # For input (N, C_in, L), this is (0, 1, 2)
    # For kernel (C_out, C_in, W), this is (0, 1, 2) (output_channels, input_channels, kernel_width)
    # For output (N, C_out, L_out), this is (0, 1, 2)
    # These are defaults for lax.conv if input_feature_dimension=1, kernel_feature_dimension=1.

    # Let's use lax.conv_general_dilated for more explicit control if needed,
    # but lax.conv should work if dimensions are standard.
    # lax.conv's `lhs` is input, `rhs` is kernel.
    # `window_strides` is a sequence, so `(stride,)` for 1D.
    # `padding` is "SAME" or "VALID".

    y_transposed = jax.lax.conv(
        lhs=x_transposed,      # (N, C_in, L)
        rhs=weights,           # (C_out, C_in, W_kernel)
        window_strides=(stride,),
        padding=padding,
        dimension_numbers=('NCW', 'OIW', 'NCW') # NCHW-like: (batch, channel, width)
                                                # OIH_W-like: (out_channel, in_channel, width_kernel)
    )

    y = y_transposed.transpose([0, 2, 1]) # (batch_size, length_out, out_dims)

    if "b" in params:
        y += params["b"] # Bias is (out_dims), broadcasts to (N, L_out, C_out)
    return y

# Placeholder for Conv2D if needed, from laxy.py
# def conv2d_init_params(...)
# def conv2d_apply(...)

# Consider if other layers from laxy (Dense, GRU, LSTM) or network_functions
# need to be ported here or if they are too specific to SMURF/Gremlin logic
# and better handled within those model files or by external libraries.
# For now, only Conv1D as it was used by SMURF (network_functions.MRF) and laxy.
```
