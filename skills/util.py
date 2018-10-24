import numpy as np
import gym
from typing import Callable


def softmax(X, theta=1.0, axis=None):
    """
    Compute the softmax of each element along an axis of X.

    Parameters
    ----------
    X: ND-Array. Probably should be floats.
    theta (optional): float parameter, used as a multiplier
        prior to exponentiation. Default = 1.0
    axis (optional): axis to compute values along. Default is the
        first non-singleton axis.

    Returns an array the same size as X. The result will sum to 1
    along the specified axis.
    """

    X = np.array(X)

    # make X at least 2d
    y = np.atleast_2d(X)

    # find axis
    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)

    # multiply y against the theta parameter,
    y = y * float(theta)

    # subtract the max for numerical stability
    y = y - np.max(y, axis=axis, keepdims=True)

    # exponentiate y
    y = np.exp(y)

    # take the sum along the specified axis
    ax_sum = np.sum(y, axis=axis, keepdims=True)

    # finally: divide elementwise
    p = y / ax_sum

    # flatten if X was 1D
    if len(X.shape) == 1:
        p = p.flatten()

    return p


def get_wrapped_attr(env: gym.Env, attr: str):
    return getattr(unwrap_env(env, lambda e: hasattr(e, attr)), attr)


def call_wrapped_method(env: gym.Env, method: str, kwargs: dict):
    return get_wrapped_attr(env=env, attr=method)(**kwargs)


def unwrap_env(env: gym.Env, condition: Callable[[gym.Env], bool]):
    while not condition(env):
        try:
            env = env.env
        except AttributeError:
            raise RuntimeError(
                f"env {env} has no children that meet condition.")
    return env
