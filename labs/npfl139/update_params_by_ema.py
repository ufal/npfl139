# This file is part of NPFL139 <http://github.com/ufal/npfl139/>.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
import os
import random

import numpy as np
import torch

def update_params_by_ema(target: torch.nn.Module, source: torch.nn.Module, tau: float) -> None:
    """Update target parameters using exponential moving average of the source parameters.

    Parameters:
      target: The target model whose parameters will be updated.
      source: The source model whose parameters will be used for the update.
      tau: The decay factor for the exponential moving average, e.g., 0.001.
    """
    with torch.no_grad():
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.mul_(1 - tau)
            target_param.add_(tau * source_param)
