# Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.

# Redistribution and use in source and binary forms, with or without modification, are permitted
# provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright notice, this list of
#       conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright notice, this list of
#       conditions and the following disclaimer in the documentation and/or other materials
#       provided with the distribution.
#     * Neither the name of the NVIDIA CORPORATION nor the names of its contributors may be used
#       to endorse or promote products derived from this software without specific prior written
#       permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
# IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
# FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY, OR TOR (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import argparse
from xmlrpc.client import boolean
import commentjson as json
import numpy as np
import os
import sys
import torch
import torch.nn as nn
import time
import tqdm
import tinycudann as tcnn
from common import read_image, write_image, ROOT_DIR

class Image(torch.nn.Module):
	def __init__(self, filename, device):
		super(Image, self).__init__()
		self.data = read_image(filename)
		self.shape = self.data.shape
		self.data = torch.from_numpy(self.data).float().to(device)

	def forward(self, xs):
		with torch.no_grad():
			# Bilinearly filtered lookup from the image. Not super fast,
			# but less than ~20% of the overall runtime of this example.
			shape = self.shape

			xs = xs * torch.tensor([shape[1], shape[0]], device=xs.device).float()
			indices = xs.long()
			lerp_weights = xs - indices.float()

			x0 = indices[:, 0].clamp(min=0, max=shape[1]-1)
			y0 = indices[:, 1].clamp(min=0, max=shape[0]-1)
			x1 = (x0 + 1).clamp(max=shape[1]-1)
			y1 = (y0 + 1).clamp(max=shape[0]-1)

			return (
				self.data[y0, x0] * (1.0 - lerp_weights[:,0:1]) * (1.0 - lerp_weights[:,1:2]) +
				self.data[y0, x1] * lerp_weights[:,0:1] * (1.0 - lerp_weights[:,1:2]) +
				self.data[y1, x0] * (1.0 - lerp_weights[:,0:1]) * lerp_weights[:,1:2] +
				self.data[y1, x1] * lerp_weights[:,0:1] * lerp_weights[:,1:2]
			)
		
def get_args():
	parser = argparse.ArgumentParser(description="Image benchmark using PyTorch bindings.")

	parser.add_argument("--img_dir", nargs="?", default="data/images", help="Image to match")
	parser.add_argument("--cfg", nargs="?", default="data/config_hash.json", help="JSON config for tiny-cuda-nn")
	parser.add_argument("--n_steps", nargs="?", type=int, default=100000, help="Number of training steps")
	parser.add_argument("--log", nargs="?", action='store_true', help="record loss log")
	args = parser.parse_args()
	return args

# check device
device = torch.device("cuda")

# fetch args
args = get_args()

# get config
with open(args.config) as config_file:
    config = json.load(config_file)

# get images as training data


# build model = positional encorder + mlp
encoding = tcnn.Encoding(n_input_dims=2, encoding_config=config["encoding"])
network = tcnn.Network(n_input_dims=encoding.n_output_dims, n_output_dims=n_channels, network_config=config["network"])
model = nn.Sequential(encoding, network)

# Loss function and Optimizer
loss_func = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# training batch for random sampling
batch_size = 2**18
interval = 10

if __name__ == "__main__":
	for i in tqdm(range(args.n_steps), desc="train"):
		batch = torch.rand([batch_size, 2], device=device, dtype=torch.float32)
		targets = images(batch)
		predicts = model(batch)
		loss = loss_func(predicts, targets)

		optimizer.zero_grad()
		loss_func.backward()
		optimizer.step()