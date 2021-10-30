/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.  All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without modification, are permitted
 * provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright notice, this list of
 *       conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright notice, this list of
 *       conditions and the following disclaimer in the documentation and/or other materials
 *       provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the names of its contributors may be used
 *       to endorse or promote products derived from this software without specific prior written
 *       permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TOR (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *//*
 */

/** @file   encoding.h
 *  @author Thomas Müller, NVIDIA
 *  @brief  API interface for input encodings
 */

#pragma once

#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/object.h>

#include <stdint.h>


TCNN_NAMESPACE_BEGIN

template <typename T>
class Encoding : public ParametricObject<T> {
public:
	virtual ~Encoding() { }

	virtual void encode(
		const uint32_t num_elements,
		const float* inputs,
		T* outputs,
		cudaStream_t stream,
		float* dy_dx = nullptr, // Gradient of output w.r.t. the generating input variable. Same size as the output portion that was encoded
		bool is_inference = false
	) const = 0;

	virtual void backward(
		cudaStream_t stream,
		const uint32_t num_elements,
		const T* dL_dy, // num_encoded_dims() x num_elements
		const float* dy_dx, // encoded output dims x num_elements
		float* dL_dx, // input dims x num_elements
		const float* inputs = nullptr
	) = 0;

	virtual uint32_t num_encoded_dims() const = 0;
	virtual uint32_t num_forward_gradient_dims() const = 0;

	// By default, an encoding has no parameters
	void initialize_params(std::mt19937& rnd, float* params_full_precision, T* params, T* inference_params, T* backward_params, T* gradients, float scale = 1) override { }
	size_t n_params() const override { return 0; }

	std::vector<std::pair<uint32_t, uint32_t>> layer_sizes() const override { return {}; }
};

template <typename T>
Encoding<T>* create_encoding(uint32_t n_dims_to_encode, uint32_t n_dims_to_pass_through, const json& params, uint32_t alignment = 8);

TCNN_NAMESPACE_END
