/*
 * SPDX-FileCopyrightText: Copyright (c) 2019 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <stdint.h>
#include <stdio.h>
//#include <cuda_runtime.h>

#include "utils/utils.h"
#include "utils/channel.hpp"

#define MAX_OPCODES 180

extern "C" __device__ __forceinline__ void lock(int *mutex)
{
	while (atomicExch(mutex, 1) == 1)
	{

	}
}

extern "C" __device__ __forceinline__ void unlock(int *mutex)
{
	atomicExch(mutex, 0);
}

extern "C" __device__ __noinline__ void count_instrs(int predicate,
													 int instr_type,
													 int count_warp_level,
													 uint32_t window_size,
													 uint64_t p_hist,
													 uint64_t p_instr_count,
													 uint64_t p_mtx,
													 uint64_t p_channel)
{
	const int active_mask = __ballot_sync(__activemask(), 1);
	const int predicate_mask = __ballot_sync(__activemask(), predicate);
	const int laneid = get_laneid();
	const int first_laneid = __ffs(active_mask) - 1;
	const int num_threads = __popc(predicate_mask);
	
	if (first_laneid == laneid)
	{
		
		uint32_t *hist = (uint32_t *)p_hist;
		uint32_t *instr_count = (uint32_t *)p_instr_count;
		int *mtx = (int *)p_mtx;

		uint32_t local_hist[MAX_OPCODES];
		bool writeback = false;

		if (num_threads > 0)
		{	

			lock(mtx);
			__threadfence(); 

			if (count_warp_level)
			{
				hist[instr_type]++;
				*instr_count += 1;
			}
			else
			{
				hist[instr_type] += num_threads;
				*instr_count += num_threads;
			}

			if (*instr_count >= window_size)
			{
				//printf("\nWindow full event, instr count = %u\n", *instr_count);
				writeback = true;
				*instr_count = 0; 

				for (int i = 0; i < MAX_OPCODES; i++)
				{
					local_hist[i] = hist[i];
					hist[i] = 0;
				}
			}

			__threadfence(); 
			unlock(mtx);
		}

		if (writeback)
		{
			//printf("\nChannel push event, instr count = %u\n", *instr_count);
			
			ChannelDev *channel = (ChannelDev*)p_channel;
			channel->push(local_hist, sizeof(local_hist));
		}
	}
}