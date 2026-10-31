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

#include <assert.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <map>
#include <unordered_map>
#include <unordered_set>

/* every tool needs to include this once */
#include "nvbit_tool.h"

/* nvbit interface file */
#include "nvbit.h"

/* provide some __device__ functions */
#include "utils/utils.h"
#include "utils/channel.hpp"



/* total instruction counter, maintained in system memory, incremented by
 * "counter" every time a kernel completes  */
uint64_t tot_app_instrs = 0;

volatile bool injection_finished = false;

#define CHANNEL_SIZE (1L << 20)

enum class RecvThreadState {
    WORKING,
    STOP,
    FINISHED,
};

struct CTXstate {
    /* context id */
    int id;

    /* Channel used to communicate from GPU to CPU receiving thread */
    ChannelDev* channel_dev;
    ChannelHost channel_host;

    // After initialization, set it to WORKING to make recv thread get data,
    // parent thread sets it to STOP to make recv thread stop working.
    // recv thread sets it to FINISHED when it cleans up.
    // parent thread should wait until the state becomes FINISHED to clean up.
    volatile RecvThreadState recv_thread_done = RecvThreadState::STOP;
};

/* map to store context state */
std::unordered_map<CUcontext, CTXstate*> ctx_state_map;

/* a pthread mutex, used to prevent multiple kernels to run concurrently and
 * therefore to "corrupt" the counter variable */
pthread_mutex_t mutex;
pthread_mutex_t cuda_event_mutex;

/* skip flag used to avoid re-entry on the nvbit_callback when issuing
 * flush_channel kernel call */
bool skip_callback_flag = false;

/* kernel id counter, maintained in system memory */
uint32_t kernel_id = 0;

/* kernel instruction counter, updated by the GPU threads */
#define MAX_OPCODES 180
uint32_t print_hist[MAX_OPCODES];
__managed__ uint32_t histogram[MAX_OPCODES];
__managed__ uint32_t instr_count;
__managed__ int mtx;

uint64_t counter = 0;

/* global control variables for this tool */
uint32_t instr_begin_interval = 0;
uint32_t instr_end_interval = UINT32_MAX;
uint32_t ker_begin_interval = 0;
uint32_t ker_end_interval = UINT32_MAX;
int verbose = 0;
int count_warp_level = 0;
int exclude_pred_off = 0;
uint32_t window_size = 500;

/* instruction to opcode map, used for final print of the opcodes */
std::map<std::string, int> instr_opcode_to_num_map;

void *recv_thread_fun(void* args);

/* nvbit_at_init() is executed as soon as the nvbit tool is loaded. We typically
 * do initializations in this call. In this case for instance we get some
 * environment variables values which we use as input arguments to the tool */
void nvbit_at_init()
{
    /* just make sure all managed variables are allocated on GPU */
    setenv("CUDA_MANAGED_FORCE_DEVICE_ALLOC", "1", 1);

    /* we get some environment variables that are going to be use to selectively
     * instrument (within a interval of kernel indexes and instructions). By
     * default we instrument everything. */
    GET_VAR_INT(
        instr_begin_interval, "INSTR_BEGIN", 0,
        "Beginning of the instruction interval where to apply instrumentation");
    GET_VAR_INT(
        instr_end_interval, "INSTR_END", UINT32_MAX,
        "End of the instruction interval where to apply instrumentation");
    GET_VAR_INT(ker_begin_interval, "KERNEL_BEGIN", 0,
                "Beginning of the kernel launch interval where to apply "
                "instrumentation");
    GET_VAR_INT(
        ker_end_interval, "KERNEL_END", UINT32_MAX,
        "End of the kernel launch interval where to apply instrumentation");
    GET_VAR_INT(verbose, "TOOL_VERBOSE", 0, "Enable verbosity inside the tool");
    GET_VAR_INT(count_warp_level, "COUNT_WARP_LEVEL", 0,
                "Count warp level or thread level instructions");
    GET_VAR_INT(exclude_pred_off, "EXCLUDE_PRED_OFF", 0,
                "Exclude predicated off instruction from count");

    GET_VAR_INT(window_size, "WINDOW_SIZE", 500, 
                "Number of instructions per channel push window");

    std::string pad(100, '-');
    printf("%s\n", pad.c_str());

    /* set mutex as recursive */
    pthread_mutexattr_t attr;
    pthread_mutexattr_init(&attr);
    pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_RECURSIVE);
    pthread_mutex_init(&mutex, &attr);

    pthread_mutex_init(&cuda_event_mutex, &attr);
}

/* Set used to avoid re-instrumenting the same functions multiple times */
std::unordered_set<CUfunction> already_instrumented;

void instrument_function_if_needed(CUcontext ctx, CUfunction func)
{
    assert(ctx_state_map.find(ctx) != ctx_state_map.end());
    CTXstate* ctx_state = ctx_state_map[ctx];

    /* Get related functions of the kernel (device function that can be
     * called by the kernel) */
    std::vector<CUfunction> related_functions =
        nvbit_get_related_functions(ctx, func);

    /* add kernel itself to the related function vector */
    related_functions.push_back(func);

    /* iterate on function */
    for (auto f : related_functions)
    {
        /* "recording" function was instrumented, if set insertion failed
         * we have already encountered this function */
        if (!already_instrumented.insert(f).second)
        {
            continue;
        }

        /* Get the vector of instruction composing the loaded CUFunction "func"
         */
        const std::vector<Instr *> &instrs = nvbit_get_instrs(ctx, func);

        /* Get kernel name */
        std::string kernel_name = nvbit_get_func_name(ctx, func);

        /* If verbose we print function name and number of" static" instructions
         */
        if (verbose)
        {
            printf("inspecting %s - num instrs %ld\n", kernel_name.c_str(),
                   instrs.size());
        }

        /* We iterate on the vector of instruction */
        for (auto i : instrs)
        {
            /* Check if the instruction falls in the interval where we want to
             * instrument */
            if (i->getIdx() < instr_begin_interval ||
                i->getIdx() >= instr_end_interval)
            {
                continue;
            }
            /* If verbose we print which instruction we are instrumenting */
            if (verbose)
            {
                i->print();
            }

            std::string opcode = i->getOpcodeShort();
            if (instr_opcode_to_num_map.find(opcode) ==
                instr_opcode_to_num_map.end())
            {
                size_t size = instr_opcode_to_num_map.size();
                instr_opcode_to_num_map[opcode] = size;
            }
            int instr_type = instr_opcode_to_num_map[opcode];

            /* Insert a call to "count_instrs" before the instruction "i" */
            nvbit_insert_call(i, "count_instrs", IPOINT_BEFORE);
            /* Add argument to the instrumentation function */
            if (exclude_pred_off)
            {
                /* pass predicate value */
                nvbit_add_call_arg_guard_pred_val(i);
            }
            else
            {
                /* pass always true */
                nvbit_add_call_arg_const_val32(i, 1);
            }

            /* add instruction type id */
            nvbit_add_call_arg_const_val32(i, instr_type);
            /* add count warps option */
            nvbit_add_call_arg_const_val32(i, count_warp_level);
            nvbit_add_call_arg_const_val32(i, window_size);
            /* add pointer to counter location */
            nvbit_add_call_arg_const_val64(i, (uint64_t)histogram);
            nvbit_add_call_arg_const_val64(i, (uint64_t)&instr_count);
            nvbit_add_call_arg_const_val64(i, (uint64_t)&mtx);
            nvbit_add_call_arg_const_val64(i, (uint64_t)ctx_state->channel_dev);
        }
    }
}

/* flush channel */
__global__ void flush_channel(ChannelDev* ch_dev) { ch_dev->flush(); }

void init_context_state(CUcontext ctx) {
    CTXstate* ctx_state = ctx_state_map[ctx];
    ctx_state->recv_thread_done = RecvThreadState::WORKING;
    cudaMallocManaged(&ctx_state->channel_dev, sizeof(ChannelDev));
    ctx_state->channel_host.init((int)ctx_state_map.size() - 1, CHANNEL_SIZE,
                                 ctx_state->channel_dev, recv_thread_fun, ctx);
    nvbit_set_tool_pthread(ctx_state->channel_host.get_thread());
}

void *recv_thread_fun(void *args)
{
    printf("\nEnter recv\n");
    CUcontext ctx = (CUcontext)args;

    pthread_mutex_lock(&mutex);
    /* get context state from map */
    assert(ctx_state_map.find(ctx) != ctx_state_map.end());
    CTXstate* ctx_state = ctx_state_map[ctx];

    ChannelHost* ch_host = &ctx_state->channel_host;
    pthread_mutex_unlock(&mutex);
    char* recv_buffer = (char*)malloc(CHANNEL_SIZE);

    FILE *csv_file = fopen("opcode_hist_output.csv", "w");

    if (csv_file == NULL) {
        printf("Error: Could not open opcode_hist_output.csv for writing.\n");
        ctx_state->recv_thread_done = RecvThreadState::FINISHED;
        free(recv_buffer);
        return nullptr;
    }

    while (!injection_finished && ctx_state->recv_thread_done == RecvThreadState::WORKING) {
        sched_yield(); 
    }

    for (auto a : instr_opcode_to_num_map) {

        if (csv_file) {
            fprintf(csv_file, "%s,", a.first.c_str());
        }
                   
    }

    if (csv_file) {
        fprintf(csv_file, "\n");
    }


    while (ctx_state->recv_thread_done == RecvThreadState::WORKING) {
        //printf("\nEnter recv/working\n");
        /* receive buffer from channel */
        uint32_t num_recv_bytes = ch_host->recv(recv_buffer, CHANNEL_SIZE);
        if (num_recv_bytes > 0) {
            //printf("\nEnter recv/working/numbytes, bytes = %u\n", num_recv_bytes);
            uint32_t num_processed_bytes = 0;
            
            while (num_processed_bytes < num_recv_bytes) {
               // printf("\nEnter recv/working/numbytes/processed\n");
                if (num_recv_bytes - num_processed_bytes < sizeof(histogram)) {
                    printf("\nChannel error: Expected bytes = %lu, received = %u\n", sizeof(histogram), num_recv_bytes);
                    ctx_state->recv_thread_done = RecvThreadState::FINISHED;
                    free(recv_buffer);
                    if (csv_file) fclose(csv_file);
                    return nullptr;
                }
                uint32_t* local_hist = (uint32_t*)&recv_buffer[num_processed_bytes];
        
                for (auto a : instr_opcode_to_num_map)
                {

                    //if (local_hist[a.second] != 0)
                    //{
                    counter += local_hist[a.second];
                    print_hist[a.second] += local_hist[a.second];
                     // }
                   if (csv_file) {
                        fprintf(csv_file, "%u,", local_hist[a.second]);
                    }
                }
                if (csv_file) {
                    fprintf(csv_file, "\n");
                    // Optional: flush the buffer so you can see live updates in the file
                    fflush(csv_file); 
                }
                num_processed_bytes += sizeof(histogram);
            }
        }
    }
    free(recv_buffer);
    if (csv_file) {
        fclose(csv_file);
    }
    ctx_state->recv_thread_done = RecvThreadState::FINISHED;
    return nullptr;
}

/* This call-back is triggered every time a CUDA event is encountered.
 * Here, we identify CUDA kernel launch events and reset the "counter" before
 * th kernel is launched, and print the counter after the kernel has completed
 * (we make sure it has completed by using cudaDeviceSynchronize()). To
 * selectively run either the original or instrumented kernel we used
 * nvbit_enable_instrumented() before launching the kernel. */
void nvbit_at_cuda_event(CUcontext ctx, int is_exit, nvbit_api_cuda_t cbid,
                         const char *name, void *params, CUresult *pStatus)
{
    pthread_mutex_lock(&cuda_event_mutex);
    
    if (skip_callback_flag) {
        pthread_mutex_unlock(&cuda_event_mutex);
        return;
    }

    skip_callback_flag = true;
    
    CTXstate* ctx_state = ctx_state_map[ctx];

    /* Identify all the possible CUDA launch events */
    if (cbid == API_CUDA_cuLaunch || cbid == API_CUDA_cuLaunchKernel_ptsz ||
        cbid == API_CUDA_cuLaunchGrid || cbid == API_CUDA_cuLaunchGridAsync ||
        cbid == API_CUDA_cuLaunchKernel ||
        cbid == API_CUDA_cuLaunchKernelEx ||
        cbid == API_CUDA_cuLaunchKernelEx_ptsz)
    {
        /* cast params to launch parameter based on cbid since if we are here
         * we know these are the right parameters types */
        CUfunction func;
        if (cbid == API_CUDA_cuLaunchKernelEx_ptsz ||
            cbid == API_CUDA_cuLaunchKernelEx)
        {
            cuLaunchKernelEx_params *p = (cuLaunchKernelEx_params *)params;
            func = p->f;
        }
        else
        {
            cuLaunchKernel_params *p = (cuLaunchKernel_params *)params;
            func = p->f;
        }

        if (!is_exit)
        {
            printf("\nEntered");
            CUDA_SAFECALL(cudaDeviceSynchronize());
            assert(cudaGetLastError() == cudaSuccess);
            /* if we are entering in a kernel launch:
             * 1. Lock the mutex to prevent multiple kernels to run concurrently
             * (overriding the counter) in case the user application does that
             * 2. Instrument the function if needed
             * 3. Select if we want to run the instrumented or original
             * version of the kernel
             * 4. Reset the kernel instruction counter */

            // pthread_mutex_lock(&mutex);

            counter = 0;

            instrument_function_if_needed(ctx, func);

            injection_finished = true;

            if (kernel_id >= ker_begin_interval &&
                kernel_id < ker_end_interval)
            {
                nvbit_enable_instrumented(ctx, func, true);
            }
            else
            {
                nvbit_enable_instrumented(ctx, func, false);
            }
            memset(histogram, 0, sizeof(uint32_t) * MAX_OPCODES);
            memset(print_hist, 0, sizeof(uint32_t) * MAX_OPCODES);
            instr_count = 0;
        }
        else
        {
            printf("\nExited\n");
            /* if we are exiting a kernel launch:
             * 1. Wait until the kernel is completed using
             * cudaDeviceSynchronize()
             * 2. Get number of thread blocks in the kernel
             * 3. Print the thread instruction counters
             * 4. Release the lock*/
            CUDA_SAFECALL(cudaDeviceSynchronize());
            flush_channel<<<1, 1>>>(ctx_state->channel_dev);
            CUDA_SAFECALL(cudaDeviceSynchronize());

            assert(cudaGetLastError() == cudaSuccess);

            /*int num_ctas = 0;
            if (cbid == API_CUDA_cuLaunchKernel_ptsz ||
                cbid == API_CUDA_cuLaunchKernel)
            {
                cuLaunchKernel_params *p2 = (cuLaunchKernel_params *)params;
                num_ctas = p2->gridDimX * p2->gridDimY * p2->gridDimZ;
            }
            else if (cbid == API_CUDA_cuLaunchKernelEx_ptsz ||
                     cbid == API_CUDA_cuLaunchKernelEx)
            {
                cuLaunchKernelEx_params *p2 = (cuLaunchKernelEx_params *)params;
                num_ctas = p2->config->gridDimX * p2->config->gridDimY *
                           p2->config->gridDimZ;
            }

            for (auto a : instr_opcode_to_num_map) {
                //if (print_hist[a.second] != 0) {
            printf("  %s = %u, opcode = %i\n", a.first.c_str(),
                            print_hist[a.second], a.second);
                //}
            }
            

            tot_app_instrs += counter;
            printf(
                "kernel %d - %s - #thread-blocks %d,  kernel "
                "instructions %ld, total instructions %ld\n",
                kernel_id++, nvbit_get_func_name(ctx, func), num_ctas, counter,
                tot_app_instrs);

                */
            // pthread_mutex_unlock(&mutex);
        }
    }

    skip_callback_flag = false;
    pthread_mutex_unlock(&cuda_event_mutex);
}

void nvbit_at_ctx_init(CUcontext ctx) {
    pthread_mutex_lock(&mutex);
    if (verbose) {
        //printf("MEMREAD_SHARED: STARTING CONTEXT %p\n", ctx);
    }
    assert(ctx_state_map.find(ctx) == ctx_state_map.end());
    CTXstate* ctx_state = new CTXstate;
    ctx_state_map[ctx] = ctx_state;
    pthread_mutex_unlock(&mutex);
}

void nvbit_tool_init(CUcontext ctx) {
    pthread_mutex_lock(&mutex);
    assert(ctx_state_map.find(ctx) != ctx_state_map.end());
    init_context_state(ctx);
    pthread_mutex_unlock(&mutex);
}

void nvbit_at_ctx_term(CUcontext ctx) {
    pthread_mutex_lock(&mutex);
    skip_callback_flag = true;
    if (verbose) {
        //printf("MEMREAD_SHARED: TERMINATING CONTEXT %p\n", ctx);
    }
    /* get context state from map */
    assert(ctx_state_map.find(ctx) != ctx_state_map.end());
    CTXstate* ctx_state = ctx_state_map[ctx];

    /* Notify receiver thread and wait for receiver thread to
     * notify back */
    ctx_state->recv_thread_done = RecvThreadState::STOP;
    while (ctx_state->recv_thread_done != RecvThreadState::FINISHED);

    for (auto a : instr_opcode_to_num_map) {
        //if (print_hist[a.second] != 0) {
    printf("  %s = %u, opcode = %i\n", a.first.c_str(),
                    print_hist[a.second], a.second);
        //}
    }
    
    tot_app_instrs += counter;
    printf("kernel %d - kernel instructions %ld, total instructions %ld\n",
        kernel_id++, counter, tot_app_instrs);

    ctx_state->channel_host.destroy(false);
    cudaFree(ctx_state->channel_dev);
    skip_callback_flag = false;
    delete ctx_state;
    pthread_mutex_unlock(&mutex);
}

