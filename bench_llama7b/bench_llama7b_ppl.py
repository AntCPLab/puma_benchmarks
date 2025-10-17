# Copyright 2023 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import argparse
import json
from contextlib import contextmanager
from typing import Any, Optional, Tuple, Union

import flax.linen as nn
import jax
import jax.nn as jnn
import jax.numpy as jnp
import pdb 
import datasets
import evaluate
import torch
from torch.utils.data import DataLoader
import json
from EasyLM.models.llama.llama_model import FlaxLLaMAForCausalLM, LLaMAConfig
from flax.linen.linear import Array
from transformers import LlamaTokenizer

import spu.intrinsic as intrinsic
import spu.spu_pb2 as spu_pb2
import spu.utils.distributed as ppd

parser = argparse.ArgumentParser(description='distributed driver.')
parser.add_argument("-c", "--config", default="examples/python/griffin/3pc.json")

args = parser.parse_args()

with open(args.config, 'r') as file:
    conf = json.load(file)

ppd.init(conf["nodes"], conf["devices"])

copts = spu_pb2.CompilerOptions()
copts.enable_pretty_print = False
copts.xla_pp_kind = 2
# enable x / broadcast(y) -> x * broadcast(1/y)
copts.enable_optimize_denominator_with_broadcast = True

dataset = datasets.load_dataset('wikitext', 'wikitext-103-v1', split='validation')

text = [text for text in dataset['text'] if text is not None and text != "" and len(text.split()) >= 20]

model_path = "path-to-llama7b"

tokenizer = LlamaTokenizer.from_pretrained(model_path)
tokenizer.pad_token_id = tokenizer.eos_token_id
config = LLaMAConfig()
pretrained_model = FlaxLLaMAForCausalLM.from_pretrained(model_path, config=config)

batch_num = 50

def hack_softmax(
    x: Array,
    axis: Optional[Union[int, Tuple[int, ...]]] = -1,
    where: Optional[Array] = None,
    initial: Optional[Array] = None,
) -> Array:
    x_max = jnp.max(x, axis, where=where, initial=initial, keepdims=True)
    x = x - x_max
    # exp on large negative is clipped to zero
    b = x > -14
    nexp = jnp.exp(x) * b
    divisor = jnp.sum(nexp, axis, where=where, keepdims=True)
    return nexp / divisor


@contextmanager
def hack_softmax_context(msg: str, enabled: bool = True):
    if not enabled:
        yield
        return
    # hijack some target functions
    raw_softmax = jnn.softmax
    jnn.softmax = hack_softmax
    yield
    # recover back
    jnn.softmax = raw_softmax


def hack_silu(x: Array) -> Array:
    b0 = x < -8.0
    b1 = x < -4.0
    b2 = x > 4.0
    b3 = b1 ^ b2 ^ True  # x in [-4.0, 4.0)
    b4 = b0 ^ b1  # x in [-8.0, -4.0)
    # seg1 =  a[2] * x^2 + a[1] * x + a[0]
    # seg2 = b[6] * x^6 + b[4] * x^4 + b[2] * x^2 + b[0]
    a_coeffs = jnp.array(
        [-0.3067541139982155, -0.0819767021525476, -0.0055465625580307]
    )
    b_coeffs = jnp.array(
        [
            0.0085064025895951,
            0.5,
            0.2281430841728270,
            -0.011113046708173,
            0.0002743776353465,
        ]
    )
    x2 = jnp.square(x)
    x4 = jnp.square(x2)
    x6 = x2 * x4
    seg1 = a_coeffs[2] * x2 + a_coeffs[1] * x + a_coeffs[0]
    seg2 = (
        b_coeffs[4] * x6
        + b_coeffs[3] * x4
        + b_coeffs[2] * x2
        + b_coeffs[1] * x
        + b_coeffs[0]
    )
    ret = b2 * x + b4 * seg1 + b3 * seg2
    return ret


@contextmanager
def hack_silu_context(msg: str, enabled: bool = True):
    if not enabled:
        yield
        return
    # hijack some target functions
    raw_silu = nn.silu
    nn.silu = hack_silu
    yield
    # recover back
    nn.silu = raw_silu


def calculate_logits(params, input_ids):
    config = LLaMAConfig()
    model = FlaxLLaMAForCausalLM(config=config)
    outputs = model(input_ids=input_ids, params=params)
    logits = outputs.logits[:, :-1, :]

    return logits  


def eval_cpu_perp():
    total_loss = 0
    total_count = 0
    for i in range(batch_num):
        ids = tokenizer.encode(text[i], return_tensors = 'jax', truncation=True, padding=True, max_length=64)
        labels = ids[:, 1:]
        logits = calculate_logits(pretrained_model.params, ids)
        loss = jnp.sum(jax.nn.log_softmax(logits, axis=-1) * jax.nn.one_hot(labels, logits.shape[-1]), axis=-1)
        count = jnp.sum(labels != tokenizer.pad_token_id) # Use the padding token ID
        total_loss += jnp.sum(loss)
        total_count += count

    perplexity = jnp.exp(- total_loss / total_count)
    return perplexity.item()

def eval_griffin_perp():
    total_loss = 0
    total_count = 0

    params = ppd.device("P2")(lambda x: x)(pretrained_model.params)

    with hack_softmax_context("hijack jax softmax", enabled = True), hack_silu_context("hijack jax silu", enabled=True):
        
        for j in range(batch_num):
            ids = tokenizer.encode(text[j], return_tensors = 'jax', truncation=True, padding=True, max_length=64)
            labels = ids[:, 1:]
            input_ids = ppd.device("P1")(lambda x: x)(ids)
            logits = ppd.device("SPU")(calculate_logits, copts=copts)(params, input_ids)
            logits = ppd.get(logits)
            loss = jnp.sum(jax.nn.log_softmax(logits, axis=-1) * jax.nn.one_hot(labels, logits.shape[-1]), axis=-1)
            count = jnp.sum(labels != tokenizer.pad_token_id) # Use the padding token ID
            total_loss += jnp.sum(loss)
            total_count += count

    perplexity = jnp.exp(- total_loss / total_count)
    return perplexity.item()

if __name__ == '__main__':
    print('\n------\nRun on CPU')
    outs_cpu = eval_cpu_perp()
    print("Perplexity-cpu:", outs_cpu)
    
    print('\n------\nRun on Griffin')
    outs_griffin = eval_griffin_perp()
    print("Perplexity-Griffin:", outs_griffin)
