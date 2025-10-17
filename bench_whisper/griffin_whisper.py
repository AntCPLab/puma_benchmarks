# Copyright 2024 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import argparse
import json
import os
import pdb

from typing import Any, Optional, Tuple, Union
from contextlib import contextmanager
from flax.linen.linear import Array

import jax.numpy as jnp
import jax.nn as jnn
from transformers import WhisperProcessor, FlaxWhisperForConditionalGeneration
from datasets import load_dataset
from evaluate import load

import spu.utils.distributed as ppd

import spu.intrinsic as intrinsic
import spu.spu_pb2 as spu_pb2

parser = argparse.ArgumentParser(description='distributed driver.')
parser.add_argument(
    "-c", "--config", default="examples/python/conf/3pc.json"
)
args = parser.parse_args()

with open(args.config, 'r') as file:
    conf = json.load(file)

ppd.init(conf["nodes"], conf["devices"])

copts = spu_pb2.CompilerOptions()
copts.enable_pretty_print = False
copts.xla_pp_kind = 2
# enable x / broadcast(y) -> x * broadcast(1/y)
copts.enable_optimize_denominator_with_broadcast = True

def hack_softmax(
    x: Array,
    axis: Optional[Union[int, Tuple[int, ...]]] = -1,
    where: Optional[Array] = None,
    initial: Optional[Array] = None,
) -> Array:
    x_clip = jnp.max(x, axis, where=where, initial=initial, keepdims=True)
    x = x - x_clip
    b = x > -14
    nexp = jnp.exp(x) * b
    divisor = jnp.sum(nexp, axis, where=where, keepdims=True)
    return nexp / divisor


@contextmanager
def hack_softmax_context(msg: str, enabled: bool = True):
    if not enabled:
        yield
        return
    
    # hijack softmax functions using bench
    raw_softmax = jnn.softmax
    jnn.softmax = hack_softmax
    yield
    jnn.softmax = raw_softmax


def hack_gelu(x: Array,
            axis: Optional[Union[int, Tuple[int, ...]]] = -1,
            where: Optional[Array] = None,
            approximate: Optional[bool] = True,
            initial: Optional[Array] = None) -> Array:
    
    b0 = x < -4.0
    b1 = x < -1.95
    b2 = x > 3.0
    b3 = b1 ^ b2 ^ True # x in [-1.95, 3.0]
    b4 = b0 ^ b1 # x in [-4, -1.95] 

    # seg1 = a[3] * x^3 + a[2] * x^2 + a[1] * x + a[0]
    # seg2 = b[6] * x^6 + b[4] * x^4 + b[2] * x^2 + b[1] * x + b[0]
    a_coeffs = jnp.array([-0.5054031199708174, -0.42226581151983866, -0.11807612951181953, -0.011034134030615728])
    b_coeffs = jnp.array([0.008526321541038084,  0.5, 0.3603292692789629, 0.0, -0.037688200365904236, 0.0, 0.0018067462606141187])
    x2 = jnp.square(x)
    x3 = jnp.multiply(x, x2)
    x4 = jnp.square(x2)
    x6 = jnp.square(x3)

    seg1 = a_coeffs[3] * x3 + a_coeffs[2] * x2 + a_coeffs[1] * x + a_coeffs[0]
    seg2 = b_coeffs[6] * x6 + b_coeffs[4] * x4 + b_coeffs[2] * x2 + b_coeffs[1] * x + b_coeffs[0]

    ret = b2 * x + b4 * seg1 + b3 * seg2

    return ret

@contextmanager
def hack_gelu_context(msg: str, enabled: bool = False):
    if not enabled:
        yield
        return
    # hijack some target functions
    raw_gelu = jnn.gelu
    jnn.gelu = hack_gelu
    yield
    # recover back
    jnn.gelu = raw_gelu

processor = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")
pretrained_model = FlaxWhisperForConditionalGeneration.from_pretrained(
    "openai/whisper-tiny.en", from_pt=True
)

NUM = 100

wer = load("wer")

ds = load_dataset(
        "librispeech_asr", "clean", split="test"
    )


def text_generation(input_features, params):
    pretrained_model.params = params
    generated_ids = pretrained_model.generate(input_features=input_features, max_length=32)
    return generated_ids.sequences


def run_on_cpu():
    reference = []
    generate = []
    for i in range(NUM):
        inputs = processor(ds[739]["audio"]["array"], return_tensors="np")
        generated_ids = text_generation(inputs.input_features, pretrained_model.params)
        generate_text = processor.batch_decode(generated_ids)[0]
        generate.extend([processor.tokenizer._normalize(generate_text)])
        reference.extend([processor.tokenizer._normalize(ds[739]["text"])])
    
    return 100 * wer.compute(references=reference, predictions=generate)

def run_on_spu():

    reference = []
    generate = []
    params = ppd.device("P2")(lambda x: x)(pretrained_model.params)

    for i in range(NUM):
        inputs_ids = processor(ds[739]["audio"]["array"], return_tensors="np")
        input_ids = ppd.device("P1")(lambda x: x)(inputs_ids.input_features)
        outputs_ids = ppd.device("SPU")(
            text_generation,
        )(input_ids, params)
        generate_ids = ppd.get(outputs_ids)

        generate_text = processor.batch_decode(generate_ids)[0]
        generate.extend([processor.tokenizer._normalize(generate_text)])
        reference.extend([processor.tokenizer._normalize(ds[739]["text"])])

    return 100 * wer.compute(references=reference, predictions=generate)

def run_on_bench():

    reference = []
    generate = []
    params = ppd.device("P2")(lambda x: x)(pretrained_model.params)

    for i in range(NUM):
        inputs_ids = processor(ds[739]["audio"]["array"], return_tensors="np")
        print(inputs_ids)

        input_ids = ppd.device("P1")(lambda x: x)(inputs_ids.input_features)
    
        with hack_softmax_context("hack jax softmax", enabled=True), hack_gelu_context("hijack jax gelu", enabled=True):
            outputs_ids = ppd.device("SPU")(
                text_generation, copts=copts,
            )(input_ids, params)
            generate_ids = ppd.get(outputs_ids)

        generate_text = processor.batch_decode(generate_ids)[0]
        generate.extend([processor.tokenizer._normalize(generate_text)])
        reference.extend([processor.tokenizer._normalize(ds[739]["text"])])

    return 100 * wer.compute(references=reference, predictions=generate)


if __name__ == '__main__':
    print('\n------\nRun on CPU')
    out_cpu = run_on_cpu()
    print("wer cpu: ", out_cpu)

    print('\n------\nRun on SPU')
    out_spu = run_on_spu()
    print("wer spu: ", out_spu)

    #print('\n------\nRun on bench')
    out_bench = run_on_bench()
    print("wer bench: ", out_bench)
