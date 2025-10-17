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

# Start nodes.
# > bazel run -c opt //examples/python/utils:nodectl -- --config `pwd`/examples/python/ml/griffin_t5/3pc.json up
#
# Run this example script.
# > bazel run -c opt //examples/python/ml/griffin_t5:griffin_t5_summ

import argparse
import json
from typing import Any, Optional, Tuple, Union
from contextlib import contextmanager
from flax.linen.linear import Array
import jax.nn as jnn
import jax.numpy as jnp 
import pdb

import spu.intrinsic as intrinsic
import spu.spu_pb2 as spu_pb2
import spu.utils.distributed as ppd

parser = argparse.ArgumentParser(description='distributed driver.')
parser.add_argument("-c", "--config", default="examples/python/conf/3pc.json")
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
    
    # hijack softmax functions using griffin
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
    print("hack gelu")

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


from transformers import AutoTokenizer, FlaxT5ForConditionalGeneration
from datasets import load_dataset
from rouge import Rouge

tokenizer = AutoTokenizer.from_pretrained("ndtran/t5-small_cnn-daily-mail")
pretrained_model = FlaxT5ForConditionalGeneration.from_pretrained("ndtran/t5-small_cnn-daily-mail", from_pt=True)

# Load CNN/Daily Mail dataset
dataset = load_dataset("ccdv/cnn_dailymail", '3.0.0', split="test")  # Specify the split you want to use
# dataset = load_dataset("wmt16", "de-en", split="test")

# Tokenize and preprocess the dataset
def tokenize_and_preprocess(example):
    source_text = example["article"]
    inputs = tokenizer(source_text, return_tensors="jax", max_length=256, truncation=True, padding=True)
    return inputs

tokenized_dataset = dataset.map(tokenize_and_preprocess, batched=False)


NUM = 1

reference_summaries = []
for i in range(NUM):
    reference_summaries.extend([tokenized_dataset[i]["highlights"]])



def summary_generation(input_ids, params):
    pretrained_model.params = params
    summary_ids = pretrained_model.generate(input_ids, max_new_tokens=50).sequences
    return summary_ids


def run_on_cpu():
    generated_summaries = []
    for i in range(NUM):
        input_ids = jnp.array(tokenized_dataset[i]["input_ids"])
        generated_ids = pretrained_model.generate(input_ids, max_new_tokens=50).sequences
        generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        generated_summaries.extend(generated_text)
        
    return generated_summaries


def run_on_spu():
    generated_summaries = []
    params = ppd.device("P2")(lambda x: x)(pretrained_model.params)
    for i in range(NUM):
        ids = jnp.array(tokenized_dataset[i]["input_ids"])
        input_ids = ppd.device("P1")(lambda x: x)(ids)
        generated_ids = ppd.device("SPU")(
            summary_generation, 
        )(input_ids, params)
        generated_ids = ppd.get(generated_ids)
        generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        generated_summaries.extend(generated_text)

    return generated_summaries

def run_on_griffin():
    generated_summaries = []
    params = ppd.device("P2")(lambda x: x)(pretrained_model.params)
    for i in range(NUM):
        ids = jnp.array(tokenized_dataset[i]["input_ids"])
        input_ids = ppd.device("P1")(lambda x: x)(ids)
        with hack_softmax_context("hack jax softmax", enabled=True), hack_gelu_context("hijack jax gelu", enabled=True):
            generated_ids = ppd.device("SPU")(
                summary_generation, copts=copts,
            )(input_ids, params)
            generated_ids = ppd.get(generated_ids)
            generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            generated_summaries.extend(generated_text)

    return generated_summaries


if __name__ == '__main__':
    
    rouge = Rouge()
    print('\nndtran/t5-small_cnn-daily-mail, Summarize, cdv/cnn_dailymail3.0test, Num=100, InputLength 256, Max New Token 50\n')
    print('\n------\nRun on CPU')
    generated_summaries_cpu = run_on_cpu()
    scores = rouge.get_scores(generated_summaries_cpu, reference_summaries, avg=True)
    # Print ROUGE scores for CPU
    print("ROUGE Scores CPU:")
    print("ROUGE-1-CPU: ", scores["rouge-1"])
    print("ROUGE-2-CPU: ", scores["rouge-2"])
    print("ROUGE-L-CPU: ", scores["rouge-l"])
    
    print('\n------\nRun on SPU')
    generated_summaries_spu = run_on_spu()
    scores = rouge.get_scores(generated_summaries_spu, reference_summaries, avg=True)
    # Print ROUGE scores for SPU
    print("ROUGE Scores SPU:")
    print("ROUGE-1-SPU: ", scores["rouge-1"])
    print("ROUGE-2-SPU: ", scores["rouge-2"])
    print("ROUGE-L-SPU: ", scores["rouge-l"])

    print('\n------\nRun on Griffin')
    generated_summaries_spu = run_on_griffin()
    scores = rouge.get_scores(generated_summaries_spu, reference_summaries, avg=True)
    # Print ROUGE scores for SPU
    print("ROUGE Scores Griffin:")
    print("ROUGE-1-Griffin: ", scores["rouge-1"])
    print("ROUGE-2-Griffin: ", scores["rouge-2"])
    print("ROUGE-L-Griffin: ", scores["rouge-l"])

    
