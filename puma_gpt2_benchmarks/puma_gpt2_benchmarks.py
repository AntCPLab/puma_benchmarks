import jax.numpy as jnp
from transformers import AutoTokenizer, FlaxGPT2LMHeadModel, GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
import datasets
import numpy as np
from typing import Any, Callable, Dict, Optional, Tuple, Union
import pdb
import datasets
import evaluate
import jax.nn as jnn
import flax.linen as nn
from flax.linen.linear import Array
import re 
import jax
import argparse
import json
import torch
from torch.utils.data import DataLoader
import pdb
import spu.utils.distributed as ppd
import spu.intrinsic as intrinsic
import spu.spu_pb2 as spu_pb2
from contextlib import contextmanager

copts = spu_pb2.CompilerOptions()
copts.enable_pretty_print = False
copts.xla_pp_kind = 2
# enable x / broadcast(y) -> x * broadcast(1/y)
copts.enable_optimize_denominator_with_broadcast = True
Array = Any
Dataset = datasets.arrow_dataset.Dataset
PRNGKey = Any
logits_fn=lambda logits: logits.argmax(-1)
batch_num = 100

parser = argparse.ArgumentParser(description='distributed driver.')
parser.add_argument("-c", "--config", default="examples/python/ml/puma_gpt2_benchmarks/3pc.json")
args = parser.parse_args()
with open(args.config, 'r') as file:
    conf = json.load(file)

ppd.init(conf["nodes"], conf["devices"])

model_path = "gpt2" # loading gpt2 from huggingface or locally
model_config = GPT2Config.from_pretrained(model_path)
tokenizer = GPT2Tokenizer.from_pretrained(model_path)

tokenizer.pad_token_id = tokenizer.eos_token_id

pretrained_model = FlaxGPT2LMHeadModel.from_pretrained(model_path, config=model_config, from_pt=True)
dataset = datasets.load_dataset('wikitext', 'wikitext-103-v1', split='validation')
text = [text for text in dataset['text'] if text is not None and text != ""]


def hack_softmax(x: Array,
            axis: Optional[Union[int, Tuple[int, ...]]] = -1,
            where: Optional[Array] = None,
            initial: Optional[Array] = None) -> Array:

    x_max = jnp.max(x, axis, where=where, initial=initial, keepdims=True)
    x = x - x_max

    # exp on large negative is clipped to zero
    b = x > -14
    nexp = jnp.exp(x) * b

    divisor = jnp.sum(nexp, axis, where=where, keepdims=True)

    return nexp / divisor

@contextmanager
def hack_softmax_context(msg: str, enabled: bool = False):
    if not enabled:
        yield
        return
    # hijack some target functions
    raw_softmax = jnn.softmax
    jnn.softmax = hack_softmax
    yield
    # recover back
    jnn.softmax = raw_softmax

def hack_gelu(x: Array,
            axis: Optional[Union[int, Tuple[int, ...]]] = -1,
            where: Optional[Array] = None,
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



def calculate_newToken(params, input_ids, new_token=1):
    model = FlaxGPT2LMHeadModel(config=model_config)
    for i in range(new_token):
        outputs = model(input_ids=input_ids, params=params)
        logits = outputs.logits[:, :-1, :].reshape(batch_num, -1)
        next_token = jnp.argmax(logits, axis = 1)
        next_token_exp = jnp.expand_dims(next_token, axis=1)
        input_ids = jnp.concatenate([input_ids, next_token_exp], axis = 1)
    return input_ids

def calculate_logits(params, input_ids):
    model = FlaxGPT2LMHeadModel(config=model_config)
    outputs = model(input_ids=input_ids, params=params)
    logits = outputs.logits[:, :-1, :]

    return logits    


    
def eval_puma_newToken():
    ids = tokenizer.encode(text[0], return_tensors = 'jax')

    with hack_softmax_context("hijack jax softmax", enabled = True), hack_gelu_context("hijack jax gelu", enabled=True):
        input_ids = ppd.device("P1")(lambda x: x)(ids)
        params = ppd.device("P2")(lambda x: x)(pretrained_model.params)
        outputs_ids = ppd.device("SPU")(calculate_newToken, copts=copts)(params, input_ids, new_token=1)
        outputs_ids = ppd.get(outputs_ids)

    return outputs_ids

def eval_puma_perp():
    total_loss = 0
    total_count = 0
    for j in range(batch_num):
        ids = tokenizer.encode(text[j], return_tensors = 'jax')
        labels = ids[:, 1:]

        with hack_softmax_context("hijack jax softmax", enabled = True), hack_gelu_context("hijack jax gelu", enabled=True):
            input_ids = ppd.device("P1")(lambda x: x)(ids)
            params = ppd.device("P2")(lambda x: x)(pretrained_model.params)
            logits = ppd.device("SPU")(calculate_logits, copts=copts)(params, input_ids, new_token=1)
            logits = ppd.get(logits)
        
        loss = jnp.sum(jax.nn.log_softmax(logits, axis=-1) * jax.nn.one_hot(labels, logits.shape[-1]), axis=-1)
        count = jnp.sum(labels != tokenizer.pad_token_id) # Use the padding token ID
        total_loss += jnp.sum(loss)
        total_count += count

    perplexity = jnp.exp(- total_loss / total_count)
    return perplexity.item()

def eval_cpu_newToken():
    ids = tokenizer.encode(text[0], return_tensors = 'jax')
    outputs_ids = calculate_newToken(pretrained_model.params, ids, new_token=1)
    return outputs_ids

def eval_cpu_perp():
    total_loss = 0
    total_count = 0
    for i in range(batch_num):
        ids = tokenizer.encode(text[i], return_tensors = 'jax')
        labels = ids[:, 1:]
        logits = calculate_logits(pretrained_model.params, ids)
        loss = jnp.sum(jax.nn.log_softmax(logits, axis=-1) * jax.nn.one_hot(labels, logits.shape[-1]), axis=-1)
        count = jnp.sum(labels != tokenizer.pad_token_id) # Use the padding token ID
        total_loss += jnp.sum(loss)
        total_count += count

    perplexity = jnp.exp(- total_loss / total_count)
    return perplexity.item()
    

if __name__ == '__main__':
    print("Perplexity of 100 wikitext-103-v1 sentences")
    perplexity_cpu = eval_cpu_perp()
    print("Perplexity-cpu:", perplexity_cpu)
    perplexity_puma = eval_puma_perp()
    print("Preplexity-puma:", perplexity_puma)

    print("New Tokens of wikitext-103-v1 sentences")
    outids_cpu = eval_cpu_newToken()
    print("New Token-cpu:", tokenizer.decode(outids_cpu[0], skip_special_tokens=True))
    outids_puma = eval_puma_newToken()
    print("New Token-puma:", tokenizer.decode(outids_puma[0], skip_special_tokens=True))
