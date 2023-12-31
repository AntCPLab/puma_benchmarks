# Puma_benchmarks
This repo is to reproduce the results in our paper [Puma](https://arxiv.org/abs/2307.12533)

## 0. Compile and Launch SecretFlow-SPU

git clone SecretFlow-SPU

```sh
git clone https://github.com/secretflow/spu.git & cd spu
```

and follow SecretFlow-SPU [README.md](https://github.com/secretflow/spu/blob/main/CONTRIBUTING.md#build) to build spu from source. 

## 1. Launch SPU-backends:

In Puma, we launch 5 nodes, 3 for ABY3 computing nodes, 1 for model provider, and the last one for input provider. On each node, run the following

```sh
bazel run -c opt //examples/python/utils:nodectl -- --config `pwd`/examples/python/ml/puma_bert_benchmarks/3pc.json up
```

To profile the costs of Puma, turn on the following in `3pc.json`

```shell
"enable_pphlo_profile": true,
"enable_hal_profile": true,
```
and you will get the time and communication costs on each ABY3 node.


## 2. Install huggingface transformers library

```sh
pip install 'transformers[flax]'
```
To hijack, we need to modify the activation function of Bert and GPT2 as `jax.nn.gelu`.


## 3. Config and Run Puma
After launching SPU, move the directory of `puma_bert_benchmarks` and `puma_gpt2_benchmarks` to `spu/examples/python/ml` and run the scripts. For example, run

```sh
bazel run -c opt //examples/python/ml/puma_bert_benchmarks:puma_bert_benchmarks
```
to re-produce our results for Bert on GLUE.
Also, you can modify the model path and task_name easily in the python scripts to run Puma on more model and dataset. For example, Puma achieves the following performance on Bert-base on GLUE-CoLA:

| MCC | Time | Comm. |
|-----|------|-------|
|0.613|  42s |3.59GB |  


To run Puma on LLaMA-7B, please refer to: 
https://github.com/secretflow/spu/tree/main/examples/python/ml/flax_llama7b

