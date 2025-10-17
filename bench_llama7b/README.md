# Flax Llama-7B Example with Puma

This example demonstrates how to use SPU to run secure inference on a pre-trained
[Llama-7B](https://research.facebook.com/publications/llama-open-and-efficient-foundation-language-models/) model using PUMA.

1. Install huggingface transformers library

    ```sh
    pip install 'transformers[flax]'
    ```

2. Download EasyML library to support Flax-LLaMA-7B

    ```sh
    git clone https://github.com/young-geng/EasyLM.git
    cd EasyLM
    export PYTHONPATH="${PWD}:$PYTHONPATH"
    ```

    Download trained LLaMA-B[PyTroch-Version] from "https://huggingface.co/openlm-research/open_llama_7b", and convert it to Flax.msgpack as:

    ```sh
    python3 -m EasyLM.scripts.convert_checkpoint --load_checkpoint='params::path-to-LLaMA-7B[Pytroch-Version]' --output_file='path-to-LLaMMA-7B.msgpack' --streaming=False
    ```

3. Launch SPU backend runtime

    ```sh
    bazel run -c opt //examples/python/utils:nodectl -- --config `pwd`/examples/python/ml/bench_llama7b/3pc.json up
    ```

4. Run `bench_llama7b` example

    ```sh
    bazel run -c opt //examples/python/ml/bench_llama7b -- --config `pwd`/examples/python/ml/bench_llama7b/3pc.json
    ```

    and you can get the following results from our example:

    ```md
    ------
    Run on CPU
    Q: What is the largest animal?
    A: The largest animal is the blue whale.
    Q: What is the smallest animal?
    A: The smallest animal is the bee.

    ------
    Run on SPU
    Q: What is the largest animal?
    A: The largest animal is the blue whale.
    Q: What is the smallest animal?
    A: The smallest animal is the bee.
    ```

5, Run `griffin_llama7b_ppl` examplem similarly to compute ppl. on Wikitext-103 V1.