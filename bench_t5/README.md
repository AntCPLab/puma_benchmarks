# Flax Whisper Example

This example demonstrates how to use SPU to run private inference on a pre-trained
[T5](https://huggingface.co/docs/transformers/model_doc/t5#transformers.FlaxT5ForConditionalGeneration) model.

1. Install huggingface transformers library

    ```sh
    pip install 'transformers[flax]'
    ```

2. Enable While with secret value

    Edit libspu/kernel/hlo/control_flow.cc, change `ENABLE_DEBUG_ONLY_REVEAL_SECRET_CONDITION` to `true`.

3. Launch SPU backend runtime

    ```sh
    bazel run -c opt //examples/python/utils:nodectl -- --config `pwd`/examples/python/ml/bench_t5/3pc.json up
    ```

4. Run `bench_t5_summ` example

    ```sh
    bazel run -c opt //examples/python/ml/bench_t5:bench_t5_summ -- --config `pwd`/examples/python/ml/bench_t5/3pc.json
    ```

5. Run `bench_t5_trans` example

    ```sh
    bazel run -c opt //examples/python/ml/bench_t5:bench_t5_trans -- --config `pwd`/examples/python/ml/bench_t5/3pc.json
    ```
