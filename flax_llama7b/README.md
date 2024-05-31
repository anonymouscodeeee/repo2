export LD_PRELOAD=libtcmalloc.so.4
python ../nodectl.py   --config ./3pc.json up
    ```

4. Run `flax_llama7b` example

    ```sh
    bazel run -c opt //examples/python/ml/flax_llama7b -- --config `pwd`/examples/python/ml/flax_llama7b/3pc.json
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
