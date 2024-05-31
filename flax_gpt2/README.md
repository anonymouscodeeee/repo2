export LD_PRELOAD=libtcmalloc.so.4
python ../nodectl.py   --config ./3pc.json up
python ./flax_gpt2.py
