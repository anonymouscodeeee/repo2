export LD_PRELOAD=libtcmalloc.so.4
screen -L -Logfile ./logs/client.log -dmS mpc_client python mpc_vulnerability_detection.py -r $1 -m spu