export LD_PRELOAD=libtcmalloc.so.4
screen -L -Logfile ./logs/client.log -dmS mpc_client python roberta.py -r $1