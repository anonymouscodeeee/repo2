export LD_PRELOAD=libtcmalloc.so.4
screen -L -Logfile ./logs/server.log -dmS mpc_server python nodectl.py   --config ./3pc.json up