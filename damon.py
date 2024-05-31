import sys, os
import subprocess
import time
import psutil
import pandas as pd


def exe_cmd(cmd):
    process = subprocess.Popen(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    stdout, stderr = process.communicate()
    return stdout.decode(), stderr


def get_pid(key, screen_out):
    id_end = screen_out.find(key)
    if id_end == -1:
        return None
    id_end -= 1
    id_begin = screen_out[:id_end].rfind("\n") + 2
    id = int(screen_out[id_begin:id_end])
    return id

def get_memory_info(pid, max_depth=0, depth=0):
    total_memory = psutil.virtual_memory().total
    try:
        process = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return 0, 0

    mem_info = process.memory_info()
    mem = mem_info.rss

    children = process.children(recursive=False)

    if max_depth > depth:
        for child in children:
            child_mem, _ = get_memory_info(child.pid, max_depth, depth+1)
            mem += child_mem

    mem_percent = mem / total_memory * 100

    return mem, mem_percent

def get_last():   
    try:
        df = pd.read_csv("./output/msr.csv", index_col=0, names=["origin_index", "pred", "token_count","duration","comm"])  
        return df.index.to_list()[-1]+1
    except:
        return 0


def damon_main():
    out, eror = exe_cmd("screen -ls")
    sid = get_pid("mpc_server", out)
    cid = get_pid("mpc_client", out)
    print(sid, cid)

    last = get_last()
    THRESHOLD = 80

    memory_used_percent = psutil.virtual_memory().percent
    if memory_used_percent > THRESHOLD or (cid is None and sid is not None):
        print(f"restart server and client from {last}!")
        if sid is not None:
            exe_cmd(f"kill {sid}")
        if cid is not None:
            exe_cmd(f"kill {cid}")
        time.sleep(5)
        exe_cmd("sh ./scripts/run_server.sh")
        time.sleep(5)
        exe_cmd(f"sh ./scripts/run_mpc_msr.sh {last}")


if __name__ == "__main__":
    while True:
        time.sleep(30)
        damon_main()
