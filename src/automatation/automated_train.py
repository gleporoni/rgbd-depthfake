import subprocess
import copy
import sys
import fcntl
import torch
import socket
import json

def get_container_name():
    hostname = socket.gethostname()
    return hostname

# Usage
container_name = get_container_name()

gpu = sys.argv[1]


base = ['python', '/workdir/src/train.py', 'run.cuda_device='+"'"+str(gpu)+"'"]

line = 1
exit = False
while not exit:
    if not torch.cuda.is_available():
        lockfile = open("shared_file_process.lock", 'w')
        fcntl.flock(lockfile, fcntl.LOCK_EX)
        with open("running_process.json", 'r+') as fp:
            data = json.load(fp)
        
        data[container_name]["status"] = "DOWN"
        
        with open("running_process.json", 'w') as fp:
            json.dump(data, fp)

        fcntl.flock(lockfile, fcntl.LOCK_UN)
        lockfile.close()
        exit = True
    else:
        line = None

        lockfile = open("shared_file_train.lock", 'w')
        fcntl.flock(lockfile, fcntl.LOCK_EX)
        with open("train_to_do.txt", 'r+') as fp:
            lines = fp.read().splitlines()
            fp.seek(0)

            lines2 = fp.readlines()
            fp.seek(0)
            fp.truncate()

            fp.writelines(lines2[1:])
            if len(lines) != 0:
                line = lines[0]
        fcntl.flock(lockfile, fcntl.LOCK_UN)
        lockfile.close()
        
        if line is None:
            lockfile = open("shared_file_process.lock", 'w')
            fcntl.flock(lockfile, fcntl.LOCK_EX)
            with open("running_process.json", 'r+') as fp:
                data = json.load(fp)
            
            data["status"] = "finished"
            
            with open("running_process.json", 'w') as fp:
                json.dump(data, fp)

            fcntl.flock(lockfile, fcntl.LOCK_UN)
            lockfile.close()
            exit = True
            
        else:
            tmp = copy.deepcopy(base)
            for change in line.split(" "):
                tmp.append(change)
            print(tmp)
            subprocess.run(tmp)
            print("---------------------------")



