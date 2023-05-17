import subprocess
import copy
import fcntl
import time
import json

base = ["docker", "exec", "-it", "a703c1aa7183", "watch", "-n", "1", "nvidia-smi"]

base_restart = ["docker", "restart"]
base_exec = ["docker", "exec", "-t", "python", "automated.py"]

running = True
while running:
    lockfile = open("shared_file_process.lock", 'r+')
    fcntl.flock(lockfile, fcntl.LOCK_EX)
    with open("running_process.json", 'r+') as fp:
        data = json.load(fp)
    if data["status"] == "finished":
        running = False
    else:
        dockers = list(data.keys())
        dockers.remove("status")
        for docker in dockers:
            if data[docker]["status"] == "DOWN":
                tmp_restart = copy.deepcopy(base_restart)
                tmp_restart.append(docker)
                tmp_exec = copy.deepcopy(base_exec)
                tmp_exec.insert(3, docker)
                tmp_exec.append(data[docker]["gpu"])
                subprocess.Popen(tmp_restart)
                time.sleep(60)
                p = subprocess.Popen(tmp_exec)
                # p.communicate()
                data[docker]["status"] = "UP"
        with open("running_process.json", "w") as fp:
            json.dump(data, fp)

    fcntl.flock(lockfile, fcntl.LOCK_UN)
    lockfile.close()
    
    time.sleep(5*60)

