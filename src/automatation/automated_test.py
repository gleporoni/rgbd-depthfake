import subprocess
import copy
import sys
import fcntl


gpu = sys.argv[1]


base = ['python', '/workdir/src/test.py', 'run.cuda_device='+"'"+str(gpu)+"'"]

while True:


    line = None

    lockfile = open("shared_file_test.lock", 'w')
    fcntl.flock(lockfile, fcntl.LOCK_EX)
    with open("test_to_do.txt", 'r+') as fp:
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
        break

    tmp = copy.deepcopy(base)
    for change in line.split(" "):
        tmp.append(change)
    print(tmp)
    subprocess.run(tmp)
    print("---------------------------")

