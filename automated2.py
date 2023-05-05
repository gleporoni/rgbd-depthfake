import subprocess
import copy
import sys
import fcntl


gpu = sys.argv[1]

# changes = [['project=ciao'],
#             ['run.seed=60'],
#             ['project=ciao', 'run.seed=60'],
#         ]

base = ['python', 'src/train.py', 'run.cuda_device='+str(gpu)]


line = 1
while True:


    line = None

    lockfile = open("shared_file.lock", 'w')
    fcntl.flock(lockfile, fcntl.LOCK_EX)
    with open("test_to_do.txt", 'r+') as fp:
       # read an store all lines into list
        lines = fp.read().splitlines()
        fp.seek(0)

        lines2 = fp.readlines()
        # move file pointer to the beginning of a file
        fp.seek(0)
        # truncate the file
        fp.truncate()

        # start writing lines except the first line
        # lines[1:] from line 2 to last line
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

