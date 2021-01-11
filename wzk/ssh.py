import os
import subprocess

from wzk.strings import uuid4

def ssh_cmd(host, cmd, check=False):
    """Not the most elegant way, bit was not able to get the stdout without error / messed up shells otherwise"""
    temp = uuid4()
    with open(temp, 'w') as f:
        result = subprocess.run(["ssh", host, cmd], stdout=f,
                                shell=False, check=check)

    with open(temp, 'r') as f:
        stdout = temp.read()

    os.remove(temp)
    return res
