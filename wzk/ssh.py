import os
import subprocess

from wzk.strings import uuid4


def ssh_cmd(host, cmd, check=False):
    """Not the most elegant way, but was not able to get the stdout without error / messed up shells otherwise"""
    temp = uuid4()
    with open(temp, 'w') as f:
        _ = subprocess.run(["ssh", host, cmd], stdout=f, shell=False, check=check)

    with open(temp, 'r') as f:
        stdout = f.read()

    os.remove(temp)
    return stdout


def get_n_cpu(host):
    try:
        n_cpu = int(ssh_cmd(host, 'grep -c ^processor /proc/cpuinfo'))
        return n_cpu

    except ValueError:
        print(ssh_cmd(host, 'grep -c ^processor /proc/cpuinfo'))

        raise ValueError
