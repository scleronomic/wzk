import os
import subprocess

import numpy as np

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


def get_load(host):
    # https://stackoverflow.com/a/24841357/7570817
    load = ssh_cmd(host=host, cmd="uptime | sed 's/.*: //'")
    load = load.split(',')
    return np.array([float(l) for l in load]).sum() / 3


def get_cpu_usage(host):
    # https://stackoverflow.com/a/9229580/7570817
    return float(ssh_cmd(host=host,
                         cmd="grep 'cpu ' /proc/stat | awk '{usage=($2+$4)*100/($2+$4+$5)} END {print usage}'"))


def get_n_cpu(host):

    return int(ssh_cmd(host, 'grep -c ^processor /proc/cpuinfo'))


