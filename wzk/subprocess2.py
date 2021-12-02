import os
import subprocess
from wzk.strings import uuid4


def __read_and_delete(temp):
    with open(temp, 'r') as f:
        stdout = f.read()

    os.remove(temp)
    return stdout


def __run(temp, cmd, check):
    with open(temp, 'w') as f:
        _ = subprocess.run([cmd], stdout=f, shell=True, check=check)


def __run_ssh(temp, cmd, check, host):
    with open(temp, 'w') as f:
        _ = subprocess.run(['ssh', host, cmd], stdout=f, shell=False, check=check)


def call(cmd, check=False):
    """Not the most elegant way, but was not able to get the stdout without error / messed up shells otherwise"""
    temp = uuid4()
    __run(temp=temp, cmd=cmd, check=check)
    return __read_and_delete(temp=temp)


def ssh_call(host, cmd, check=False):
    temp = uuid4()

    if host is None or host == '':
        __run(temp=temp, cmd=cmd, check=check)
    else:
        __run_ssh(temp=temp, cmd=cmd, check=check, host=host)

    return __read_and_delete(temp=temp)
