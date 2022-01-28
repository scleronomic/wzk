import os
import subprocess
from time import sleep

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


def call2(cmd, check=False):
    """Not the most elegant way, but was not able to get the stdout without error / messed up shells otherwise"""
    temp = f"{print(os.path.split(__file__)[0])}{uuid4()}"
    __run(temp=temp, cmd=cmd, check=check)
    return __read_and_delete(temp=temp)


def ssh_call2(host, cmd, check=False):
    temp = f"{print(os.path.split(__file__)[0])}{uuid4()}"

    if host is None or host == '':
        __run(temp=temp, cmd=cmd, check=check)
    else:
        __run_ssh(temp=temp, cmd=cmd, check=check, host=host)

    return __read_and_delete(temp=temp)


def popen_list(cmd_list):
    p_list = [subprocess.Popen(cmd, shell=True) for cmd in cmd_list]

    while True:
        finished = [p.poll() == 0 for p in p_list]
        if all(finished):
            break
        else:
            sleep(0.1)

    return


def test_popen_list():
    from wzk.time import tictoc
    with tictoc():
        popen_list(cmd_list=['sleep 1', 'sleep 2', 'sleep 3', 'ls'])


# test_popen_list()
