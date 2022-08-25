import re
import socket

import ray  # noqa
import fire
import numpy as np

from wzk.ltd import squeeze, atleast_list
from wzk.cpu import ssh_call2, get_n_cpu

# rmc-lx0095
# Johannes Pitz: 0392, 0179, 0145, 0115
# Leon: 0144
# ['rmc-lx0140',  'rmc-lx0271'] no longer available
# 'rmc-galene'
# __default_nodes = ['rmc-lx0062',
#                    'philotes', 'polyxo', 'poros',
#                    'rmc-galene', 'rmc-lx0271', 'rmc-lx0141', 'rmc-lx0392']
__default_nodes = []
# __default_nodes = ['rmc-lx0144', 'rmc-lx0062']
# __default_nodes = ['rmc-lx0062', 'philotes', 'polyxo', 'poros']
#

_address = ['auto']
_password = ['']


def __start_head(head, perc, verbose=0):
    # start_head_cmd = f'ray start --head --port=6379 --num-cpus='
    start_head_cmd = f'ray start --head --port=6378 --num-cpus='
    n_cpu = int(max(1, get_n_cpu(head) * perc))
    stdout = ssh_call2(host=head, cmd=start_head_cmd+str(n_cpu))
    head = socket.gethostname() if head is None else head
    if verbose > 0:
        print(head, ':', stdout)

    return head, stdout, n_cpu


def __get_address_password(stdout):
    pattern_adr = r"--address='\S*'"
    pattern_pwd = r"--redis-password='\S*'"
    address = squeeze(re.compile(pattern_adr).findall(stdout))
    address = address[address.find('=')+1:]
    password = squeeze(re.compile(pattern_pwd).findall(stdout))
    password = password[password.find('=')+1:]

    _address[0] = address
    _password[0] = password
    return address, password


def __start_nodes(nodes, address, password, perc, verbose=0):
    n_cpu = 0
    for node in nodes:
        n_cpu_i = int(max(1, get_n_cpu(node) * perc))
        start_node_cmd = f"ray start --address='{address}' --redis-password='{password}' --num-cpus={n_cpu_i}"
        stdout = ssh_call2(host=node, cmd=start_node_cmd)
        if verbose > 1:
            print(node, ':', stdout)

        n_cpu += n_cpu_i

    return n_cpu


def start_ray_cluster(head=None, nodes=None, perc=80, verbose=2):
    assert 1.0 <= perc <= 100.0
    perc = perc / 100

    if nodes is None:
        nodes = __default_nodes

    head, stdout, n_cpu = __start_head(head=head, perc=perc, verbose=verbose-1)
    address, password = __get_address_password(stdout=stdout)
    nodes = [] if (nodes is None or nodes == []) else np.setdiff1d(atleast_list(nodes), [head])
    n_cpu += __start_nodes(nodes=nodes, address=address, password=password, perc=perc, verbose=verbose-2)

    if verbose > 0:
        print('Started Ray-Cluster')
        print('Nodes: ', *nodes)
        print('Total Number of CPUs: ', n_cpu)

    return n_cpu


def stop_ray_cluster(nodes=None, verbose=1):
    if nodes is None:
        nodes = __default_nodes

    if verbose > 0:
        print('Stop Ray-Cluster')
        print('Nodes: ', *nodes)

    for node in atleast_list(nodes):
        stdout = ssh_call2(host=node, cmd='ray stop --force')
        if verbose > 1:
            print(node, ':', stdout)


def ray_main(mode='start', nodes=None, head=None, perc=80, verbose=2):
    if mode == 'start':
        start_ray_cluster(head=head, nodes=nodes, perc=perc, verbose=verbose)
    elif mode == 'stop':
        stop_ray_cluster(nodes=nodes, verbose=verbose)
    else:
        raise ValueError


def ray_init(perc=100):
    try:
        ray.init(address=_address[0], log_to_driver=False, ignore_reinit_error=True)
    except ConnectionError:
        start_ray_cluster(perc=perc, verbose=1)
        ray.init(address='auto', log_to_driver=False, ignore_reinit_error=True)
        # ray.init(address=f"ray://{_address[0]}", log_to_driver=False, ignore_reinit_error=True)


def ray_wrapper(fun, n, **kwargs):
    futures = []
    for i in range(n):
        futures.append(fun.remote(**kwargs))
    return ray.get(futures)


if __name__ == '__main__':
    fire.Fire(ray_main)
