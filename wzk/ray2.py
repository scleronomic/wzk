import os
import re
import socket

import ray  # noqa
import fire
import numpy as np

from wzk.ssh import ssh_cmd, get_n_cpu
from wzk.dicts_lists_tuples import safe_squeeze, atleast_list

# Johannes Pitz: 0392, 0179, 0145, 0115
# ['rmc-lx0140', 'rmc-lx0144', 'rmc-lx0271'] no longer available
__default_nodes = [#  'rmc-lx0062',
                   'rmc-lx0115', 'rmc-lx0179', 'rmc-lx0271',
                   'rmc-galene',
                   'philotes', 'polyxo', 'poros']


def start_ray_cluster(head=None, nodes=None, perc=80, verbose=2):

    perc = perc / 100

    log = ''
    if nodes is None:
        nodes = __default_nodes

    if verbose > 0:
        print('Starting Ray-Cluster...')
        print('Nodes: ', *nodes)

    if head is None:
        head = socket.gethostname()

    n_cpu = int(get_n_cpu(head) * perc)
    start_head_cmd = f'ray start --head --port=6379 --num-cpus={n_cpu}'
    stdout = ssh_cmd(host=head, cmd=start_head_cmd)

    log += head + ':\n' + stdout + '\n'
    if verbose > 1:
        print(head, ':', stdout)

    pattern_adr = r"--address='\S*'"
    pattern_pwd = r"--redis-password='\S*'"
    address = safe_squeeze(re.compile(pattern_adr).findall(stdout))
    password = safe_squeeze(re.compile(pattern_pwd).findall(stdout))
    address = address[address.find('=')+1:]
    password = password[password.find('=')+1:]

    nodes = np.setdiff1d(atleast_list(nodes), [head])
    n_cpu_total = 0
    for node in nodes:
        n_cpu = int(get_n_cpu(node) * perc)
        start_node_cmd = f"ray start --address='{address}' --redis-password='{password}' --num-cpus={n_cpu}"
        stdout = ssh_cmd(host=node, cmd=start_node_cmd)
        if verbose > 1:
            print(node, ':', stdout)
            log += node + ':\n' + stdout + '\n'

        n_cpu_total += n_cpu

    if verbose > 0:
        print('Started Ray-Cluster')
        print('Nodes: ', *nodes)
        print('Total Number of CPUs: ', n_cpu_total)

    np.save(os.path.abspath(os.path.dirname(__file__)) + '/' + 'ray_start.text', log)


def stop_ray_cluster(nodes=None, verbose=1):
    if nodes is None:
        nodes = __default_nodes

    if verbose > 0:
        print('Stop Ray-Cluster')
        print('Nodes: ', *nodes)

    for node in atleast_list(nodes):
        stdout = ssh_cmd(host=node, cmd='ray stop --force')
        if verbose > 1:
            print(node, ':', stdout)


def ray_main(mode='start', nodes=None, head=None, perc=80, verbose=2):
    if mode == 'start':
        start_ray_cluster(head=head, nodes=nodes, perc=perc, verbose=verbose)
    elif mode == 'stop':
        stop_ray_cluster(nodes=nodes, verbose=verbose)
    else:
        raise ValueError


if __name__ == '__main__':
    fire.Fire(ray_main)
