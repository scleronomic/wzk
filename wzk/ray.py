import ray
import re
import socket

import fire

from wzk.ssh import execute_via_ssh
from wzk.dicts_lists_tuples import safe_squeeze


__default_nodes = ['rmc-lx0062', 'rmc-lx0144', 'rmc-lx0140', 'rmc-lx0271',
                   'philotes', 'polyxo', 'poros']


def start_ray_cluster(head=None, nodes=None, verbose=1):

    if nodes is None:
        nodes = __default_nodes

    if verbose > 0:
        print('Start Ray-Cluster')
        print('Nodes: ', *nodes)

    if head is None:
        head = socket.gethostname()

    start_head_cmd = 'ray start --head --port=6379'
    stdout = execute_via_ssh(head, cmd=start_head_cmd)

    pattern_adr = r"--address='\S*'"
    pattern_pwd = r"--redis-password='\S*'"
    address = safe_squeeze(re.compile(pattern_adr).findall(stdout))
    password = safe_squeeze(re.compile(pattern_pwd).findall(stdout))
    address = address[address.find('=')+1:]
    password = password[password.find('=')+1:]

    start_node_cmd = f"ray start --address='{address}' --redis-password='{password}'"
    for node in nodes:
        execute_via_ssh(remote_client=node, cmd=start_node_cmd)


def stop_ray_cluster(nodes=None, verbose=1):
    if nodes is None:
        nodes = __default_nodes

    if verbose > 0:
        print('Stop Ray-Cluster')
        print('Nodes: ', *nodes)
    for node in nodes:
        execute_via_ssh(remote_client=node, cmd='ray stop')


def ray_main(mode='start', nodes=None, head=None):
    if mode == 'start':
        start_ray_cluster(head=head, nodes=nodes)
    elif mode == 'stop':
        stop_ray_cluster(nodes=nodes)
    else:
        raise ValueError


if __name__ == '__main__':
    fire.Fire(ray_main)
