import os
import re
import socket

import ray
import fire
import numpy as np

from wzk.ssh import execute_via_ssh
from wzk.dicts_lists_tuples import safe_squeeze


__default_nodes = ['rmc-lx0062', 'rmc-lx0144', 'rmc-lx0140', 'rmc-lx0271',
                   'philotes', 'polyxo', 'poros']


def start_ray_cluster(head=None, nodes=None, verbose=2):

    log = ''
    if nodes is None:
        nodes = __default_nodes

    if verbose > 0:
        print('Start Ray-Cluster')
        print('Nodes: ', *nodes)

    if head is None:
        head = socket.gethostname()

    start_head_cmd = 'ray start --head --port=6379'
    stdout = execute_via_ssh(head, cmd=start_head_cmd)

    log += head + ':\n' + stdout + '\n'
    if verbose > 1:
        print(head, ':', stdout)


    pattern_adr = r"--address='\S*'"
    pattern_pwd = r"--redis-password='\S*'"
    address = safe_squeeze(re.compile(pattern_adr).findall(stdout))
    password = safe_squeeze(re.compile(pattern_pwd).findall(stdout))
    address = address[address.find('=')+1:]
    password = password[password.find('=')+1:]

    start_node_cmd = f"ray start --address='{address}' --redis-password='{password}'"
    for node in nodes:
        stdout = execute_via_ssh(remote_client=node, cmd=start_node_cmd)
        if verbose > 1:
            print(node, ':', stdout)
            log += node + ':\n' + stdout + '\n'

    np.save(os.path.abspath(os.path.dirname(__file__)) + '/' + 'ray_start.text', log)



if __name__ == '__main__':
    fire.Fire(start_ray_cluster)
