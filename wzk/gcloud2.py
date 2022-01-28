import os
import subprocess
import pandas as pd
from io import StringIO

from wzk.subprocess2 import call2, popen_list
from wzk.time import tictoc
from wzk.strings import uuid4
from wzk.dicts_lists_tuples import atleast_list, flatten


GCP_PROJECT = os.environ['GCP_PROJECT']
GCP_ACCOUNT_NR = os.environ['GCP_ACCOUNT_NR']
GCP_ZONE = 'us-central1-a'
__GCP_SCOPES = "https://www.googleapis.com/auth/devstorage.read_only," \
               "https://www.googleapis.com/auth/logging.write," \
               "https://www.googleapis.com/auth/monitoring.write," \
               "https://www.googleapis.com/auth/servicecontrol," \
               "https://www.googleapis.com/auth/service.management.readonly," \
               "https://www.googleapis.com/auth/trace.append"

GCP_USER = os.environ['GCP_USER']
GCP_USER_SHORT = os.environ['GCP_USER_SHORT']

GCP_USER_LABEL = f'user={GCP_USER_SHORT}'


def add_old_disks_flag(disks):
    disks = atleast_list(disks, convert=False)
    cmd = [f"--disk=boot=no,device-name={d['name']},name={d['name']},mode=rw" for d in disks]
    cmd = ' '.join(cmd)
    return cmd


def add_new_disks_flag(disks):
    disks = atleast_list(disks, convert=False)
    cmd = [f"--create-disk=boot={d['boot']}," 
           f"auto-delete={d['autodelete']},"
           f"device-name={d['name']},"
           f"mode=rw,size={d['size']},"
           f"source-snapshot=projects/{GCP_PROJECT}/global/snapshots/{d['snapshot']},"
           f"type=projects/{GCP_PROJECT}/zones/{GCP_ZONE}/diskTypes/pd-balanced"
           for d in disks]
    cmd = ' '.join(cmd)
    return cmd


def get_disks():
    cmd = "gcloud compute disks list"
    s = call2(cmd)
    disks = pd.read_table(StringIO(s), delim_whitespace=True)
    return disks


def add_startup_script_flag(startup_script):
    if startup_script is not None and startup_script != '':
        return f'--metadata=startup-script="{startup_script}"'
    else:
        return ''


def create_instance_cmd(config):
    cmd = f"gcloud compute instances create {config['name']} " \
          f"--machine-type={config['machine']} " \
          f"{add_new_disks_flag(config['disks_new'])} " \
          f"{add_old_disks_flag(config['disks_old'])} " \
          f"{add_startup_script_flag(config['startup_script'])} " \
          f"--zone={GCP_ZONE} " \
          f"--project={GCP_PROJECT} " \
          f"--scopes={__GCP_SCOPES} " \
          f"--labels={config['labels']} " \
          f"--service-account={GCP_ACCOUNT_NR}-compute@developer.gserviceaccount.com " \
          f"--preemptible " \
          f"--no-restart-on-failure " \
          f"--reservation-affinity=any " \
          f"--maintenance-policy=TERMINATE "
    return cmd


def create_disk_cmd(disk):
    cmd = f"gcloud beta compute disks create {disk['name']} " \
          f"--size={disk['size']}GB " \
          f"--labels={disk['labels']} " \
          f"--project={GCP_PROJECT}  " \
          f"--zone={GCP_ZONE} " \
          f"--type=pd-balanced"
    return cmd


def attach_disk_cmd(instance, disk):
    if isinstance(instance, dict):
        instance = instance['name']
    if isinstance(disk, dict):
        disk = disk['name']
    cmd = f"gcloud compute instances attach-disk {instance} --disk {disk}"
    return cmd


def create_instances_and_disks_ompgen(name='ompgen', n=10):
    machine = 'c2-standard-60'
    startup_script = f"#!/bin/bash\n" \
                     f"sudo -H -u {GCP_USER} tmux new-session -d -s main\n" \
                     f"source ~/.bashrc\n" \
                     f"sudo chmod 777 -R /home/{GCP_USER}/src/*\n" \
                     f"python /home/{GCP_USER}/src/wzk/wzk/git2.py\n" \
                     f"sudo -H -u {GCP_USER} tmux send 'source /home/{GCP_USER}/src/mogen/mogen/cloud/startup/ompgen.sh' C-m\n" \
                     f"sudo -H -u {GCP_USER} tmux -2  attach-session -t main\n"


                     # f"tmux new-session -d -s main 'source /home/{GCP_USER}/src/mogen/mogen/cloud/startup/ompgen.sh'" \

    # startup_script = f"#! /bin/bash\n touch /home/{GCP_USER}/testtest.txt"
    snapshot = 'tenh-setup'

    instance_list = [f"{GCP_USER_SHORT}-{name}-{i}" for i in range(n)]
    disk_list = [f"{GCP_USER_SHORT}-{name}-disk-{i}" for i in range(n)]

    cmd_disks = []
    cmd_instances = []
    cmd_attach_disks = []
    for i in range(n):
        disk = dict(name=disk_list[i], size=100, labels=GCP_USER_LABEL)
        disk_boot = dict(name=instance_list[i], snapshot=snapshot, size=30, autodelete='yes', boot='yes')
        instance = dict(name=instance_list[i],
                        machine=machine, disks_new=disk_boot, disks_old=[],
                        startup_script=startup_script, labels=GCP_USER_LABEL)

        cmd_disks.append(create_disk_cmd(disk))
        cmd_instances.append(create_instance_cmd(instance))
        cmd_attach_disks.append(attach_disk_cmd(instance=instance, disk=disk))

    with tictoc('Create Disks') as _:
        popen_list(cmd_list=cmd_disks)
    with tictoc('Create Instances') as _:
        popen_list(cmd_list=cmd_instances)
    with tictoc('Attach Disks') as _:
        popen_list(cmd_list=cmd_attach_disks)


def connect_cmd(instance):
    return f'gcloud beta compute ssh --project "{GCP_PROJECT}" --zone "{GCP_ZONE}" {GCP_USER}@"{instance}"'


def copy(src, dst):
    subprocess.call(f"gsutil cp {src} {dst}", shell=True)

# def mount_disk_cmd():
#     return [f"sudo mkfs.ext4 /dev/sdb"
#             f"sudo mount -t ext4 /dev/sdb /home/{GCP_USER}/sdb"]


# def pull_git_cmd():
#     return f"python  /home/{GCP_USER}/src/wzk/wzk.git2.py"


# def tmux_cmd(name='default'):
#     return f"tmux new -s {name}"


# def connect_pull_mount_call(instance, cmd):
#     cmd2 = [connect_cmd(instance=instance),
#             mount_disk_cmd(),
#             pull_git_cmd(),
#             tmux_cmd(),
#             atleast_list(cmd,
#                          convert=False)]
#     cmd2 = flatten(cmd2)
#     # call('; '.join(cmd2), shell=False)
#     run('; '.join(cmd2), shell=False)


def __delete_xyz(_type, names):
    names = atleast_list(names, convert=False)
    cmd = f"yes | gcloud compute {_type} delete {' '.join(names)}"
    subprocess.call(cmd, shell=True)


def delete_instances(instances):
    __delete_xyz(_type='instance', names=instances)


def delete_disks(disks):
    __delete_xyz(_type='disk', names=disks)


def delete_snapshots(snapshots):
    __delete_xyz(_type='snapshots', names=snapshots)


if __name__ == '__main__':
    create_instances_and_disks_ompgen(n=1)
    # connect_pull_mount_call(instance='ompgen-0', cmd=['ls', 'whoami'])


# gcloud compute instances create instance-2
# --project=neon-polymer-214621
# --zone=us-central1-a
# --machine-type=e2-standard-8
# --network-interface=network-tier=PREMIUM,subnet=default
# --metadata=enable-oslogin=true
# --maintenance-policy=MIGRATE
# --service-account=508084122889-compute@developer.gserviceaccount.com
# --scopes=https://www.googleapis.com/auth/devstorage.read_only,https://www.googleapis.com/auth/logging.write,https://www.googleapis.com/auth/monitoring.write,https://www.googleapis.com/auth/servicecontrol,https://www.googleapis.com/auth/service.management.readonly,https://www.googleapis.com/auth/trace.append
# --create-disk=auto-delete=yes,boot=yes,device-name=instance-2,mode=rw,size=30,source-snapshot=projects/neon-polymer-214621/global/snapshots/tenh-setup,type=projects/neon-polymer-214621/zones/us-central1-a/diskTypes/pd-balanced
# --disk=boot=no,device-name=nfs-server,mode=rw,name=nfs-server --reservation-affinity=any


# "gcloud compute project-info add-metadata --metadata enable-oslogin=FALSE"

#
# import typing
# from google.cloud import compute_v1
#
#
# def list_instances(project_id: str, zone: str) -> typing.Iterable[compute_v1.Instance]:
#     """
#     List all instances in the given zone in the specified project.
#
#     Args:
#         project_id: project ID or project number of the Cloud project you want to use.
#         zone: name of the zone you want to use. For example: “us-west3-b”
#     Returns:
#         An iterable collection of Instance objects.
#     """
#     instance_client = compute_v1.InstancesClient()
#     instance_list = instance_client.list(project=project_id, zone=zone)
#
#     print(f"Instances found in zone {zone}:")
#     for instance in instance_list:
#         print(f" - {instance.name} ({instance.machine_type})")
#
#     return instance_list
#


# s = "gcloud compute instances create tenh-ompgen-0 --project=neon-polymer-214621 --zone=us-central1-a --machine-type=e2-medium --network-interface=network-tier=PREMIUM,subnet=default --no-restart-on-failure --maintenance-policy=TERMINATE --preemptible --service-account=508084122889-compute@developer.gserviceaccount.com --disk=boot=no,device-name=tenh-ompgen-disk-0,mode=rw,name=tenh-ompgen-disk-0 --create-disk=auto-delete=yes,boot=yes,device-name=tenh-ompgen-0,mode=rw,size=30,source-snapshot=projects/neon-polymer-214621/global/snapshots/tenh-default-setup,type=projects/neon-polymer-214621/zones/us-central1-a/diskTypes/pd-balanced --labels=user=tenh --reservation-affinity=any"


# print('\n'.join(s.split()))
