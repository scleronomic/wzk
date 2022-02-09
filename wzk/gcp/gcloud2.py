import os
import fire
import socket
import subprocess
import pandas as pd
from io import StringIO

from wzk.subprocess2 import call2, popen_list
from wzk.dicts_lists_tuples import atleast_list

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
    if disks is None:
        return ''
    disks = atleast_list(disks, convert=False)
    cmd = [f"--disk=boot=no,device-name={d['name']},name={d['name']},mode=rw" for d in disks]
    cmd = ' '.join(cmd)
    return cmd


def add_new_disks_flag(disks):
    if disks is None:
        return ''
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


def add_local_disks_flag(disks):
    if disks is None:
        return ''
    interface = disks['interface']  # 'SCSI' or 'NVME'
    n = disks['n']  # 8

    cmd = f"--local-ssd=interface={interface}"
    cmd = ' '.join([cmd]*n)
    return cmd


def get_disks():
    cmd = "gcloud compute disks list"
    s = call2(cmd)
    disks = pd.read_table(StringIO(s), delim_whitespace=True)  # noqa
    return disks


def add_startup_script_flag(startup_script):
    if startup_script is None or startup_script == '':
        return ''
    else:
        return f'--metadata=startup-script="{startup_script}"'


def create_instance_cmd(config):
    cmd = f"gcloud compute instances create {config['name']} " \
          f"--machine-type={config['machine']} " \
          f"{add_new_disks_flag(config['disks_new'])} " \
          f"{add_old_disks_flag(config['disks_old'])} " \
          f"{add_local_disks_flag(config['disks_local'])} " \
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


def __resource2name(resource):
    if isinstance(resource, dict):
        resource = resource['name']
    return resource


def attach_disk_cmd(instance, disk):
    instance = __resource2name(instance)
    disk = __resource2name(disk)
    cmd = f'gcloud compute instances attach-disk {instance} --disk {disk} --zone "{GCP_ZONE}"'
    return cmd


def detach_disk_cmd(instance, disk):
    instance = __resource2name(instance)
    disk = __resource2name(disk)
    cmd = f'gcloud compute instances detach-disk {instance} --disk {disk} --zone "{GCP_ZONE}"'
    return cmd


def mount_disk_cmd(disk, directory):
    return f"sudo mount -t ext4 {disk} {directory}"


def lsblk():
    s = call2('lsblk')
    blk = pd.read_table(StringIO(s), delim_whitespace=True)  # noqa
    return blk


def umount_disk_cmd(disk):
    return f"sudo umount {disk}"


def upload2bucket(disk, file, bucket, n, n0=0):

    instance = socket.gethostname()
    file_name, file_ext = os.path.splitext(os.path.split(file)[1])

    directory = f'/home/{GCP_USER}/sdb'
    for i in range(n0, n+n0):
        disk_i = f"{disk}-{i}"
        file_i = f"{bucket}/{file_name}_{i}{file_ext}"

        blk0 = set(lsblk().NAME.values)
        subprocess.call(attach_disk_cmd(instance=instance, disk=disk_i), shell=True)
        blk1 = set(lsblk().NAME.values)
        sdX = blk1.difference(blk0)
        sdX = list(sdX)[0]
        subprocess.call(mount_disk_cmd(disk=f"/dev/{sdX}", directory=directory), shell=True)
        copy(src=file, dst=file_i)
        subprocess.call(umount_disk_cmd(disk=f"/dev/{sdX}"), shell=True)
        subprocess.call(detach_disk_cmd(instance=instance, disk=disk_i), shell=True)


def connect_cmd(instance):
    return f'gcloud beta compute ssh --project "{GCP_PROJECT}" --zone "{GCP_ZONE}" {GCP_USER}@"{instance}"'


def copy(src, dst):
    subprocess.call(f"gsutil cp {src} {dst}", shell=True)


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


def connect2(name):
    cmd = f'gcloud compute ssh "{name}" --zone "{GCP_ZONE}" --project "{GCP_PROJECT}"'
    subprocess.call(cmd, shell=True)


if __name__ == '__main__':
    fire.Fire({'connect2': connect2})
    
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
