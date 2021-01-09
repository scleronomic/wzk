import subprocess


# SSH
def execute_via_ssh(remote_client, cmd):
    # ssh -tt for short to force pseudo-tty allocation even if stdin isn't a terminal.
    # https://stackoverflow.com/questions/7114990/pseudo-terminal-will-not-be-allocated-because-stdin-is-not-a-terminal
    process = subprocess.Popen(f"ssh {remote_client} -tt; {cmd}", stdout=subprocess.PIPE,  shell=True)
    process = subprocess.Popen(f"{cmd}; {cmd}", stdout=subprocess.PIPE,  shell=True)
    stdout = process.communicate()[0].strip()
    return stdout

def test_execute_via_ssh():
    print(execute_via_ssh('rmc-lx0144', 'ls'))

