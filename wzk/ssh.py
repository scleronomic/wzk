import subprocess


# SSH
def execute_via_ssh(remote_client, cmd):
    # ssh -tt for short to force pseudo-tty allocation even if stdin isn't a terminal.
    # https://stackoverflow.com/questions/7114990/pseudo-terminal-will-not-be-allocated-because-stdin-is-not-a-terminal
    ssh_process = subprocess.Popen(['ssh', remote_client, '-tt'],
                                   stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   universal_newlines=True, bufsize=0)
    ssh_process.stdin.write(cmd + '\n')
    ssh_process.stdin.close()


