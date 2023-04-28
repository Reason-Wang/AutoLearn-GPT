import paramiko

class SSHClient:
    def __init__(self, hostname, port, username, password):
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(hostname=hostname, port=port, username=username, password=password)
        self.ssh = ssh

    def exec_command(self, cmd: str):
        return self.ssh.exec_command(cmd)
