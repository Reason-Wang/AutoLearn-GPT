import paramiko


class SSHClient:
    def __init__(self, hostname, port, username, password, working_space: str=""):
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(hostname=hostname, port=port, username=username, password=password)
        self.ssh = ssh
        self.scp = ssh.open_sftp()
        self.working_space = working_space

    def exec_command(self, cmd: str):
        return self.ssh.exec_command(cmd)

    def get_file(self, file_name: str):
        self.scp.get(self.working_space+"AutoLearn-GPT/"+file_name, file_name)

    def send_file(self, file_name: str):
        self.scp.put(file_name, self.working_space+"AutoLearn-GPT/"+file_name)

    def close(self):
        self.ssh.close()
        self.scp.close()


class LocalClient:
    def __init__(self, working_space: str=""):
        self.working_space = working_space

    def get_file(self, file_name: str):
        pass

    def send_file(self, file_name: str):
        pass

    def close(self):
        pass

