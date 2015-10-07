#!/usr/bin/python

# aws command line tool is required by boto3 to get the aws connection
# run  <-- brew install awscli --> to install it in mac (don't forget to config it after installing)
# more installation instruction at: http://docs.aws.amazon.com/cli/latest/userguide/installing.html

from boto3.session import Session
import fabric.api as api
import argparse, time, json

parser = argparse.ArgumentParser()
parser.add_argument('-u', '--profile', type=str, dest='profile', help='What user profile you want to logon', default='cv')
parser.add_argument('action', help='What you want to do to the instances [list|stop|start|exec]', nargs='*', default=['list'])
args = parser.parse_args()

class EC2:

    def __init__(self, profile):
        # default cv instance id
        self.instance_id = 'i-29bf18ec'
        self.session = Session(profile_name=profile)
        self.ec2 = self.session.resource('ec2', 'us-west-2')

    def list(self):
        instances = self.ec2.instances.all()
        for instance in instances:
            id = instance.id
            status = instance.state['Name']
            tags = instance.tags

            print 'instance ', id, status

    def stop(self, instance_ids):
        self.ec2.instances.filter(InstanceIds=instance_ids).stop()
        print 'stopping ', instance_ids

    def start(self, instance_ids):
        self.ec2.instances.filter(InstanceIds=instance_ids).start()
        print 'start ', instance_ids

    def status(self, instance_id):
        instance = self.ec2.Instance(instance_id)
        return instance.state['Name']

    def exec_cmd(self, instance_id, cmd):
        instance = self.ec2.Instance(instance_id)
        if instance.state['Name'] != 'running':
            print 'The instance {0} is not running.'.format(instance_id)

        with api.settings(host_string=instance.public_ip_address, user = "ubuntu", key_filename="gpu_instance.pem"):
            ret = api.run(cmd)

    def dispatch(self, action):

        if action[0]=='list':
            self.list()
        elif action[0]=='stop':
            instance_ids = action[1:] if len(action) > 1 else [self.instance_id]
            self.stop(instance_ids)
        elif action[0]=='start':
            instance_ids = action[1:] if len(action) > 1 else [self.instance_id]
            self.start(instance_ids)
        elif action[0]=='exec':
            cmd = ' '.join(action[1:])
            self.exec_cmd(self.instance_id, cmd)

if __name__ == '__main__':
    action = args.action
    profile = args.profile

    ec2 = EC2(profile)
    ec2.dispatch(action)
