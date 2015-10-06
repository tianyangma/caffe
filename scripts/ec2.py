from boto3.session import Session
import argparse, time, json

parser = argparse.ArgumentParser()
parser.add_argument('-u', '--profile', type=str, dest='profile', help='What user profile you want to logon', default='cv')
parser.add_argument('action', help='What you want to do to the instances [list|stop]', nargs='*', default=['list'])
args = parser.parse_args()

def get_ec2(profile):
    session = Session(profile_name=profile)
    ec2 = session.resource('ec2', 'us-west-2')
    return ec2

def list(ec2):
    instances = ec2.instances.all()
    for instance in instances:
        id = instance.id
        status = instance.state['Name']
        tags = instance.tags

        print 'instance ', id, status

def stop(ec2, instance_id):
    instances = ec2.instances.all()
    for instance in instances:
        if instance.id == instance_id:
            resp_raw = instance.stop()
            resp = json.loads(resp_raw)
            if 200 == resp['ResponseMetadata']['HTTPStatusCode']:
                print 'stopping ', instance_id
            return

def parse_action(ec2, action):
    if action[0]=='list':
        list(ec2)
    elif action[0]=='stop':
        stop(ec2, action[1])

if __name__ == '__main__':
    action = args.action
    profile = args.profile

    ec2 = get_ec2(profile)
    parse_action(ec2, action)
