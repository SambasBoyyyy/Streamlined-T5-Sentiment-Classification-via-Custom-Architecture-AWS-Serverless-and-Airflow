"""
Airflow/MWAA Setup Module (Optional)

Creates a managed Airflow environment for batch processing orchestration.

WARNING: MWAA costs ~$300/month minimum. Only use if you need:
- Scheduled batch processing
- Complex workflow orchestration
- Production-grade job scheduling

For simple batch processing, use examples/batch_process.py instead.
"""

import time
import boto3
from botocore.exceptions import ClientError
from .config import config
from .utils import print_step


def setup_airflow(role_arn: str) -> dict:
    """
    Setup MWAA environment (optional).
    
    WARNING: This costs ~$300/month!
    
    Args:
        role_arn: IAM role ARN (must have MWAA permissions)
        
    Returns:
        Dict with environment info
    """
    print_step("OPTIONAL", "Setting up Airflow (MWAA)")
    print("⚠️  WARNING: MWAA costs ~$300/month minimum!")
    print("⚠️  Only proceed if you need production batch orchestration\n")
    
    response = input("Continue with MWAA setup? (yes/no): ")
    if response.lower() not in ['yes', 'y']:
        print("Skipping MWAA setup")
        return None
    
    # Setup network infrastructure
    vpc_id, subnet_ids, sg_id = _setup_network()
    
    # Create MWAA environment
    env_name = _create_mwaa_environment(vpc_id, subnet_ids, sg_id, role_arn)
    
    return {
        'environment_name': env_name,
        'vpc_id': vpc_id,
        'subnet_ids': subnet_ids,
        'security_group_id': sg_id
    }


def _setup_network():
    """Setup VPC, subnets, and security group for MWAA"""
    print("\n→ Setting up network infrastructure...")
    
    ec2 = boto3.client('ec2', region_name=config.region)
    
    # Use default VPC or create new one
    vpcs = ec2.describe_vpcs(Filters=[{'Name': 'isDefault', 'Values': ['true']}])
    
    if vpcs['Vpcs']:
        vpc_id = vpcs['Vpcs'][0]['VpcId']
        print(f"  Using default VPC: {vpc_id}")
    else:
        # Create VPC
        vpc = ec2.create_vpc(CidrBlock='10.0.0.0/16')
        vpc_id = vpc['Vpc']['VpcId']
        ec2.modify_vpc_attribute(VpcId=vpc_id, EnableDnsHostnames={'Value': True})
        print(f"  Created VPC: {vpc_id}")
    
    # Get or create subnets (MWAA needs at least 2)
    subnets = ec2.describe_subnets(Filters=[{'Name': 'vpc-id', 'Values': [vpc_id]}])
    
    if len(subnets['Subnets']) >= 2:
        subnet_ids = [s['SubnetId'] for s in subnets['Subnets'][:2]]
        print(f"  Using existing subnets: {subnet_ids}")
    else:
        # Create subnets
        azs = ec2.describe_availability_zones()['AvailabilityZones'][:2]
        subnet_ids = []
        
        for i, az in enumerate(azs):
            subnet = ec2.create_subnet(
                VpcId=vpc_id,
                CidrBlock=f'10.0.{i}.0/24',
                AvailabilityZone=az['ZoneName']
            )
            subnet_ids.append(subnet['Subnet']['SubnetId'])
        print(f"  Created subnets: {subnet_ids}")
    
    # Create security group
    sg_name = f'{config.project_name}-mwaa-sg'
    
    try:
        sg = ec2.create_security_group(
            GroupName=sg_name,
            Description='Security group for MWAA',
            VpcId=vpc_id
        )
        sg_id = sg['GroupId']
        
        # Add self-referencing rule (required by MWAA)
        ec2.authorize_security_group_ingress(
            GroupId=sg_id,
            IpPermissions=[{
                'IpProtocol': '-1',
                'UserIdGroupPairs': [{'GroupId': sg_id}]
            }]
        )
        print(f"  Created security group: {sg_id}")
        
    except ClientError as e:
        if 'already exists' in str(e):
            # Get existing SG
            sgs = ec2.describe_security_groups(
                Filters=[
                    {'Name': 'group-name', 'Values': [sg_name]},
                    {'Name': 'vpc-id', 'Values': [vpc_id]}
                ]
            )
            sg_id = sgs['SecurityGroups'][0]['GroupId']
            print(f"  Using existing security group: {sg_id}")
        else:
            raise
    
    return vpc_id, subnet_ids, sg_id


def _create_mwaa_environment(vpc_id: str, subnet_ids: list, sg_id: str, role_arn: str) -> str:
    """Create MWAA environment"""
    print("\n→ Creating MWAA environment...")
    print("  This takes 20-30 minutes...")
    
    mwaa = boto3.client('mwaa', region_name=config.region)
    env_name = f"{config.project_name}-airflow"
    
    # Check if exists
    try:
        env = mwaa.get_environment(Name=env_name)
        print(f"  Environment already exists: {env_name}")
        print(f"  Status: {env['Environment']['Status']}")
        return env_name
    except ClientError as e:
        if 'ResourceNotFoundException' not in str(e):
            raise
    
    # Create environment
    try:
        mwaa.create_environment(
            Name=env_name,
            ExecutionRoleArn=role_arn,
            SourceBucketArn=f"arn:aws:s3:::{config.bucket_name}",
            DagS3Path='airflow/dags',
            NetworkConfiguration={
                'SubnetIds': subnet_ids,
                'SecurityGroupIds': [sg_id]
            },
            EnvironmentClass='mw1.small',  # Smallest/cheapest
            MaxWorkers=2,
            AirflowVersion='2.7.2',
            LoggingConfiguration={
                'DagProcessingLogs': {'Enabled': True, 'LogLevel': 'INFO'},
                'TaskLogs': {'Enabled': True, 'LogLevel': 'INFO'}
            },
            WebserverAccessMode='PUBLIC_ONLY'
        )
        
        print(f"  Environment creation started: {env_name}")
        print("  ⏳ This will take 20-30 minutes to complete")
        print("  Check status: aws mwaa get-environment --name", env_name)
        
    except ClientError as e:
        print(f"  Error: {e}")
        raise
    
    return env_name


def upload_dag(dag_file: str):
    """
    Upload Airflow DAG to S3.
    
    Args:
        dag_file: Path to DAG Python file
    """
    print_step("AIRFLOW", "Uploading DAG")
    
    s3 = boto3.client('s3', region_name=config.region)
    
    # Upload DAG
    dag_key = f"airflow/dags/{dag_file.split('/')[-1]}"
    s3.upload_file(dag_file, config.bucket_name, dag_key)
    
    print(f"✓ Uploaded DAG to s3://{config.bucket_name}/{dag_key}")
    print("  Wait 5-10 minutes for Airflow to sync the DAG")
