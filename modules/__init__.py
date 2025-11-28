"""Modules for T5 AWS deployment"""

from .config import config, load_config
from .aws_setup import create_iam_role, create_s3_bucket
from .model import deploy_model
from .api import create_lambda, create_api_gateway
from .utils import print_step, save_deployment_info

# Optional Airflow module
try:
    from .airflow_setup import setup_airflow, upload_dag
    __all__ = [
        'config',
        'load_config',
        'create_iam_role',
        'create_s3_bucket',
        'deploy_model',
        'create_lambda',
        'create_api_gateway',
        'print_step',
        'save_deployment_info',
        'setup_airflow',
        'upload_dag'
    ]
except ImportError:
    __all__ = [
        'config',
        'load_config',
        'create_iam_role',
        'create_s3_bucket',
        'deploy_model',
        'create_lambda',
        'create_api_gateway',
        'print_step',
        'save_deployment_info'
    ]
