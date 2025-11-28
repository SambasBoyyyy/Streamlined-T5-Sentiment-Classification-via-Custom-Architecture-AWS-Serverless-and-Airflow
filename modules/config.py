"""
Configuration Management

Loads and validates configuration from environment variables.
Provides a simple Config object for the entire application.
"""

import os
from dataclasses import dataclass
from dotenv import load_dotenv
import boto3

# Load .env file
load_dotenv()


@dataclass
class Config:
    """Application configuration"""
    
    # AWS Settings
    region: str
    account_id: str
    
    # Project Settings
    project_name: str
    
    # Computed names (based on project_name)
    @property
    def role_name(self) -> str:
        return f"{self.project_name}-role"
    
    @property
    def bucket_name(self) -> str:
        return f"{self.project_name}-models-{self.account_id}"
    
    @property
    def model_name(self) -> str:
        return f"{self.project_name}-model"
    
    @property
    def endpoint_name(self) -> str:
        return f"{self.project_name}-endpoint"
    
    @property
    def lambda_name(self) -> str:
        return f"{self.project_name}-function"
    
    @property
    def api_name(self) -> str:
        return f"{self.project_name}-api"
    
    @property
    def role_arn(self) -> str:
        return f"arn:aws:iam::{self.account_id}:role/{self.role_name}"


def load_config() -> Config:
    """
    Load configuration from environment variables.
    Auto-detects AWS account ID if not provided.
    """
    region = os.getenv('AWS_REGION', 'us-east-1')
    project_name = os.getenv('PROJECT_NAME', 't5-simple')
    
    # Auto-detect account ID
    account_id = os.getenv('AWS_ACCOUNT_ID')
    if not account_id:
        sts = boto3.client('sts', region_name=region)
        account_id = sts.get_caller_identity()['Account']
    
    return Config(
        region=region,
        account_id=account_id,
        project_name=project_name
    )


# Global config instance
config = load_config()
