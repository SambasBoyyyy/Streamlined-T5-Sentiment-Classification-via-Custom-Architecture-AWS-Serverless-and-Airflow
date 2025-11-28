"""
Model Deployment

Downloads T5-small from Hugging Face and deploys to SageMaker.
"""

import tempfile
import tarfile
from pathlib import Path
import boto3
from transformers import T5ForConditionalGeneration, T5Tokenizer
from .config import config
from .utils import print_step


def deploy_model(bucket_name: str, role_arn: str) -> str:
    """
    Download T5-small, package it, and deploy to SageMaker.
    
    Args:
        bucket_name: S3 bucket for model storage
        role_arn: IAM role ARN for SageMaker
        
    Returns:
        Endpoint name
    """
    print_step(3, "Deploying T5 Model to SageMaker")
    print("This takes ~10 minutes (downloading model, creating endpoint)")
    
    # Step 1: Download and package model
    print("\n→ Downloading T5-small from Hugging Face...")
    model_s3_uri = _download_and_package_model(bucket_name)
    
    # Step 2: Create SageMaker model
    print("\n→ Creating SageMaker model...")
    _create_sagemaker_model(model_s3_uri, role_arn)
    
    # Step 3: Deploy endpoint
    print("\n→ Deploying serverless endpoint...")
    _deploy_endpoint()
    
    print(f"\n✓ Model deployed to endpoint: {config.endpoint_name}")
    return config.endpoint_name


def _download_and_package_model(bucket_name: str) -> str:
    """Download T5 model and create model.tar.gz"""
    
    with tempfile.TemporaryDirectory() as tmpdir:
        model_dir = Path(tmpdir) / "model"
        model_dir.mkdir()
        
        # Download model and tokenizer
        model = T5ForConditionalGeneration.from_pretrained("t5-small")
        tokenizer = T5Tokenizer.from_pretrained("t5-small")
        
        model.save_pretrained(str(model_dir))
        tokenizer.save_pretrained(str(model_dir))
        print("  Downloaded model files")
        
        # Create inference script
        code_dir = model_dir / "code"
        code_dir.mkdir()
        
        inference_code = '''import json
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

model = None
tokenizer = None
device = None

def model_fn(model_dir):
    """Load model once on startup"""
    global model, tokenizer, device
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INIT] Using device: {device}")
    
    tokenizer = T5Tokenizer.from_pretrained(model_dir)
    model = T5ForConditionalGeneration.from_pretrained(model_dir)
    model.to(device)
    model.eval()
    
    print(f"[INIT] Model loaded successfully on {device}")
    return model

def input_fn(request_body, content_type='application/json'):
    """Parse input"""
    data = json.loads(request_body)
    return data.get('text', '')

def predict_fn(input_data, model):
    """Generate summary"""
    global tokenizer, device
    
    text = f"summarize: {input_data}"
    inputs = tokenizer(text, max_length=512, truncation=True, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        output_ids = model.generate(
            inputs['input_ids'],
            max_length=150,
            num_beams=4,
            early_stopping=True
        )
    
    summary = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return summary

def output_fn(prediction, accept='application/json'):
    """Format output"""
    return json.dumps({"summary": prediction})
'''
        
        (code_dir / "inference.py").write_text(inference_code)
        # Only install sentencepiece - container has torch & transformers pre-installed
        (code_dir / "requirements.txt").write_text("sentencepiece>=0.1.99")
        print("  Created inference script")
        
        # Create tar.gz
        tar_path = Path(tmpdir) / "model.tar.gz"
        with tarfile.open(tar_path, "w:gz") as tar:
            for item in model_dir.rglob("*"):
                if item.is_file():
                    tar.add(item, arcname=item.relative_to(model_dir))
        print("  Packaged model")
        
        # Upload to S3
        s3 = boto3.client('s3', region_name=config.region)
        s3.upload_file(str(tar_path), bucket_name, "model.tar.gz")
        print(f"  Uploaded to s3://{bucket_name}/model.tar.gz")
        
        return f"s3://{bucket_name}/model.tar.gz"


def _create_sagemaker_model(model_s3_uri: str, role_arn: str):
    """Create SageMaker model resource"""
    
    sagemaker = boto3.client('sagemaker', region_name=config.region)
    
    # Use HuggingFace GPU container (optimized for transformers + CUDA)
    image_uri = f"763104351884.dkr.ecr.{config.region}.amazonaws.com/huggingface-pytorch-inference:2.1.0-transformers4.37.0-gpu-py310-cu118-ubuntu20.04"
    
    try:
        sagemaker.create_model(
            ModelName=config.model_name,
            PrimaryContainer={
                'Image': image_uri,
                'ModelDataUrl': model_s3_uri,
                'Environment': {
                    'SAGEMAKER_PROGRAM': 'inference.py',
                    'SAGEMAKER_SUBMIT_DIRECTORY': model_s3_uri
                }
            },
            ExecutionRoleArn=role_arn
        )
        print(f"  Created model: {config.model_name}")
    except Exception as e:
        error_msg = str(e)
        if 'already exists' in error_msg or 'already existing' in error_msg:
            print(f"  Model already exists: {config.model_name}")
        else:
            print(f"\n❌ Error creating SageMaker model:")
            if hasattr(e, 'response'):
                print(f"   Error code: {e.response['Error']['Code']}")
                print(f"   Error message: {e.response['Error']['Message']}")
            else:
                print(f"   Error: {error_msg}")
            print(f"\n   Model S3 URI: {model_s3_uri}")
            print(f"   Role ARN: {role_arn}")
            print(f"   Image URI: {image_uri}")
            raise


def _deploy_endpoint():
    """Deploy serverless endpoint"""
    
    sagemaker = boto3.client('sagemaker', region_name=config.region)
    
    endpoint_config_name = f"{config.endpoint_name}-config"
    
    # Create endpoint configuration
    try:
        sagemaker.create_endpoint_config(
            EndpointConfigName=endpoint_config_name,
            ProductionVariants=[{
                'VariantName': 'AllTraffic',
                'ModelName': config.model_name,
                'InstanceType': 'ml.g4dn.xlarge',  # GPU instance
                'InitialInstanceCount': 1,
                'InitialVariantWeight': 1.0
            }]
        )
        print(f"  Created endpoint config")
    except Exception as e:
        if 'already exists' in str(e).lower() or 'cannot create' in str(e).lower():
            print(f"  Endpoint config already exists")
        else:
            print(f"  Error creating endpoint config: {e}")
            raise
    
    # Create endpoint
    try:
        sagemaker.create_endpoint(
            EndpointName=config.endpoint_name,
            EndpointConfigName=endpoint_config_name
        )
        print(f"  Creating endpoint...")
    except Exception as e:
        if 'already exists' in str(e).lower() or 'cannot create' in str(e).lower():
            print(f"  Endpoint already exists or is being created")
        else:
            print(f"  Error creating endpoint: {e}")
            raise
    
    # Wait for endpoint with better error handling
    print("  Waiting for endpoint (5-10 minutes)...")
    import time
    max_wait = 600  # 10 minutes
    wait_interval = 30  # Check every 30 seconds
    elapsed = 0
    
    while elapsed < max_wait:
        try:
            response = sagemaker.describe_endpoint(EndpointName=config.endpoint_name)
            status = response['EndpointStatus']
            
            if status == 'InService':
                print(f"  Endpoint is ready!")
                return
            elif status == 'Failed':
                failure_reason = response.get('FailureReason', 'Unknown error')
                print(f"\n❌ Endpoint creation failed!")
                print(f"   Reason: {failure_reason}")
                raise Exception(f"Endpoint creation failed: {failure_reason}")
            else:
                print(f"  Status: {status}... ({elapsed}s elapsed)")
                time.sleep(wait_interval)
                elapsed += wait_interval
                
        except sagemaker.exceptions.ClientError as e:
            if 'Could not find endpoint' in str(e):
                # Endpoint not created yet, wait a bit
                time.sleep(wait_interval)
                elapsed += wait_interval
            else:
                raise
    
    raise Exception(f"Endpoint creation timed out after {max_wait} seconds")
