"""
Cleanup Script

Deletes all AWS resources created by deploy.py
"""

import boto3
from modules.config import config


def cleanup():
    """Delete all AWS resources"""
    
    print("\n🧹 Cleaning up AWS resources...\n")
    
    try:
        # Delete SageMaker endpoint
        sagemaker = boto3.client('sagemaker', region_name=config.region)
        
        print(f"Deleting SageMaker endpoint: {config.endpoint_name}")
        try:
            sagemaker.delete_endpoint(EndpointName=config.endpoint_name)
            print("✓ Endpoint deleted")
        except:
            print("  (not found)")
        
        # Delete endpoint config
        try:
            sagemaker.delete_endpoint_config(
                EndpointConfigName=f"{config.endpoint_name}-config"
            )
            print("✓ Endpoint config deleted")
        except:
            pass
        
        # Delete model
        try:
            sagemaker.delete_model(ModelName=config.model_name)
            print("✓ Model deleted")
        except:
            pass
        
        # Delete Lambda function
        lambda_client = boto3.client('lambda', region_name=config.region)
        
        print(f"\nDeleting Lambda function: {config.lambda_name}")
        try:
            lambda_client.delete_function(FunctionName=config.lambda_name)
            print("✓ Lambda deleted")
        except:
            print("  (not found)")
        
        # Delete API Gateway
        apigateway = boto3.client('apigateway', region_name=config.region)
        
        print(f"\nDeleting API Gateway: {config.api_name}")
        apis = apigateway.get_rest_apis()
        for api in apis['items']:
            if api['name'] == config.api_name:
                apigateway.delete_rest_api(restApiId=api['id'])
                print("✓ API Gateway deleted")
        
        # Delete S3 bucket
        s3 = boto3.client('s3', region_name=config.region)
        
        print(f"\nDeleting S3 bucket: {config.bucket_name}")
        try:
            # Delete all objects first
            objects = s3.list_objects_v2(Bucket=config.bucket_name)
            if 'Contents' in objects:
                for obj in objects['Contents']:
                    s3.delete_object(Bucket=config.bucket_name, Key=obj['Key'])
            
            # Delete bucket
            s3.delete_bucket(Bucket=config.bucket_name)
            print("✓ S3 bucket deleted")
        except:
            print("  (not found)")
        
        print("\n✅ Cleanup complete!")
        print("\nNote: IAM role was not deleted (requires manual deletion)")
        print(f"To delete role manually:")
        print(f"  aws iam detach-role-policy --role-name {config.role_name} --policy-arn <policy-arn>")
        print(f"  aws iam delete-role --role-name {config.role_name}")
        
    except Exception as e:
        print(f"\n❌ Cleanup error: {e}")


if __name__ == "__main__":
    cleanup()
