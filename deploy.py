"""
T5 AWS Deployment - Main Orchestrator

This script coordinates the entire deployment process:
1. Setup AWS infrastructure (IAM, S3)
2. Deploy T5 model to SageMaker
3. Create Lambda function
4. Setup API Gateway

Run this to deploy everything!
"""

import sys
from modules import (
    config,
    create_iam_role,
    create_s3_bucket,
    deploy_model,
    create_lambda,
    create_api_gateway,
    save_deployment_info
)


def main():
    """Main deployment flow"""
    
    print("\n🚀 T5 AWS Deployment Starting...")
    print(f"   Region: {config.region}")
    print(f"   Project: {config.project_name}\n")
    
    try:
        # Step 1: Setup AWS infrastructure
        role_arn = create_iam_role()
        bucket_name = create_s3_bucket()
        
        # Step 2: Deploy model to SageMaker
        endpoint_name = deploy_model(bucket_name, role_arn)
        
        # Step 3: Create Lambda function
        lambda_arn = create_lambda(role_arn, endpoint_name)
        
        # Step 4: Setup API Gateway
        api_url = create_api_gateway(lambda_arn)
        
        # Save deployment info
        save_deployment_info(endpoint_name, api_url)
        
        # Success!
        print("\n" + "="*60)
        print("🎉 DEPLOYMENT COMPLETE!")
        print("="*60)
        print(f"\nYour API URL:")
        print(f"  {api_url}")
        print(f"\nTest it:")
        print(f"  python test_api.py")
        print(f"\nOr use curl:")
        print(f'  curl -X POST {api_url} \\')
        print(f'    -H "Content-Type: application/json" \\')
        print(f'    -d \'{{"text": "Your text here"}}\'')
        print()
        
    except Exception as e:
        print(f"\n❌ Deployment failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
