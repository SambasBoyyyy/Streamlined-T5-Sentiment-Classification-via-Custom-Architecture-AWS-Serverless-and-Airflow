"""
Deploy with Airflow (Optional)

This script deploys everything including MWAA for batch orchestration.

WARNING: MWAA costs ~$300/month!
Only use if you need production-grade batch job scheduling.

For simple batch processing, use examples/batch_process.py instead.
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
from modules.airflow_setup import setup_airflow, upload_dag


def main():
    """Main deployment flow with Airflow"""
    
    print("\n🚀 T5 AWS Deployment with Airflow")
    print(f"   Region: {config.region}")
    print(f"   Project: {config.project_name}\n")
    
    print("⚠️  This includes MWAA which costs ~$300/month!")
    print("⚠️  For learning without Airflow, use deploy.py instead\n")
    
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
        
        # Step 5: Setup Airflow (optional)
        airflow_info = setup_airflow(role_arn)
        
        if airflow_info:
            # Upload DAG
            print("\n→ Uploading Airflow DAG...")
            upload_dag('airflow/batch_dag.py')
            print("✓ DAG uploaded")
            print("\nNote: Update BUCKET_NAME in batch_dag.py with your account ID")
        
        # Save deployment info
        save_deployment_info(endpoint_name, api_url)
        
        # Success!
        print("\n" + "="*60)
        print("🎉 DEPLOYMENT COMPLETE!")
        print("="*60)
        print(f"\nYour API URL:")
        print(f"  {api_url}")
        
        if airflow_info:
            print(f"\nAirflow Environment:")
            print(f"  Name: {airflow_info['environment_name']}")
            print(f"  Status: Creating (check AWS Console)")
            print(f"  Access Airflow UI after it's AVAILABLE (~30 min)")
        
        print(f"\nTest it:")
        print(f"  python test_api.py")
        print()
        
    except Exception as e:
        print(f"\n❌ Deployment failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
