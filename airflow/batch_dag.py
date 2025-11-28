"""
Simple Airflow DAG for T5 Batch Processing

This DAG runs daily and processes texts from S3 using the SageMaker endpoint.

Flow:
1. Read texts from S3 (inputs/ folder)
2. Process each text through SageMaker
3. Write summaries to S3 (outputs/ folder)
"""

from datetime import datetime, timedelta
import json
import boto3
from airflow import DAG
from airflow.operators.python import PythonOperator

# Configuration - update these to match your deployment
AWS_REGION = 'us-east-1'
PROJECT_NAME = 't5-simple'
BUCKET_NAME = f'{PROJECT_NAME}-models-ACCOUNT_ID'  # Update with your account ID
ENDPOINT_NAME = f'{PROJECT_NAME}-endpoint'

# S3 paths
INPUT_PREFIX = 'batch-inputs/'
OUTPUT_PREFIX = 'batch-outputs/'


def read_inputs_from_s3(**context):
    """Read input texts from S3"""
    s3 = boto3.client('s3', region_name=AWS_REGION)
    
    # List files in input folder
    response = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=INPUT_PREFIX)
    
    if 'Contents' not in response:
        print("No input files found")
        return []
    
    texts = []
    for obj in response['Contents']:
        if obj['Key'].endswith('.txt'):
            # Read text file
            file_obj = s3.get_object(Bucket=BUCKET_NAME, Key=obj['Key'])
            text = file_obj['Body'].read().decode('utf-8')
            texts.append({
                'text': text,
                'filename': obj['Key'].split('/')[-1]
            })
    
    print(f"Found {len(texts)} texts to process")
    
    # Pass to next task via XCom
    context['task_instance'].xcom_push(key='texts', value=texts)
    return texts


def process_with_sagemaker(**context):
    """Process texts through SageMaker endpoint"""
    # Get texts from previous task
    texts = context['task_instance'].xcom_pull(key='texts', task_ids='read_inputs')
    
    if not texts:
        print("No texts to process")
        return []
    
    sagemaker = boto3.client('sagemaker-runtime', region_name=AWS_REGION)
    results = []
    
    for item in texts:
        print(f"Processing: {item['filename']}")
        
        try:
            # Call SageMaker endpoint
            response = sagemaker.invoke_endpoint(
                EndpointName=ENDPOINT_NAME,
                ContentType='application/json',
                Body=json.dumps({'text': item['text']})
            )
            
            result = json.loads(response['Body'].read())
            
            results.append({
                'filename': item['filename'],
                'original': item['text'],
                'summary': result['summary']
            })
            
        except Exception as e:
            print(f"Error processing {item['filename']}: {e}")
            results.append({
                'filename': item['filename'],
                'error': str(e)
            })
    
    # Pass to next task
    context['task_instance'].xcom_push(key='results', value=results)
    return results


def write_outputs_to_s3(**context):
    """Write summaries to S3"""
    results = context['task_instance'].xcom_pull(key='results', task_ids='process_texts')
    
    if not results:
        print("No results to write")
        return
    
    s3 = boto3.client('s3', region_name=AWS_REGION)
    timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    
    for result in results:
        if 'error' in result:
            continue
        
        # Write summary to S3
        output_key = f"{OUTPUT_PREFIX}{timestamp}_{result['filename']}"
        s3.put_object(
            Bucket=BUCKET_NAME,
            Key=output_key,
            Body=result['summary'],
            ContentType='text/plain'
        )
        print(f"Wrote summary: {output_key}")
    
    print(f"✓ Processed {len(results)} texts")


# Define DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

dag = DAG(
    dag_id='t5_batch_processing',
    default_args=default_args,
    description='Process texts through T5 model',
    schedule_interval='@daily',  # Run once per day
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['t5', 'ml', 'batch']
)

# Define tasks
read_task = PythonOperator(
    task_id='read_inputs',
    python_callable=read_inputs_from_s3,
    dag=dag
)

process_task = PythonOperator(
    task_id='process_texts',
    python_callable=process_with_sagemaker,
    dag=dag
)

write_task = PythonOperator(
    task_id='write_outputs',
    python_callable=write_outputs_to_s3,
    dag=dag
)

# Set task dependencies
read_task >> process_task >> write_task
