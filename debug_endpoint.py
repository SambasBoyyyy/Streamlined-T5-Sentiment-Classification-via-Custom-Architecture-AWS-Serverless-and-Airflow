import boto3
import json
import time
from modules.config import config

def debug_endpoint():
    print(f"Testing endpoint: {config.endpoint_name}")
    client = boto3.client('sagemaker-runtime', region_name=config.region)
    
    text = """
    The Amazon rainforest is a moist broadleaf tropical rainforest in the Amazon biome
    that covers most of the Amazon basin of South America. This basin encompasses
    7,000,000 km2, of which 5,500,000 km2 are covered by the rainforest. This region
    includes territory belonging to nine nations and 3,344 formally acknowledged
    indigenous territories.
    """
    
    payload = {"text": text}
    
    print("Invoking endpoint directly...")
    start_time = time.time()
    
    try:
        response = client.invoke_endpoint(
            EndpointName=config.endpoint_name,
            ContentType='application/json',
            Body=json.dumps(payload)
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        result = json.loads(response['Body'].read().decode())
        print(f"\n✅ Success! Duration: {duration:.2f} seconds")
        print(f"Summary: {result}")
        
    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        print(f"\n❌ Failed! Duration: {duration:.2f} seconds")
        print(f"Error: {e}")

if __name__ == "__main__":
    debug_endpoint()
