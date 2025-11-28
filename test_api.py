"""
API Test Script

Tests the deployed T5 summarization API.
"""

import json
import requests
from pathlib import Path


def test_api():
    """Test the deployed API"""
    
    # Read API URL from deployment info
    deployment_file = Path('.deployment.json')
    if not deployment_file.exists():
        # Try old format
        api_url_file = Path('.api_url')
        if not api_url_file.exists():
            print("❌ No deployment found. Run deploy.py first!")
            return
        api_url = api_url_file.read_text().strip()
    else:
        with open(deployment_file) as f:
            deployment_info = json.load(f)
        api_url = deployment_info['api_url']
    
    print("🧪 Testing T5 Summarization API\n")
    print(f"API URL: {api_url}\n")
    
    # Test text
    test_text = """
    Since it was raining, we canceled the picnic," "The woman who lives next door is a doctor," and "Even though he was late, he still managed to attend the meeting
    """
    
    print(f"Input text:\n{test_text.strip()}\n")
    print("Sending request...")
    
    try:
        response = requests.post(
            api_url,
            headers={'Content-Type': 'application/json'},
            json={'text': test_text.strip()},
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            print("\n✅ Success!")
            print(f"\nSummary:\n{result.get('summary', 'No summary')}\n")
        else:
            print(f"\n❌ Error: {response.status_code}")
            print(response.text)
            
    except requests.exceptions.Timeout:
        print("\n⏱️  Request timed out")
        print("Note: First request may take 30 seconds (cold start)")
        print("Try again in a few seconds")
    except Exception as e:
        print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    test_api()
