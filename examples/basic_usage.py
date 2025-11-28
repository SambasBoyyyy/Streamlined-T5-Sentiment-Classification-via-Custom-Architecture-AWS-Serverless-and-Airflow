"""
Basic API Usage Example

Shows how to use the deployed T5 API in your own code.
"""

import json
import requests
from pathlib import Path


def summarize_text(text: str) -> str:
    """
    Summarize text using the deployed T5 API.
    
    Args:
        text: Text to summarize
        
    Returns:
        Summary string
    """
    # Load API URL
    with open('.deployment.json') as f:
        deployment_info = json.load(f)
    api_url = deployment_info['api_url']
    
    # Make request
    response = requests.post(
        api_url,
        headers={'Content-Type': 'application/json'},
        json={'text': text},
        timeout=60
    )
    
    if response.status_code == 200:
        result = response.json()
        return result['summary']
    else:
        raise Exception(f"API error: {response.status_code} - {response.text}")


if __name__ == "__main__":
    # Example usage
    article = """
    Machine learning is a method of data analysis that automates analytical model 
    building. It is a branch of artificial intelligence based on the idea that systems 
    can learn from data, identify patterns and make decisions with minimal human 
    intervention. Machine learning algorithms are trained on sample data, known as 
    training data, in order to make predictions or decisions without being explicitly 
    programmed to do so.
    """
    
    print("Summarizing article...")
    summary = summarize_text(article.strip())
    print(f"\nSummary: {summary}")
