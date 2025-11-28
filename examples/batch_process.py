"""
Batch Processing Example

Shows how to process multiple texts efficiently.
"""

import json
import requests
import time
from pathlib import Path
from typing import List


def summarize_batch(texts: List[str]) -> List[str]:
    """
    Summarize multiple texts.
    
    Args:
        texts: List of texts to summarize
        
    Returns:
        List of summaries
    """
    # Load API URL
    with open('.deployment.json') as f:
        deployment_info = json.load(f)
    api_url = deployment_info['api_url']
    
    summaries = []
    
    for i, text in enumerate(texts, 1):
        print(f"Processing {i}/{len(texts)}...")
        
        try:
            response = requests.post(
                api_url,
                headers={'Content-Type': 'application/json'},
                json={'text': text},
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                summaries.append(result['summary'])
            else:
                summaries.append(f"Error: {response.status_code}")
            
            # Small delay to avoid rate limiting
            time.sleep(0.5)
            
        except Exception as e:
            summaries.append(f"Error: {str(e)}")
    
    return summaries


if __name__ == "__main__":
    # Example: Summarize multiple articles
    articles = [
        "Climate change includes both global warming driven by human-induced emissions of greenhouse gases and the resulting large-scale shifts in weather patterns.",
        "Quantum computing is the use of quantum phenomena such as superposition and entanglement to perform computation.",
        "Artificial intelligence is intelligence demonstrated by machines, as opposed to natural intelligence displayed by animals including humans."
    ]
    
    print(f"Processing {len(articles)} articles...\n")
    summaries = summarize_batch(articles)
    
    print("\nResults:")
    for i, (article, summary) in enumerate(zip(articles, summaries), 1):
        print(f"\n{i}. Original: {article[:60]}...")
        print(f"   Summary: {summary}")
