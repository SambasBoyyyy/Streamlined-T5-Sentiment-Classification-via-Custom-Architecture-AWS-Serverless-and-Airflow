"""
Utility Functions

Helper functions for printing and saving deployment info.
"""

import json
from pathlib import Path


def print_step(step_num: int, title: str):
    """
    Print a nice step header.
    
    Args:
        step_num: Step number
        title: Step title
    """
    print(f"\n{'='*60}")
    print(f"STEP {step_num}: {title}")
    print(f"{'='*60}\n")


def save_deployment_info(endpoint_name: str, api_url: str):
    """
    Save deployment information to a JSON file.
    
    Args:
        endpoint_name: SageMaker endpoint name
        api_url: API Gateway URL
    """
    info = {
        'endpoint_name': endpoint_name,
        'api_url': api_url
    }
    
    # Save to .deployment.json
    with open('.deployment.json', 'w') as f:
        json.dump(info, f, indent=2)
    
    # Also save API URL to .api_url for backward compatibility
    with open('.api_url', 'w') as f:
        f.write(api_url)
    
    print(f"\n✓ Deployment info saved to .deployment.json")
