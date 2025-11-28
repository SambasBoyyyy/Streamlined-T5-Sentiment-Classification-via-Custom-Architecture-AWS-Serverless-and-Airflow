# T5 Text Summarization on AWS

Deploy T5-small model to AWS with a clean, modular architecture perfect for learning!

## What You'll Build

An AI-powered text summarization API with:
- **SageMaker**: Serverless model hosting (scales to zero)
- **Lambda**: API request handling
- **API Gateway**: Public HTTPS endpoint
- **Airflow** (optional): Batch processing orchestration

**Cost**: ~$5-10/month (without Airflow), ~$310/month (with Airflow)

## Quick Start

### 1. Setup

```bash
# Configure AWS
aws configure

# Install dependencies
cd t5-aws-mlops-pipeline
cp .env.template .env
pip install -r requirements.txt
```

### 2. Deploy (Choose One)

**Option A: Without Airflow (Recommended for Learning)**
```bash
python deploy.py
```
Takes ~20 minutes. Cost: ~$5-10/month.

**Option B: With Airflow (Production Batch Processing)**
```bash
python deploy_with_airflow.py
```
Takes ~50 minutes. Cost: ~$310/month (includes MWAA).

### 3. Test

```bash
# Test your API
python test_api.py
```

### 4. Use

```python
import requests

response = requests.post(
    "YOUR_API_URL",
    json={"text": "Your long text here..."}
)
print(response.json()["summary"])
```

## Project Structure

```
t5-aws-mlops-pipeline/
├── deploy.py                    # Deploy without Airflow
├── deploy_with_airflow.py       # Deploy with Airflow (optional)
├── cleanup.py                   # Delete all resources
├── test_api.py                  # Test the API
│
├── modules/                     # Core modules
│   ├── config.py               # Configuration
│   ├── aws_setup.py            # IAM & S3
│   ├── model.py                # SageMaker deployment
│   ├── api.py                  # Lambda & API Gateway
│   ├── airflow_setup.py        # Airflow setup (optional)
│   └── utils.py                # Helpers
│
├── airflow/                     # Airflow DAGs (optional)
│   └── batch_dag.py            # Batch processing DAG
│
└── examples/                    # Usage examples
    ├── basic_usage.py          # Simple example
    └── batch_process.py        # Batch without Airflow
```

## Learning Path

1. **Read `deploy.py`** - See the deployment flow
2. **Read each module** - Understand each AWS service
3. **Run deployment** - Watch it work
4. **Check examples** - Learn how to use it
5. **Try Airflow** (optional) - Learn workflow orchestration

## Airflow (Optional)

### When to Use Airflow

✅ **Use Airflow if you need:**
- Scheduled batch processing (daily, hourly, etc.)
- Complex workflow dependencies
- Production-grade job orchestration
- Monitoring and retry logic

❌ **Skip Airflow if you:**
- Just want to learn the basics
- Process batches occasionally (use `examples/batch_process.py`)
- Want to minimize costs

### Airflow Costs

- **MWAA**: ~$300/month (mw1.small environment)
- **Total with Airflow**: ~$310/month

### How Airflow Works

1. **DAG** (`airflow/batch_dag.py`): Defines the workflow
2. **Tasks**: Read from S3 → Process with SageMaker → Write to S3
3. **Schedule**: Runs daily (configurable)
4. **Monitoring**: Airflow UI shows task status

### Deploy with Airflow

```bash
python deploy_with_airflow.py
```

This will:
1. Deploy everything (SageMaker, Lambda, API Gateway)
2. Ask if you want MWAA (confirm to proceed)
3. Create MWAA environment (20-30 minutes)
4. Upload the batch processing DAG

## Key Features

✅ **Modular**: Each file has one clear job  
✅ **Well-commented**: Learn WHY, not just WHAT  
✅ **Complete**: Full SageMaker + Lambda + API Gateway + Airflow  
✅ **Flexible**: Use with or without Airflow  
✅ **Simple**: No over-engineering  

## Cleanup

```bash
# Delete all AWS resources
python cleanup.py
```

Note: MWAA environment must be deleted separately:
```bash
aws mwaa delete-environment --name t5-simple-airflow
```

## Troubleshooting

**"Access Denied"**
```bash
aws sts get-caller-identity  # Check credentials
```

**First request slow (30s)**
- Normal! It's a "cold start"
- Next requests are fast (2-3s)

**Airflow DAG not showing**
- Wait 5-10 minutes for sync
- Check S3: `s3://BUCKET/airflow/dags/batch_dag.py`
- Update BUCKET_NAME in `batch_dag.py`

## What's Included

- ✅ Full AWS deployment
- ✅ Serverless architecture
- ✅ Public API endpoint
- ✅ Optional Airflow orchestration
- ✅ Usage examples
- ✅ Clean, learnable code

## Next Steps

- Try different Hugging Face models
- Add authentication to API
- Create custom Airflow DAGs
- Build a web interface

---

**Perfect for learning AWS + ML deployment!** 🚀
