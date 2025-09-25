#!/usr/bin/env python3
"""
S3 Big Data Setup Script
Downloads and uploads NYC taxi data files to CloudFormation-created S3 bucket
"""

import boto3
import requests
import os
import re
from pathlib import Path
from botocore.exceptions import ClientError

def get_cfn_stack_outputs(stack_name):
    """Get CloudFormation stack outputs"""
    try:
        cf_client = boto3.client('cloudformation')
        response = cf_client.describe_stacks(StackName=stack_name)
        
        if not response['Stacks']:
            print(f"❌ Stack {stack_name} not found")
            return None
            
        outputs = response['Stacks'][0].get('Outputs', [])
        output_dict = {}
        
        for output in outputs:
            output_dict[output['OutputKey']] = output['OutputValue']
            
        return output_dict
        
    except ClientError as e:
        if 'does not exist' in str(e):
            print(f"❌ CloudFormation stack {stack_name} does not exist")
            print("💡 Please deploy the infrastructure first using deploy_cfn.py")
        else:
            print(f"❌ Error getting stack outputs: {e}")
        return None
    except Exception as e:
        print(f"❌ Error getting stack outputs: {e}")
        return None

def verify_bucket_exists(bucket_name):
    """Verify that the S3 bucket exists and is accessible"""
    s3_client = boto3.client('s3')
    
    try:
        s3_client.head_bucket(Bucket=bucket_name)
        print(f"✅ Verified bucket exists: {bucket_name}")
        return True
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == '404':
            print(f"❌ Bucket {bucket_name} does not exist")
        elif error_code == '403':
            print(f"❌ Access denied to bucket {bucket_name}")
        else:
            print(f"❌ Error accessing bucket {bucket_name}: {e}")
        return False

def download_file(url, local_path):
    """Download file from URL to local path"""
    try:
        print(f"📥 Downloading {url}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        file_size = os.path.getsize(local_path) / (1024 * 1024)  # MB
        print(f"✅ Downloaded {local_path.name} ({file_size:.1f} MB)")
        return True
    except Exception as e:
        print(f"❌ Error downloading {url}: {e}")
        return False

def derive_s3_prefix_from_filename(filename):
    """
    Derive S3 prefix from filename pattern: {taxi_class}_tripdata_{year}-{month}.parquet
    Returns: {taxi_class}_tripdata/year={year}/month={month}/
    """
    # Pattern to match: {taxi_class}_tripdata_{year}-{month}.parquet
    pattern = r'^([^_]+)_tripdata_(\d{4})-(\d{2})\.parquet$'
    match = re.match(pattern, filename)
    
    if match:
        taxi_class, year, month = match.groups()
        prefix = f"taxi_class={taxi_class}/year={year}/month={month}/"
        return prefix
    else:
        # Fallback to original filename if pattern doesn't match
        print(f"⚠️  Filename {filename} doesn't match expected pattern, using default prefix")
        return "data/"

def upload_to_s3(local_path, bucket_name, s3_key):
    """Upload file to S3"""
    s3_client = boto3.client('s3')
    
    try:
        print(f"📤 Uploading {local_path.name} to s3://{bucket_name}/{s3_key}...")
        s3_client.upload_file(str(local_path), bucket_name, s3_key)
        print(f"✅ Uploaded {s3_key}")
        return True
    except Exception as e:
        print(f"❌ Error uploading {local_path.name}: {e}")
        return False

def prepare_data(stack_name, suffix, file_urls, table_name):
    """Main execution function"""
    print("🚀 Starting S3 Big Data Setup...")
    
    # Default stack name if not provided
    if not stack_name:
        stack_name = f"big-data-agent-infrastructure-{suffix}"
    
    print(f"📋 Using CloudFormation stack: {stack_name}")
    
    # Get bucket name from CloudFormation stack outputs
    stack_outputs = get_cfn_stack_outputs(stack_name)
    if not stack_outputs:
        print("❌ Could not retrieve CloudFormation stack outputs")
        print("💡 Make sure you've deployed the infrastructure using deploy_cfn.py first")
        return
    
    bucket_name = stack_outputs.get('BucketName')
    if not bucket_name:
        print("❌ BucketName not found in stack outputs")
        print("📊 Available outputs:", list(stack_outputs.keys()))
        return
    
    print(f"🪣 Using bucket from CloudFormation: {bucket_name}")
    
    # Create temp directory for downloads
    temp_dir = Path("temp_downloads")
    temp_dir.mkdir(exist_ok=True)
    
    try:
        # Download and upload each file
        for url in file_urls:
            filename = url.split('/')[-1]
            local_path = temp_dir / filename
            
            # Download file
            if download_file(url, local_path):
                # Derive S3 prefix from filename pattern
                derived_prefix = derive_s3_prefix_from_filename(filename)
                s3_key = f"{table_name}/{derived_prefix}{filename}"
                upload_to_s3(local_path, bucket_name, s3_key)
                
                # Clean up local file
                local_path.unlink()
    
    finally:
        # Clean up temp directory
        if temp_dir.exists():
            temp_dir.rmdir()
    
    print(f"🎉 Setup complete! Bucket: {bucket_name}")
    print(f"📁 Files uploaded with partitioned structure based on filename patterns")