import boto3
from pprint import pprint
import pandas as pd

# Helper function definition
from retrying import retry
import boto3
import json

import time
from botocore.exceptions import ClientError





######################################################################
#notebook 1.1
######################################################################
# Prerequisites utility functions for advanced_rag_utils.py

import boto3
import json
import os
import shutil
import time
import warnings
from urllib.request import urlretrieve
import requests
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth, RequestError

def suppress_warnings():
    """Suppress warnings for cleaner output."""
    warnings.filterwarnings("ignore")

def update_packages(packages=None):
    """
    Update specified Python packages.
    
    Args:
        packages (list): List of packages to update. Defaults to ["boto3", "opensearch-py"].
    """
    import subprocess
    if packages is None:
        packages = ["boto3", "opensearch-py"]
    
    subprocess.check_call(["pip", "install", "-U"] + packages + ["2>/dev/null"])

def get_aws_account_info(region_name="us-west-2"):
    """
    Retrieve AWS account information.
    
    Args:
        region_name (str): AWS region name.
        
    Returns:
        dict: AWS account information including account number and role ARN.
    """
    # Initialize Boto3 session
    boto3_session = boto3.session.Session()
    credentials = boto3_session.get_credentials()
    
    # Retrieve AWS account details
    sts_client = boto3_session.client("sts")
    account_number = sts_client.get_caller_identity()["Account"]
    role_arn = sts_client.get_caller_identity()["Arn"]
    
    # Set up authentication for OpenSearch
    awsauth = AWSV4SignerAuth(credentials, region_name, "aoss")
    
    return {
        "account_number": account_number,
        "role_arn": role_arn,
        "credentials": credentials,
        "awsauth": awsauth
    }

def get_resource_names(account_number, region_name="us-west-2"):
    """
    Generate resource names for the workshop.
    
    Args:
        account_number (str): AWS account number.
        region_name (str): AWS region name.
        
    Returns:
        dict: Resource names.
    """
    s3_bucket_name = f"{account_number}-{region_name}-advanced-rag-workshop"
    knowledge_base_name_aoss = "advanced-rag-workshop-knowledgebase-aoss"
    knowledge_base_name_graphrag = "advanced-rag-workshop-knowledgebase-graphrag"
    oss_vector_store_name = "advancedrag"
    oss_index_name = "ws-index-"
    
    
    return {
        "s3_bucket_name": s3_bucket_name,
        "knowledge_base_name_aoss": knowledge_base_name_aoss,
        "knowledge_base_name_graphrag": knowledge_base_name_graphrag,
        "oss_vector_store_name": oss_vector_store_name,
        "oss_index_name": oss_index_name
    }

def create_bedrock_execution_role(role_handler, s3_bucket_name):
    """
    Create or retrieve a Bedrock execution role.
    
    Args:
        role_handler: IAM role handler instance.
        s3_bucket_name (str): S3 bucket name.
        
    Returns:
        str: Bedrock execution role ARN.
    """
    import boto3
    from botocore.exceptions import ClientError
    
    # Initialize IAM client
    iam_client = boto3.client("iam")
    role_name = f"advanced-rag-workshop-bedrock_execution_role-{role_handler.region_name}"
    bedrock_kb_execution_role_arn = ""
    
    
    try:
        # Try to get the existing role
        existing_role = iam_client.get_role(RoleName=role_name)
        bedrock_kb_execution_role_arn = existing_role["Role"]["Arn"]
        print(f"Policy and roles have been created already. ARN: {bedrock_kb_execution_role_arn}")
    except Exception as e:
        if "NoSuchEntity" in str(e):
            try:
                # Role does not exist, create it
                bedrock_kb_execution_role = role_handler.create_bedrock_execution_role(s3_bucket_name)
                bedrock_kb_execution_role_arn = bedrock_kb_execution_role["Role"]["Arn"]
                print(f"Created Bedrock Knowledge Base Execution Role ARN: {bedrock_kb_execution_role_arn}")
            except Exception as e:
                print(e)
                print("Policies already exist. Please clean them up first.")
        else:
            # Handle other client errors
            print("Policy and roles have been created already.")
    
    if not bedrock_kb_execution_role_arn:
        print("WARNING: Could not determine the Bedrock KB execution role ARN.")
        bedrock_kb_execution_role_arn = f"arn:aws:iam::{role_handler.account_number}:role/{role_name}"
    
    return bedrock_kb_execution_role_arn


def create_cloudwatch_log_group(log_group_name, region_name='us-west-2', retention_days=None, tags=None):
    """
    Creates a CloudWatch log group with the specified configuration.
    
    Args:
        log_group_name (str): The name of the log group to create
        region_name (str, optional): AWS region name. If None, uses the default region from session
        retention_days (int, optional): Number of days to retain logs. If None, logs never expire
            Valid values: 1, 3, 5, 7, 14, 30, 60, 90, 120, 150, 180, 365, 400, 545, 731, 1827, 3653
        tags (dict, optional): Tags to apply to the log group
    
    Returns:
        dict: Response from the API call with success/failure information and details
    
    Raises:
        ClientError: If the API call fails
    """
    # Create CloudWatch logs client with optional region specification
    if region_name:
        logs_client = boto3.client('logs', region_name=region_name)
    else:
        logs_client = boto3.client('logs')
    
    try:
        # Prepare the API call parameters
        create_params = {'logGroupName': log_group_name}
        
        # Create the log group
        response = logs_client.create_log_group(**create_params)
        
        # Set retention policy if specified
        if retention_days is not None:
            logs_client.put_retention_policy(
                logGroupName=log_group_name,
                retentionInDays=retention_days
            )
        
        # Apply tags if specified
        if tags is not None:
            logs_client.tag_log_group(
                logGroupName=log_group_name,
                tags=tags
            )
        
        return {
            'status': 'success',
            'message': f'Log group {log_group_name} created successfully',
            'response': response
        }
    
    except ClientError as e:
        error_code = e.response.get('Error', {}).get('Code')
        
        # Handle the case where the log group already exists
        if error_code == 'ResourceAlreadyExistsException':
            # Check if we should update the existing log group
            if retention_days is not None or tags is not None:
                try:
                    # Update retention policy if specified
                    if retention_days is not None:
                        logs_client.put_retention_policy(
                            logGroupName=log_group_name,
                            retentionInDays=retention_days
                        )
                    
                    # Update tags if specified
                    if tags is not None:
                        # First get existing tags
                        existing_tags = logs_client.list_tags_log_group(
                            logGroupName=log_group_name
                        ).get('tags', {})
                        
                        # Merge new tags with existing tags
                        merged_tags = {**existing_tags, **tags}
                        
                        # Apply merged tags
                        logs_client.tag_log_group(
                            logGroupName=log_group_name,
                            tags=merged_tags
                        )
                    
                    return {
                        'status': 'updated',
                        'message': f'Log group {log_group_name} already exists. Updated its configuration.',
                        'error': str(e)
                    }
                except Exception as update_error:
                    return {
                        'status': 'error',
                        'message': f'Log group {log_group_name} already exists. Failed to update its configuration.',
                        'error': str(update_error)
                    }
            else:
                return {
                    'status': 'exists',
                    'message': f'Log group {log_group_name} already exists.',
                    'error': str(e)
                }
        else:
            return {
                'status': 'error',
                'message': f'Failed to create log group {log_group_name}',
                'error': str(e)
            }



def create_s3_bucket(bucket_name, region_name="us-west-2"):
    """
    Create an S3 bucket if it doesn't exist.
    
    Args:
        bucket_name (str): S3 bucket name.
        region_name (str): AWS region name.
        
    Returns:
        bool: True if bucket exists or was created, False otherwise.
    """
    # Initialize S3 client with the specified AWS region
    s3 = boto3.client("s3", region_name=region_name)
    
    try:
        # Check if the S3 bucket already exists
        s3.head_bucket(Bucket=bucket_name)
        print(f"Bucket '{bucket_name}' already exists.")
        return True
    except:
        try:
            # Create the S3 bucket if it does not exist
            s3.create_bucket(
                Bucket=bucket_name, 
                CreateBucketConfiguration={'LocationConstraint': region_name}
            )
            print(f"Bucket '{bucket_name}' created.")
            return True
        except Exception as e:
            print(f"Error creating bucket: {e}")
            return False

def upload_directory(path, bucket_name, data_s3_prefix, region_name="us-west-2"):
    """
    Upload all files from a local directory to an S3 bucket.
    
    Args:
        path (str): Local directory path.
        bucket_name (str): S3 bucket name.
        data_s3_prefix (str): S3 prefix for the uploaded files.
        region_name (str): AWS region name.
    """
    s3 = boto3.client("s3", region_name=region_name)
    
    for root, dirs, files in os.walk(path):
        for file in files:
            key = f"{data_s3_prefix}/{file}"  # Construct the S3 object key
            s3.upload_file(os.path.join(root, file), bucket_name, key)  # Upload the file

def download_and_prepare_science_papers(bucket_name, region_name="us-west-2"):
    """
    Download Amazon Science papers, prepare metadata, and upload to S3.
    
    Args:
        bucket_name (str): S3 bucket name.
        region_name (str): AWS region name.
    """
    # Define URLs of Amazon Science Publications to download as example documents
    urls = [
        "https://assets.amazon.science/44/ba/e16182124eac8687e89d3cb0ea3d/retrieval-reranking-and-multi-task-learning-for-knowledge-base-question-answering.pdf",
        "https://assets.amazon.science/36/be/2669792342f2ba366ddca794069f/practiq-a-practical-conversational-text-to-sql-dataset-with-ambiguous-and-unanswerable-queries.pdf",
        "https://assets.amazon.science/a7/7c/8bdade5c4eda9168f3dee6434fff/pc-amazon-frontier-model-safety-framework-2-7-final-2-9.pdf"
    ]
    
    # Define standard filenames to maintain consistency when loading data to Amazon S3
    filenames = [
        "retrieval-reranking-and-multi-task-learning-for-knowledge-base-question-answering.pdf",
        "practiq-a-practical-conversational-text-to-sql-dataset-with-ambiguous-and-unanswerable-queries.pdf",
        "pc-amazon-frontier-model-safety-framework-2-7-final-2-9.pdf"
    ]
    
    # Create a local temporary directory to store downloaded files before uploading to S3
    os.makedirs("./data", exist_ok=True)
    
    # Define local directory path for storing downloaded files
    local_data_path = "./data/"
    
    # Download files from URLs and save them in the local directory
    for idx, url in enumerate(urls):
        file_path = os.path.join(local_data_path, filenames[idx])
        urlretrieve(url, file_path)
    
    # Define metadata corresponding to each document for indexing in the vector database
    metadata = [
        {
            "metadataAttributes": {
                "company": "Amazon",
                "authors": ["Zhiguo Wang", "Patrick Ng", "Ramesh Nallapati", "Bing Xiang"],
                "docType": "science",
                "year": 2021
            }
        },
        {
            "metadataAttributes": {
                "company": "Amazon",
                "authors": ["Marvin Dong", "Nischal Ashok Kumar", "Yiqun Hu", "Anuj Chauhan", "Chung-Wei Hang", "Shuaichen Chang", 
                            "Lin Pan", "Wuwei Lan", "Henry Zhu", "Jiarong Jiang", "Patrick Ng", "Zhiguo Wang"],
                "docType": "science",
                "year": 2025
            }
        },
        {
            "metadataAttributes": {
                "company": "Amazon",
                "authors": ["Amazon"],
                "docType": "science",
                "year": 2025
            }
        }
    ]
    
    # Save metadata as JSON files alongside the corresponding documents
    for i, file in enumerate(filenames):
        with open(f"{local_data_path}{file}.metadata.json", "w") as f:
            json.dump(metadata[i], f)
    
    # Upload the directory to Amazon S3 under the 'pdf_documents' prefix
    upload_directory(local_data_path, bucket_name, "data/pdf_documents", region_name)
    
    # Delete the local directory and its contents after upload to save space
    shutil.rmtree(local_data_path)

def download_and_prepare_10k_reports(bucket_name, region_name="us-west-2"):
    """
    Download Amazon 10-K reports, prepare metadata, and upload to S3.
    
    Args:
        bucket_name (str): S3 bucket name.
        region_name (str): AWS region name.
    """
    # Define URLs of Amazon's 10-K reports to be downloaded as example documents
    urls = [
        "https://d18rn0p25nwr6d.cloudfront.net/CIK-0001018724/e42c2068-bad5-4ab6-ae57-36ff8b2aeffd.pdf",
        "https://d18rn0p25nwr6d.cloudfront.net/CIK-0001018724/c7c14359-36fa-40c3-b3ca-5bf7f3fa0b96.pdf",
        "https://d18rn0p25nwr6d.cloudfront.net/CIK-0001018724/d2fde7ee-05f7-419d-9ce8-186de4c96e25.pdf"
    ]
    
    # Define standard filenames to maintain consistency when loading data to Amazon S3
    filenames = [
        "Amazon-10k-2025.pdf",
        "Amazon-10k-2024.pdf",
        "Amazon-10k-2023.pdf"
    ]
    
    # Create a local temporary directory to store downloaded files before uploading to S3
    local_data_path = "./data/"
    os.makedirs(local_data_path, exist_ok=True)
    
    # Download files from URLs and save them in the local directory
    for idx, url in enumerate(urls):
        file_path = os.path.join(local_data_path, filenames[idx])
        urlretrieve(url, file_path)
    
    # Define metadata corresponding to each document for indexing in the vector database
    metadata = [
        {
            "metadataAttributes": {
                "company": "Amazon",
                "authors": ["Amazon"],
                "docType": "10K Report",
                "year": 2025
            }
        },
        {
            "metadataAttributes": {
                "company": "Amazon",
                "authors": ["Amazon"],
                "docType": "10K Report",
                "year": 2024
            }
        },
        {
            "metadataAttributes": {
                "company": "Amazon",
                "authors": ["Amazon"],
                "docType": "10K Report",
                "year": 2023
            }
        }
    ]
    
    # Save metadata as JSON files alongside the corresponding documents
    for i, file in enumerate(filenames):
        metadata_file_path = os.path.join(local_data_path, f"{file}.metadata.json")
        with open(metadata_file_path, "w") as f:
            json.dump(metadata[i], f, indent=4)
    
    # Upload the directory to Amazon S3 under the 'pdf_documents' prefix
    upload_directory(local_data_path, bucket_name, "data/pdf_documents", region_name)
    
    # Delete the local directory and its contents after upload to save space
    shutil.rmtree(local_data_path)

def download_and_prepare_csv_data(bucket_name, region_name="us-west-2"):
    """
    Download CSV data, prepare metadata, and upload to S3.
    
    Args:
        bucket_name (str): S3 bucket name.
        region_name (str): AWS region name.
    """
    # Create a directory to store the video game CSV dataset
    local_dir = "./videogame/"
    os.makedirs(local_dir, exist_ok=True)
    
    # Define the URL of the dataset and the local file path
    csv_url = "https://raw.githubusercontent.com/ali-ce/datasets/master/Most-Expensive-Things/Videogames.csv"
    csv_file_path = os.path.join(local_dir, "video_games.csv")
    
    # Download the CSV file
    response = requests.get(csv_url)  # `verify=False` ignores SSL certificate issues
    if response.status_code == 200:
        with open(csv_file_path, 'wb') as file:
            file.write(response.content)
        print(f"CSV file downloaded successfully: {csv_file_path}")
    else:
        print("Failed to download the CSV file.")
    
    # Generate JSON metadata for the downloaded CSV file
    generate_json_metadata(
        csv_file=csv_file_path,
        content_fields=["Description"],
        metadata_fields=["Year", "Developer", "Publisher"],
        excluded_fields=[]  # Automatically determine excluded fields
    )
    
    # Upload directory containing the CSV and metadata JSON to S3
    upload_directory(local_dir, bucket_name, "data/csv", region_name)
    
    # Remove the local directory after upload to save space
    shutil.rmtree(local_dir)

def generate_json_metadata(csv_file, content_fields, metadata_fields, excluded_fields):
    """
    Generate JSON metadata for a CSV file.
    
    Args:
        csv_file (str): Path to the CSV file.
        content_fields (list): List of fields that contain document content.
        metadata_fields (list): List of fields to include as metadata.
        excluded_fields (list): List of fields to exclude (automatically populated if empty).
    """
    import csv
    
    # Open the CSV file and read its headers
    with open(csv_file, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        headers = reader.fieldnames  # Get column names
    
    # Define JSON structure for metadata
    json_data = {
        "metadataAttributes": {},
        "documentStructureConfiguration": {
            "type": "RECORD_BASED_STRUCTURE_METADATA",
            "recordBasedStructureMetadata": {
                "contentFields": [{"fieldName": field} for field in content_fields],
                "metadataFieldsSpecification": {
                    "fieldsToInclude": [{"fieldName": field} for field in metadata_fields],
                    "fieldsToExclude": []
                }
            }
        }
    }
    
    # Determine fields to exclude (all fields not in content_fields or metadata_fields)
    if not excluded_fields:
        excluded_fields = set(headers) - set(content_fields + metadata_fields)
    
    json_data["documentStructureConfiguration"]["recordBasedStructureMetadata"]["metadataFieldsSpecification"]["fieldsToExclude"] = [
        {"fieldName": field} for field in excluded_fields
    ]
    
    # Generate the output JSON file name
    output_file = f"{os.path.splitext(csv_file)[0]}.metadata.json"
    
    # Save metadata to a JSON file
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(json_data, file, indent=4)
    
    print(f"JSON metadata file '{output_file}' has been generated.")

def create_opensearch_policies(iam_role_handler, vector_store_name, aoss_client, execution_role_arn):
    """
    Create security, network, and data access policies for OpenSearch.
    
    Args:
        iam_role_handler: IAM role handler instance.
        vector_store_name (str): Vector store name.
        aoss_client: OpenSearch Serverless client.
        execution_role_arn (str): Bedrock execution role ARN.
        
    Returns:
        tuple: (encryption_policy, network_policy, access_policy)
    """
    try:
        result = iam_role_handler.create_policies_in_oss(
            vector_store_name=vector_store_name,
            aoss_client=aoss_client,
            bedrock_kb_execution_role_arn=execution_role_arn
        )
        if result is not None:  # Check if the result is valid
            encryption_policy, network_policy, access_policy = result
            return encryption_policy, network_policy, access_policy
        else:
            print("Policies already exist or were not created properly.")
            return None, None, None
    except Exception as e:
        print(f"Error creating policies: {str(e)}")
        return None, None, None

def create_opensearch_collection(aoss_client, collection_name, region_name="us-west-2"):
    """
    Create an OpenSearch Serverless collection.
    
    Args:
        aoss_client: OpenSearch Serverless client.
        collection_name (str): Collection name.
        region_name (str): AWS region name.
        
    Returns:
        tuple: (collection_id, host_url)
    """
    # Check if the collection already exists before creation
    try:
        response = aoss_client.batch_get_collection(names=[collection_name])
        if response['collectionDetails']:
            print(f"Collection '{collection_name}' already exists.")
            # Extract the collection ID from the existing collection
            collection_id = response['collectionDetails'][0]['id']
            host = f"{collection_id}.{region_name}.aoss.amazonaws.com"  # Construct the host URL
            print(f"Collection Host URL: {host}")
            return collection_id, host
        else:
            # Create an OpenSearch Serverless Vector Collection
            collection = aoss_client.create_collection(name=collection_name, type='VECTORSEARCH')
            collection_id = collection['createCollectionDetail']['id']
            host = f"{collection_id}.{region_name}.aoss.amazonaws.com"  # Construct the host URL
            print(f"Collection Host URL: {host}")
            return collection_id, host
    except Exception as e:
        print(f"Error with collection operation: {e}")
        return None, None

def wait_for_collection_active(aoss_client, collection_name):
    """
    Wait for OpenSearch collection to become active.
    
    Args:
        aoss_client: OpenSearch Serverless client.
        collection_name (str): Collection name.
    """
    response = aoss_client.batch_get_collection(names=[collection_name])
    print(response)
    
    # Periodically check the collection's status until it's no longer 'CREATING'
    while response['collectionDetails'][0]['status'] == 'CREATING':
        print('Collection is still being created...')
        time.sleep(10)  # Sleep for 10 seconds before checking again
        response = aoss_client.batch_get_collection(names=[collection_name])
    
    # Confirm successful collection creation
    print('\nCollection successfully created!')



def create_opensearch_access_policy(iam_role_handler, collection_id, execution_role):
    """
    Create an OpenSearch access policy and attach it to the execution role.
    
    Args:
        iam_role_handler: IAM role handler instance.
        collection_id (str): Collection ID.
        execution_role: Execution role.
    """
    try:
        iam_role_handler.create_oss_policy_attach_bedrock_execution_role(
            collection_id=collection_id,
            bedrock_kb_execution_role=execution_role
        )
        # Wait for the data access rules to be enforced (may take a minute)
        time.sleep(10)
    except Exception as e:
        print(f"Policy already exists or has been attached previously: {e}")


iam = boto3.client("iam")
def create_oss_policy_attach_bedrock_execution_role(collection_id, bedrock_kb_execution_role,region_name,account_number):
        """Creates and attaches an OpenSearch Serverless (OSS) policy to the Bedrock execution role."""
        
        oss_policy_document = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Action": ["aoss:APIAccessAll"],
                    "Resource": [
                        f"arn:aws:aoss:{region_name}:{account_number}:collection/{collection_id}"
                    ]
                }
            ]
        }

        oss_policy = iam.create_policy(
            PolicyName=f"advanced-rag-oss-policy-{region_name}",
            PolicyDocument=json.dumps(oss_policy_document),
            Description="Policy for accessing OpenSearch Serverless",
        )
        print(oss_policy)
        print("created oss policy successfully, proceeding to attach policy") 
        # Attach the policy to the Bedrock execution role
        iam.attach_role_policy(
            RoleName=bedrock_kb_execution_role,
            PolicyArn=oss_policy["Policy"]["Arn"]
        )
        print("Successfully attached oss policy to Bedrock Execution Role") 
        return None

def create_opensearch_index(collection_host, index_name_prefix, region_name="us-west-2"):
    """
    Create OpenSearch indexes for different chunking strategies.
    
    Args:
        collection_host (str): OpenSearch collection host.
        index_name_prefix (str): Index name prefix.
        region_name (str): AWS region name.
    """
    # Initialize AWS credentials for authentication
    credentials = boto3.Session().get_credentials()
    awsauth = AWSV4SignerAuth(credentials, region_name, "aoss")
    
    # Define the index body
    body_json = {
        "settings": {
            "index.knn": "true",
            "number_of_shards": 1,
            "knn.algo_param.ef_search": 512,
            "number_of_replicas": 0,
        },
        "mappings": {
            "properties": {
                "vector": {
                    "type": "knn_vector",
                    "dimension": 1024,
                    "method": {
                        "name": "hnsw",
                        "engine": "faiss",
                        "space_type": "l2"
                    },
                },
                "text": {
                    "type": "text"
                },
                "text-metadata": {
                    "type": "text"
                }
            }
        }
    }
    
    # Build the OpenSearch client
    oss_client = OpenSearch(
        hosts=[{'host': collection_host, 'port': 443}],
        http_auth=awsauth,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
        timeout=300
    )
    
    # Create indexes for different chunking strategies
    for strategy in ["fixed", "hierarchical", "semantic", "custom"]:
        index_name = index_name_prefix + strategy
        try:
            # Check if the index already exists
            if oss_client.indices.exists(index=index_name):
                print(f'Index "{index_name}" already exists. Skipping creation.')
                continue
            
            # Create the index if it doesn't exist
            oss_client.indices.create(index=index_name, body=json.dumps(body_json))
            print(f'Creating Index: {index_name}...')
        except RequestError as e:
            print(f'Error while trying to create the index "{index_name}", with error {e.error}')
    
    print('Index Creation Process Completed.')

def save_variables_to_json(variables, file_path="../variables.json"):
    """
    Save variables to a JSON file.
    
    Args:
        variables (dict): Variables to save.
        file_path (str): Path to save the JSON file.
    """
    with open(file_path, "w") as f:
        json.dump(variables, f, indent=4)
    
    print(f"Variables saved to {file_path}")


######################################################################
#notebook 1.2, 1.3,1.4
######################################################################

@retry(wait_random_min=1000, wait_random_max=2000, stop_max_attempt_number=3)
def create_knowledge_base_func(name, description, chunking_type, variables, model_id):
    """
    Creates a knowledge base in Amazon Bedrock with specified configuration and OpenSearch Serverless storage.
    
    Args:
        name (str): Name of the knowledge base
        description (str): Description of the knowledge base
        chunking_type (str): Type of text chunking strategy (e.g., 'SEMANTIC', 'FIXED')
        variables (dict): Configuration dictionary containing:
            - regionName: AWS region
            - vectorIndexName: Base name for vector index
            - collectionArn: OpenSearch collection ARN
            - bedrockExecutionRoleArn: IAM role ARN for Bedrock
    
    Returns:
        dict: Details of the created knowledge base
    
    Raises:
        boto3.exceptions.ClientError: If AWS API call fails
        Exception: For other unexpected errors
    
    Note:
        Function includes retry mechanism with random wait between 1-2 seconds,
        making up to 3 attempts before failing
    """
    # Create Bedrock agent client for the specified AWS region
    bedrock_agent = boto3.client("bedrock-agent", region_name=variables["regionName"])

    # Configure ARN for Titan embedding model v2 - used for document and query embedding
    embedding_model_arn = f"arn:aws:bedrock:{variables['regionName']}::foundation-model/{model_id}"

    # Create unique vector index name by combining base name with chunking type
    vectorIndexName = variables["vectorIndexName"] + chunking_type
    
    # Set up OpenSearch Serverless configuration for vector storage
    opensearch_serverless_configuration = {
            "collectionArn": variables["collectionArn"],  # OpenSearch collection identifier
            "vectorIndexName": vectorIndexName,          # Name of vector index
            "fieldMapping": {                            # Define field structure
                "vectorField": "vector",                 # Field for storing vectors
                "textField": "text",                     # Field for storing text content, i.e., chunk
                "metadataField": "text-metadata"         # Field for storing metadata
            }
        }

    
    # Log configuration for debugging and verification
    print(opensearch_serverless_configuration)
    
    # Make API call to create knowledge base with specified configuration
    create_kb_response = bedrock_agent.create_knowledge_base(
        name=name,                                      # Knowledge base name
        description=description,                        # Knowledge base description
        roleArn=variables["bedrockExecutionRoleArn"],  # IAM role for permissions
        knowledgeBaseConfiguration={                    # Vector-based configuration
            "type": "VECTOR",
            "vectorKnowledgeBaseConfiguration": {
                "embeddingModelArn": embedding_model_arn # Specify embedding model
            }
        },
        storageConfiguration={                          # Storage configuration
            "type": "OPENSEARCH_SERVERLESS",
            "opensearchServerlessConfiguration": opensearch_serverless_configuration
        }
    )
    
    # Return the newly created knowledge base configuration
    return create_kb_response["knowledgeBase"]
def update_kb_id(variables, kb_chunking_type_in, kb_id):
    """
    Updates a knowledge base ID in the variables dictionary based on the chunking type
    
    Parameters:
        variables: Dictionary storing different knowledge base IDs
        kb_chunking_type_in: String indicating the type of chunking (semantic/fixed/hierarchical/custom)
        kb_id: The knowledge base ID to be stored
    """
    # Convert input chunking type to uppercase for case-insensitive comparison
    kb_chunking_type = kb_chunking_type_in.upper()
    
    # Update the appropriate dictionary key based on chunking type
    if kb_chunking_type == "SEMANTIC":                 
        variables["kbSemanticChunk"] = kb_id
    elif kb_chunking_type == "FIXED":
        variables["kbFixedChunk"] = kb_id
    elif kb_chunking_type == "HIERARCHICAL":
        variables["kbHierarchicalChunk"] = kb_id
    elif kb_chunking_type == "CUSTOM":
        variables["kbCustomChunk"] = kb_id

def create_kb(kb_name, kb_description, kb_chunking_type, variables, model_id):
    """
    Creates or retrieves a knowledge base in AWS Bedrock and updates configuration variables.
    
    Args:
        kb_name (str): Name of the knowledge base to create
        kb_description (str): Description of the knowledge base
        kb_chunking_type (str): Type of text chunking strategy (e.g., 'SEMANTIC', 'FIXED')
        variables (dict): Configuration dictionary containing AWS settings and IDs
    
    Returns:
        dict: Knowledge base configuration object, or None if creation fails
    
    Raises:
        Exception: Propagates any unexpected errors during creation
    
    Note:
        - If knowledge base already exists, retrieves existing KB instead of creating new one
        - Updates variables.json file with knowledge base ID
        - Handles datetime serialization in JSON output
    """
    # Initialize Bedrock agent client with region from variables
    bedrock_agent = boto3.client("bedrock-agent", region_name=variables["regionName"])
    kb = None

    try:
        # Attempt to create new knowledge base
        kb = create_knowledge_base_func(
            name=kb_name,
            description=kb_description,
            chunking_type=kb_chunking_type,
            variables=variables,
            model_id=model_id
        )
    
        # Fetch details of newly created knowledge base
        get_kb_response = bedrock_agent.get_knowledge_base(knowledgeBaseId=kb['knowledgeBaseId'])

        # Update variables dictionary with new knowledge base ID
        update_kb_id(variables, kb_chunking_type, kb['knowledgeBaseId'])
    
        # Persist updated variables to JSON file
        with open("../variables.json", "w") as f:
            json.dump(variables, f, indent=4, default=str)  # Handle datetime serialization
    
        # Log knowledge base details
        print(f'OpenSearch Knowledge Response: {json.dumps(get_kb_response, indent=4, default=str)}')
        
    except Exception as e:
        # Handle case where knowledge base already exists
        error_message = str(e).lower()
        if any(phrase in error_message for phrase in ["already exist", "duplicate", "already been created"]):
            print("Knowledge Base already exists. Retrieving its ID...")
            
            # List all knowledge bases to find existing one
            list_kb_response = bedrock_agent.list_knowledge_bases()
            
            # Search for knowledge base with matching name
            for kb in list_kb_response.get('knowledgeBaseSummaries', []):
                # print(f"kb_name = {kb['name']}")
                if kb['name'] == kb_name:
                    kb_id = kb['knowledgeBaseId']
                    print(f"Found existing knowledge base with Name: {kb_name} and ID: {kb_id}")
                    
                    # Get existing knowledge base details
                    get_kb_response = bedrock_agent.get_knowledge_base(knowledgeBaseId=kb_id)
                    
                    # Load existing variables from JSON file
                    try:
                        with open("../variables.json", "r") as f:
                            existing_variables = json.load(f)
                    except (FileNotFoundError, json.JSONDecodeError):
                        existing_variables = {}
                    
                    # Update variables with existing knowledge base ID
                    update_kb_id(existing_variables, kb_chunking_type, kb_id)
                    
                    # Save updated variables back to JSON file
                    with open("../variables.json", "w") as f:
                        json.dump(existing_variables, f, indent=4, default=str)
                    
                    # Log existing knowledge base details
                    print(f'OpenSearch Knowledge Response: {json.dumps(get_kb_response["knowledgeBase"], indent=4, default=str)}')
                    break            
        else:
            # Propagate unexpected errors
            raise e
    
    if kb is None :
        raise ValueError(f"Knowledge base '{kb_name}' not found or could not be created.")
    
    return kb

def create_chunking_strategy(chunking_strategy_in):
    """
    Creates a configuration dictionary for different text chunking strategies in Amazon Bedrock.
    
    Args:
        chunking_strategy_in (str): The type of chunking strategy to use.
            Supported values: 'SEMANTIC', 'FIXED', 'HIERARCHICAL'
    
    Returns:
        dict: Configuration dictionary for the specified chunking strategy
    
    Note:
        - SEMANTIC: Chunks based on semantic meaning with configurable token limits
        - FIXED: Creates chunks of fixed size with overlap
        - HIERARCHICAL: Creates nested chunks at different levels
        - Custom chunking is planned but not yet implemented
    """
    # Convert input to uppercase for case-insensitive comparison
    chunking_strategy = chunking_strategy_in.upper()

    if chunking_strategy == "SEMANTIC":
        # Configure semantic chunking with natural language understanding
        chunking_strategy_configuration = {
            "chunkingStrategy": chunking_strategy,
            "semanticChunkingConfiguration": {
                "maxTokens": 300,                    # Maximum tokens per chunk
                "bufferSize": 1,                     # Overlap buffer size
                "breakpointPercentileThreshold": 95  # Threshold for chunk breaks
            }
        }
    elif chunking_strategy == "FIXED":
        # Configure fixed-size chunking with overlap
        chunking_strategy_configuration = {
            "chunkingStrategy": "FIXED_SIZE",
            "fixedSizeChunkingConfiguration": {
                "maxTokens": 300,        # Fixed chunk size in tokens
                "overlapPercentage": 20   # Percentage of overlap between chunks
            }
        }
    elif chunking_strategy == "HIERARCHICAL":
        # Configure hierarchical chunking with multiple levels
        chunking_strategy_configuration = {
            "chunkingStrategy": "HIERARCHICAL",
            "hierarchicalChunkingConfiguration": {
                "levelConfigurations": [
                    {"maxTokens": 1500},  # Top level chunks
                    {"maxTokens": 300}    # Sub-level chunks
                ],
                "overlapTokens": 60       # Token overlap between levels
            }
        }
    # TODO: Implement custom chunking strategy configuration
    
    return chunking_strategy_configuration

def create_data_source_for_kb(chunking_strategy, data_source_name, kb, variables):
    """
    Creates or recreates a data source for a Bedrock knowledge base with specified chunking strategy.
    
    Args:
        chunking_strategy (str): Type of chunking strategy ('SEMANTIC', 'FIXED', 'HIERARCHICAL')
        data_source_name (str): Name for the data source
        kb (dict): Knowledge base configuration dictionary
        variables (dict): Configuration variables including S3 bucket and region
    
    Returns:
        dict: Created or retrieved data source object
    
    Raises:
        ClientError: For AWS API-related errors
        Exception: For other unexpected errors
    
    Note:
        - If a data source with the same name exists, it will be deleted and recreated
        - Only processes S3 objects with 'data' prefix
        - Includes error handling and waiting periods for AWS operations
    """
    # Initialize Bedrock agent client
    bedrock_agent = boto3.client("bedrock-agent", region_name=variables["regionName"])
    
    # Get chunking strategy configuration based on specified type
    chunking_strategy_configuration = create_chunking_strategy(chunking_strategy)
    
    # Configure S3 data source with specific prefix filter
    s3_configuration = {
        "bucketArn": f"arn:aws:s3:::{variables['s3Bucket']}",
        "inclusionPrefixes": ["data"]  # Filter to only include 'data' prefixed objects
    }
    
    # Check for existing data source and delete if found
    try:
        # Get list of all data sources for the knowledge base
        list_ds_response = bedrock_agent.list_data_sources(
            knowledgeBaseId=kb['knowledgeBaseId']
        )
        
        # Search for matching data source name
        existing_ds = None
        for ds in list_ds_response.get('dataSourceSummaries', []):
            if ds['name'] == data_source_name:
                existing_ds = ds
                break
        
        # Delete existing data source if found
        if existing_ds:
            print(f"Found existing data source '{data_source_name}'. Deleting it...")
            bedrock_agent.delete_data_source(
                knowledgeBaseId=kb['knowledgeBaseId'],
                dataSourceId=existing_ds["dataSourceId"]
            )
            print("Waiting for data source deletion to complete...")
            time.sleep(10)  # Wait for deletion to complete
            print("Data source deleted successfully.")
            
    except Exception as e:
        print(f"Error while checking or deleting data source: {e}")
    
    # Create new data source
    try:
        print(f"Creating new data source '{data_source_name}' with {chunking_strategy_configuration} chunking...")
        create_ds_response = bedrock_agent.create_data_source(
            name=data_source_name,
            description="A data source for Advanced RAG workshop",
            knowledgeBaseId=kb['knowledgeBaseId'],
            dataSourceConfiguration={
                "type": "S3",
                "s3Configuration": s3_configuration
            },
            vectorIngestionConfiguration={
                "chunkingConfiguration": chunking_strategy_configuration
            }
        )
        
        # Store and return created data source
        ds_object = create_ds_response["dataSource"]
        print(f"{chunking_strategy} chunking data source created successfully.")
        
    except ClientError as e:
        # Handle case where data source still exists
        if e.response['Error']['Code'] == 'ConflictException':
            print(f"Data source '{data_source_name}' still exists. Retrieving it...")
            # Retrieve existing data source
            list_ds_response = bedrock_agent.list_data_sources(
                knowledgeBaseId=kb['knowledgeBaseId']
            )
            for ds in list_ds_response.get('dataSourceSummaries', []):
                if ds['name'] == data_source_name:
                    ds_object = ds
                    print(f"Retrieved existing data source: {ds['dataSourceId']}")
                    break            
        else:
            raise e
    
    return ds_object

import time
def create_ingestion_job(kb, ds_object, variables):
    """
    Creates and monitors an ingestion job for a Bedrock knowledge base data source.
    
    Args:
        kb (dict): Knowledge base configuration dictionary containing:
            - knowledgeBaseId: Unique identifier for the knowledge base
            - name: Name of the knowledge base
        ds_object (dict): Data source configuration dictionary containing:
            - dataSourceId: Unique identifier for the data source
        variables (dict): Configuration variables including:
            - regionName: AWS region for the Bedrock agent
    
    Returns:
        None
    
    Raises:
        Exception: If job creation or monitoring fails
    
    Note:
        - Function polls job status every 10 seconds until completion
        - Prints progress updates to console
        - Handles job creation and monitoring errors
    """
    # Initialize Bedrock agent client with specified region
    bedrock_agent = boto3.client("bedrock-agent", region_name=variables["regionName"])
    
    # Initialize list for tracking ingestion jobs (for future use)
    ingest_jobs = []
    
    try:
        # Start new ingestion job with knowledge base and data source IDs
        start_job_response = bedrock_agent.start_ingestion_job(
            knowledgeBaseId=kb['knowledgeBaseId'],
            dataSourceId=ds_object["dataSourceId"]
        )
        
        # Extract job details from response
        job = start_job_response["ingestionJob"]
        print(f"Ingestion job started successfully for kb_name = {kb['name']} "
              f"and kb_id = {kb['knowledgeBaseId']}\n")
    
        # Monitor job status until completion
        while job['status'] != 'COMPLETE':
            # Wait 10 seconds between status checks
            print("running...")
            time.sleep(10)
            
            # Get updated job status
            get_job_response = bedrock_agent.get_ingestion_job(
                knowledgeBaseId=kb['knowledgeBaseId'],
                dataSourceId=ds_object["dataSourceId"],
                ingestionJobId=job["ingestionJobId"]
            )
            
            # Update job status from response
            job = get_job_response["ingestionJob"]
    
        print(f"Job completed successfully\n")
    
    except Exception as e:
        # Log error if job creation or monitoring fails
        print(f"Couldn't start job.\n")
        print(e)
        
def extract_retrieval_results(relevant_documents, min_score=None):
    """
    Extract content, metadata and score from relevant_documents array
    
    Args:
        relevant_documents (list): List of retrieval result objects from Knowledge Base
        
    Returns:
        list: List of dictionaries containing extracted content, metadata and score
    """
    extracted_results = []
    retrieval_results = relevant_documents['retrievalResults']
    
    for result in retrieval_results:
        # Skip results below minimum score threshold if specified
        if min_score is not None and result.get('score', 0) < min_score:
            continue
        
        extracted_item = {}
        
        # Extract content - handle both text and byte content
        if 'content' in result:
            content = result['content']
            if 'text' in content:
                extracted_item['content'] = content['text']
            elif 'byteContent' in content:
                extracted_item['content'] = content['byteContent']
            else:
                extracted_item['content'] = None
        
        # Extract metadata if present
        if 'metadata' in result:
            extracted_item['metadata'] = result['metadata']
        else:
            extracted_item['metadata'] = None
            
        # Extract relevancy score if present
        if 'score' in result:
            extracted_item['score'] = result['score']
        else:
            extracted_item['score'] = None
            
        extracted_results.append(extracted_item)
        
    return extracted_results
    
def retrieve_from_kb(query, kb, n_chunks, variables, min_score=None):
    # Initialize the Bedrock agent runtime client to interact with the Bedrock service
    bedrock_agent_runtime = boto3.client("bedrock-agent-runtime", region_name=variables["regionName"])
    
    # Use the Bedrock agent runtime to retrieve relevant documents from the knowledge base
    relevant_documents = bedrock_agent_runtime.retrieve(
        retrievalQuery= {
            'text': query  # The text query for retrieving documents
        },
        knowledgeBaseId=kb['knowledgeBaseId'],  # The knowledge base ID to search within
        retrievalConfiguration= {
            'vectorSearchConfiguration': {
                'numberOfResults': n_chunks  # Fetch the top n that closely match the query
            }
        }
    )
    
    #extract the text, metadata, and score
    extracted_chunks = extract_retrieval_results(relevant_documents, min_score)
    return extracted_chunks


import boto3
from datetime import datetime

def analyze_chunk_scores_above_threshold(json_data, threshold, return_matching_elements=False):
    """
    Analyze scores in the JSON array and return statistics
    
    Args:
        json_data: List of dictionaries containing 'score' field
        threshold: Minimum score threshold
        return_matching_elements: Whether to include matching elements in the result
        
    Returns:
        dict: Dictionary containing:
            - count_above_threshold: Number of elements with score >= threshold
            - min_score: Minimum score in the dataset
            - max_score: Maximum score in the dataset
            - avg_score: Average score in the dataset
            - matching_elements: List of elements with scores >= threshold (if requested)
    """
    # Initialize variables
    scores = [item.get('score', 0) for item in json_data]
    matching_elements = []
    
    # Calculate statistics
    min_score = min(scores) if scores else 0
    max_score = max(scores) if scores else 0
    avg_score = sum(scores) / len(scores) if scores else 0
    
    # Get matching elements
    for item in json_data:
        score = item.get('score', 0)
        if score >= threshold:
            matching_elements.append({
                'score': score,
                'content': item.get('content', ''),
                'metadata': item.get('metadata', {})
            })
    
    score_struct = {
        'total_chunks': len(json_data),        
        'min_score': min_score,
        'max_score': max_score,
        'avg_score': avg_score,
        'count_above_threshold': len(matching_elements),
    }
    
    if return_matching_elements == True:
        score_struct['matching_elements'] = matching_elements
    
    return score_struct

# Model pricing dictionary - add all models and their costs here
BEDROCK_MODEL_PRICING = {
    # Titan Embedding model
    'amazon.titan-embed-text-v2:0': {
        'input_per_million': 0.02,
        'output_per_million': 0.0
    },
    # Amazon Nova models - added based on CSV and search data
    'us.amazon.nova-micro-v1:0': {
        'input_per_million': 0.035,  # $0.000035 per 1,000 tokens
        'output_per_million': 0.12   # $0.00012 per 1,000 tokens
    },
    'us.amazon.nova-lite-v1:0': {
        'input_per_million': 0.12,   # $0.00012 per 1,000 tokens
        'output_per_million': 0.36   # $0.00036 per 1,000 tokens
    },
    'us.amazon.nova-pro-v1:0': {
        'input_per_million': 0.8,    # $0.0008 per 1,000 tokens
        'output_per_million': 3.2    # $0.0032 per 1,000 tokens
    }
}

def get_embedding_LLM_costs_for_KB(
    model_id: str,
    start_time: datetime,
    end_time: datetime,
    granularity_seconds_period: int = 10,
    assumed_GB_text=1,
    region ="us-west-2"
):
    """
    Calculate the cost of using an embedding model for a knowledge base.

    Args:
        model_id: The Bedrock model ID
        start_time: Start time for querying CloudWatch metrics
        end_time: End time for querying CloudWatch metrics
        granularity_seconds_period: Period in seconds for CloudWatch metrics
        assumed_GB_text: Assumed amount of text in GB (not currently used)

    Returns:
        dict: Dictionary containing token counts and costs
    """
    cloudwatch = boto3.client('cloudwatch')

    namespace = 'AWS/Bedrock'
    metrics = ['InputTokenCount', 'OutputTokenCount', 'Invocations']

    results = {}

    for metric in metrics:
        response = cloudwatch.get_metric_statistics(
            Namespace=namespace,
            MetricName=metric,
            Dimensions=[
                {
                    'Name': 'ModelId',
                    'Value': f"arn:aws:bedrock:{region}::foundation-model/{model_id}"
                }
            ],
            StartTime=start_time,
            EndTime=end_time,
            Period=granularity_seconds_period,
            Statistics=['Sum']
        )

        # Sum up all datapoints
        total_tokens = sum(dp['Sum'] for dp in response['Datapoints'])
        results[metric] = int(total_tokens)

    json_token = {
        'WARNING': (
            "These costs are approximate and directional as of April 2025. "
            "They will vary as per region and may change in future. "
            "The costs are for Embedding LLM using Bedrock On-Demand model."
        ),
        'model_id': model_id,
        'start_time': start_time.isoformat(),
        'end_time': end_time.isoformat(),
        'duration in minutes': (end_time - start_time).total_seconds() / 60,
        'input_tokens': results.get('InputTokenCount', 0),
        'output_tokens': results.get('OutputTokenCount', 0),
        'invocation_count': results.get('Invocations', 0)
    }

    model_pricing = BEDROCK_MODEL_PRICING.get(
        model_id, {'input_per_million': 0.0, 'output_per_million': 0.0}
    )

    # Add pricing info
    json_token['per million input token costs'] = model_pricing['input_per_million']
    json_token['per million output token costs'] = model_pricing['output_per_million']

    json_token['input token costs'] = (
        json_token['per million input token costs'] * (float(json_token['input_tokens']) / 1000000.0)
    )
    json_token['output token costs'] = (
        json_token['per million output token costs'] * (float(json_token['output_tokens']) / 1000000.0)
    )
    json_token['total token costs'] = (
        json_token['input token costs'] + json_token['output token costs']
    )

    if json_token['invocation_count'] != 0:
        json_token['average token costs per invocation'] = (
            json_token['total token costs'] / float(json_token['invocation_count'])
        )
    else:
        json_token['average token costs per invocation'] = 0.0

    json_token['token costs per MILLION such invocations'] = (
        json_token['average token costs per invocation'] * 1000000.0
    )

    return json_token


def get_bedrock_token_based_cost(model_id: str, start_time: datetime, end_time: datetime, granularity_seconds_period: int = 5, region="us-west-2"):
    """
    Calculate token counts and costs for Bedrock model usage
    
    Args:
        model_id: The Bedrock model ID
        start_time: Start time for querying CloudWatch metrics
        end_time: End time for querying CloudWatch metrics
        granularity_seconds_period: Period in seconds for CloudWatch metrics
        
    Returns:
        dict: Dictionary containing token counts and costs
    """

    cloudwatch = boto3.client('cloudwatch')

    namespace = 'AWS/Bedrock'
    metrics = ['InputTokenCount', 'OutputTokenCount', 'Invocations']

    results = {}

    for metric in metrics:
        response = cloudwatch.get_metric_statistics(
            Namespace=namespace,
            MetricName=metric,
            Dimensions=[
                {
                    'Name': 'ModelId',
                    'Value': f"arn:aws:bedrock:{region}::foundation-model/{model_id}" if "titan" in model_id else model_id
                }
            ],
            StartTime=start_time,
            EndTime=end_time,
            Period=granularity_seconds_period,
            Statistics=['Sum']
        )
        
        # Sum up all datapoints
        total_tokens = sum(dp['Sum'] for dp in response['Datapoints'])
        results[metric] = int(total_tokens)
    
    json_token = {
        'WARNING': "These costs are approximate and directional as of April 2025. They will vary as per region and may change in future. The costs are for Bedrock On-Demand model. If you see zero costs, chances are high that the LLM was not considered in cost calculations when the notebook was prepared. Please use AWS calculator for more accurate cost calculations.",
        'model_id': model_id,
        'start_time': start_time.isoformat(),
        'end_time': end_time.isoformat(),
        'duration in minutes': (end_time - start_time).total_seconds() / 60,
        'input_tokens': results.get('InputTokenCount', 0),
        'output_tokens': results.get('OutputTokenCount', 0),
        'invocation_count': results.get('Invocations', 0)
    }
    
    # Get pricing for the model (default to zero if model not found)
    model_pricing = BEDROCK_MODEL_PRICING.get(model_id, {'input_per_million': 0.0, 'output_per_million': 0.0})
    
    # Add pricing info
    json_token['per million input token costs'] = model_pricing['input_per_million']
    json_token['per million output token costs'] = model_pricing['output_per_million']
        
    json_token['input token costs'] = json_token['per million input token costs'] * (float(json_token['input_tokens'])/1000000.0)
    json_token['output token costs'] = json_token['per million output token costs'] * (float(json_token['output_tokens'])/1000000.0)
    json_token['total token costs'] = json_token['input token costs'] + json_token['output token costs']
    
    if json_token['invocation_count'] != 0:
        json_token['average token costs per invocation'] = json_token['total token costs'] / float(json_token['invocation_count'])
    else:
        json_token['average token costs per invocation'] = 0

    json_token['token costs per MILLION such invocations'] = json_token['average token costs per invocation'] * 1000000.0
    
    return json_token
    

def fmt_n(number, n=2):
    format_str = f"{{:,.{n}f}}"
    return format_str.format(number)

import pandas as pd
import os

def load_df_from_csv() :
    # Get parent directory path
    parent_dir = os.path.dirname(os.getcwd())
    file_path = os.path.join(parent_dir, 'embed_algo_costs.csv')
    
    # Create DataFrame with specified data types
    if os.path.exists(file_path):
        # Load existing CSV file
        df = pd.read_csv(file_path)
        # Ensure correct data types
        df = df.astype({
            'chunking_algo': str,
            'embedding_seconds': float,
            'input_tokens': int,
            'invocation_count': int,
            'total_token_costs': float
        })
        print(f"Loaded existing file: {file_path}")
    else:
        # Create new DataFrame if file doesn't exist
        df = pd.DataFrame(columns=[
            'chunking_algo',
            'embedding_seconds',
            'input_tokens',
            'invocation_count',
            'total_token_costs'
        ]).astype({
            'chunking_algo': str,
            'embedding_seconds': float,
            'input_tokens': int,
            'invocation_count': int,
            'total_token_costs': float
        })
    return df

def save_df_to_csv(df) :
    # Get parent directory path
    parent_dir = os.path.dirname(os.getcwd())
    file_path = os.path.join(parent_dir, 'embed_algo_costs.csv')
    
    # Save DataFrame to CSV
    df.to_csv(file_path, index=False)
    print(f"Successfully saved DataFrame to: {file_path}")

def update_or_add_row(df, new_row):
    """
    Update existing row if chunking_algo exists, otherwise add new row.
    
    Args:
        df (pd.DataFrame): Existing DataFrame
        new_row (dict): Dictionary containing new row data
    
    Returns:
        pd.DataFrame: Updated DataFrame
    """
    try:
        # Check if chunking_algo exists
        algo_exists = df['chunking_algo'].eq(new_row['chunking_algo']).any()
        
        if algo_exists:
            # Update existing row
            idx = df.index[df['chunking_algo'] == new_row['chunking_algo']].tolist()[0]
            for col in df.columns:
                df.at[idx, col] = new_row[col]
            print(f"Updated existing row for: {new_row['chunking_algo']}")
        else:
            # Add new row
            df = pd.concat([df, pd.DataFrame([new_row], columns=df.columns)], ignore_index=True)
            print(f"Added new row for: {new_row['chunking_algo']}")
        
        # Ensure correct data types
        df = df.astype({
            'chunking_algo': str,
            'embedding_seconds': float,
            'input_tokens': int,
            'invocation_count': int,
            'total_token_costs': float
        })
        
        return df
        
    except Exception as e:
        print(f"Error updating DataFrame: {str(e)}")
        return df


def embedding_cost_report(vector_store_embedding_cost, cost_for_notebook, scenario_number_of_documents, scenario_number_of_queries, notebook_number_of_documents=7):
    number_of_query_invocation = cost_for_notebook["invocation_count"] - vector_store_embedding_cost["invocation_count"]
    cost_of_query_embedding = cost_for_notebook["total token costs"]-vector_store_embedding_cost["total token costs"]
    cost_of_query_invocation = cost_for_notebook["total token costs"]-vector_store_embedding_cost["total token costs"]
    average_cost_of_query_embedding = cost_of_query_embedding/float(number_of_query_invocation)
    documents_multiplier = float(scenario_number_of_documents)/float(notebook_number_of_documents)
    scenario_vector_store_cost = round(vector_store_embedding_cost["total token costs"]*documents_multiplier,6)
    scenario_query_cost = round(average_cost_of_query_embedding*scenario_number_of_queries,6)
    md = f"""
#### Scenario
* Number of documents to ingest: {scenario_number_of_documents}
* Number of queries: {scenario_number_of_queries}

#### Cost Estimation based on the Scenario (USD)
|-| Notebook Cost | Scenario Cost |
|-|-|-|
|VectorStore|{round(vector_store_embedding_cost["total token costs"],6)}|{scenario_vector_store_cost}|
|Queries|{cost_of_query_invocation}|{scenario_query_cost}|
|**TOTAL**|{round(cost_for_notebook["total token costs"],6)}|{scenario_vector_store_cost+scenario_query_cost}|

#### The cost estimation is based on a scenario that the similar documents and queries are multiplied.
        """
    return md



######################################################################
#notebook 1.5 
######################################################################

from io import BytesIO
import zipfile
import boto3
import time
import json
import botocore
from botocore.exceptions import ClientError

def create_or_update_custom_chunking_lambda(region_name, account_number, role_name, function_name, s3_bucket):
    """
    Creates or updates a Lambda function for custom chunking with the necessary IAM role.
    
    Args:
        region_name (str): AWS region
        account_number (str): AWS account number
        role_name (str): Name for the IAM role
        function_name (str): Name for the Lambda function
        s3_bucket (str): S3 bucket name for permissions
        
    Returns:
        tuple: (role_arn, function_arn) containing the ARNs of the created/updated role and function
    """
    # Create IAM client
    iam = boto3.client("iam", region_name=region_name)
    lambda_client = boto3.client("lambda", region_name=region_name)
    
    # Try to get the IAM role if it exists
    try:
        # Check if the role already exists
        get_role_response = iam.get_role(RoleName=role_name)
        lambda_iam_role = get_role_response  # Store the entire response
        print(f"IAM role '{role_name}' already exists. Using the existing role.")
    except iam.exceptions.NoSuchEntityException:
        # Define the IAM assume role policy for the Lambda function
        assume_role_policy_document = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {
                        "Service": "lambda.amazonaws.com"
                    },
                    "Action": "sts:AssumeRole"
                }
            ]
        }
        
        # Convert the IAM assume role policy into JSON format
        assume_role_policy_document_json = json.dumps(assume_role_policy_document)
        
        # Create the IAM role for the Lambda function with the assume role policy
        lambda_iam_role = iam.create_role(
            RoleName=role_name,
            AssumeRolePolicyDocument=assume_role_policy_document_json
        )
        print(f"Created new IAM role: {role_name}")

    # Always put the policy (it will update if it exists or create if it doesn't)
    iam.put_role_policy(
        RoleName=role_name,
        PolicyName="s3policy",
        PolicyDocument=json.dumps(
            {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": [
                            "s3:GetObject",
                            "s3:ListBucket", 
                            "s3:PutObject"
                        ],
                        "Resource": [
                            f"arn:aws:s3:::{s3_bucket}-custom-chunk",
                            f"arn:aws:s3:::{s3_bucket}-custom-chunk/*"
                        ],
                        "Condition": {
                            "StringEquals": {
                                "aws:ResourceAccount": f"{account_number}"
                            }
                        }
                    }
                ]
            }
        )
    )

    # Prepare the Lambda function code by creating a ZIP file
    s = BytesIO()
    z = zipfile.ZipFile(s, 'w')
    z.write("lambda_function.py")
    z.close()
    zip_content = s.getvalue()

    # Sleep for 10 seconds to ensure resources are available
    time.sleep(10)

    # Get the role ARN
    role_arn = lambda_iam_role["Role"]["Arn"]

    # Check if the Lambda function already exists
    try:
        lambda_client.get_function(FunctionName=function_name)
        print(f"Lambda function '{function_name}' already exists. Updating code...")
        
        # Update existing function code
        lambda_function = lambda_client.update_function_code(
            FunctionName=function_name,
            ZipFile=zip_content
        )
        print("Lambda function code updated successfully")
    except lambda_client.exceptions.ResourceNotFoundException:
        print(f"Creating new Lambda function: {function_name}")
        
        # Create the Lambda function
        lambda_function = lambda_client.create_function(
            FunctionName=function_name,
            Runtime='python3.12',
            Timeout=60,
            Role=role_arn,
            Code={'ZipFile': zip_content},
            Handler='lambda_function.lambda_handler'
        )
        print("Lambda function created successfully")
    
    # Get the function ARN
    function_arn = f"arn:aws:lambda:{region_name}:{account_number}:function:{function_name}"
    
    return role_arn, function_arn

def create_custom_chunk_s3_bucket(s3_bucket, region_name):
    """
    Creates an S3 bucket for custom chunking if it doesn't exist.
    
    Args:
        s3_bucket (str): Base S3 bucket name
        region_name (str): AWS region
        
    Returns:
        str: The created bucket name
    """
    # Create an S3 client to interact with the AWS S3 service in the specified region
    s3 = boto3.client("s3", region_name=region_name)
    
    # Create the custom chunk bucket name
    bucket_name = f"{s3_bucket}-custom-chunk"

    try:
        # Check if the bucket already exists by sending a HEAD request to S3
        s3.head_bucket(Bucket=bucket_name)
        # If the bucket exists, print a message
        print(f"Bucket '{bucket_name}' already exists.")
    except:
        # If the bucket does not exist, create a new one
        s3.create_bucket(
            Bucket=bucket_name, 
            CreateBucketConfiguration={'LocationConstraint': region_name}
        )
        # Print a message indicating the bucket has been created
        print(f"Bucket '{bucket_name}' created.")
    
    return bucket_name

def create_custom_data_source_for_kb(kb, variables, data_source_name, function_arn):
    """
    Creates a data source with custom transformation configuration for a knowledge base.
    
    Args:
        kb (dict): Knowledge base configuration
        variables (dict): Configuration variables
        data_source_name (str): Name for the data source
        function_arn (str): ARN of the Lambda function for custom chunking
        
    Returns:
        dict: Created data source object
    """
    # Initialize Bedrock clients
    bedrock_agent = boto3.client("bedrock-agent", region_name=variables["regionName"])
    
    # Get the knowledge base ID
    kb_id = kb['knowledgeBaseId']
    
    # Define custom transformation configuration
    custom_transformation_configuration = {
        "intermediateStorage": {
            "s3Location": {
                "uri": f"s3://{variables['s3Bucket']}-custom-chunk/"
            }
        },
        "transformations": [
            {
                "transformationFunction": {
                    "transformationLambdaConfiguration": {
                        "lambdaArn": function_arn
                    }
                },
                "stepToApply": "POST_CHUNKING"
            }
        ]
    }
    
    # Define S3 configuration
    s3_configuration = {
        "bucketArn": f"arn:aws:s3:::{variables['s3Bucket']}",
        "inclusionPrefixes": ["data"]
    }
    
    # Check if data source already exists and delete if needed
    try:
        print(f"Checking for existing data sources in knowledge base {kb_id}...")
        list_ds_response = bedrock_agent.list_data_sources(knowledgeBaseId=kb_id)
        
        existing_ds = None
        for ds in list_ds_response.get('dataSourceSummaries', []):
            if ds['name'] == data_source_name:
                existing_ds = ds
                break
        
        if existing_ds:
            print(f"Found existing data source '{data_source_name}'. Deleting it...")
            bedrock_agent.delete_data_source(
                knowledgeBaseId=kb_id,
                dataSourceId=existing_ds["dataSourceId"]
            )
            print("Waiting for data source deletion to complete...")
            time.sleep(20)
            print("Data source deleted.")
            
    except Exception as e:
        print(f"Error while checking or deleting data source: {e}")

    # Create the new data source
    try:
        print(f"Creating new data source '{data_source_name}' with custom chunking...")
        create_ds_response = bedrock_agent.create_data_source(
            name=data_source_name,
            description="A data source for Advanced RAG workshop",
            knowledgeBaseId=kb_id,
            dataSourceConfiguration={
                "type": "S3",
                "s3Configuration": s3_configuration
            },
            vectorIngestionConfiguration={
                "chunkingConfiguration": {"chunkingStrategy": "NONE"},
                "customTransformationConfiguration": custom_transformation_configuration
            }
        )
        
        ds_custom_chunk = create_ds_response["dataSource"]
        ds_id = ds_custom_chunk["dataSourceId"]
        print(f"Custom chunking data source created successfully with ID: {ds_id}")
        
    except ClientError as e:
        error_code = e.response.get('Error', {}).get('Code', '')
        if error_code == 'ConflictException':
            print(f"Data source '{data_source_name}' already exists. Retrieving it...")
            list_ds_response = bedrock_agent.list_data_sources(knowledgeBaseId=kb_id)
            for ds in list_ds_response.get('dataSourceSummaries', []):
                if ds['name'] == data_source_name:
                    ds_custom_chunk = ds
                    ds_id = ds['dataSourceId']
                    print(f"Retrieved existing data source with ID: {ds_id}")
                    break
        else:
            print(f"Error creating data source: {e}")
            raise
    
    return ds_custom_chunk
######################################################################
#notebook 1.6
######################################################################

def get_default_generation_config(max_tokens=4096, temperature=0.2, top_p=0.5, stop_sequences=None):
    """
    Creates a default generation configuration for Bedrock retrieve and generate.
    
    Args:
        max_tokens (int): Maximum number of tokens in the generated response
        temperature (float): Controls randomness (lower values = more deterministic output)
        top_p (float): Controls diversity of output by considering top P probability mass
        stop_sequences (list): List of sequences that indicate stopping points
    
    Returns:
        dict: Generation configuration dictionary
    """
    if stop_sequences is None:
        stop_sequences = []
        
    return {
        'inferenceConfig': {
            'textInferenceConfig': {
                'maxTokens': max_tokens,
                'stopSequences': stop_sequences,
                'temperature': temperature,
                'topP': top_p
            }
        }
    }

def add_prompt_template(generation_config, prompt_template=None):
    """
    Adds a prompt template to a generation configuration if provided.
    
    Args:
        generation_config (dict): Base generation configuration
        prompt_template (str): Prompt template to add to the configuration
        
    Returns:
        dict: Updated generation configuration
    """
    if prompt_template:
        new_config = generation_config.copy()
        new_config["promptTemplate"] = {"textPromptTemplate": prompt_template}
        return new_config
    return generation_config

def retrieve_and_generate(query, kb_id, model_id, number_of_results=5, 
                         generation_config=None, prompt_template=None, region_name="us-west-2"):
    """
    Performs retrieval-augmented generation (RAG) using Amazon Bedrock.
    
    Args:
        query (str): The user's query
        kb_id (str): Knowledge base ID
        model_id (str): Bedrock model ARN
        number_of_results (int): Number of relevant documents to retrieve
        generation_config (dict): Generation configuration (if None, uses default)
        prompt_template (str): Optional prompt template
        region_name (str): AWS region name
        
    Returns:
        dict: The RAG response
    """
    # Initialize the Bedrock Agent Runtime client
    bedrock_agent_runtime = boto3.client("bedrock-agent-runtime", region_name=region_name)
    
    # Use default generation config if none provided
    if generation_config is None:
        generation_config = get_default_generation_config()
    
    # Add prompt template if provided
    if prompt_template:
        generation_config = add_prompt_template(generation_config, prompt_template)
    
    # Perform retrieval-augmented generation
    response = bedrock_agent_runtime.retrieve_and_generate(
        input={
            "text": query
        },
        retrieveAndGenerateConfiguration={
            "type": "KNOWLEDGE_BASE",
            "knowledgeBaseConfiguration": {
                'knowledgeBaseId': kb_id,
                "modelArn": model_id,
                "generationConfiguration": generation_config,
                "retrievalConfiguration": {
                    "vectorSearchConfiguration": {
                        "numberOfResults": number_of_results
                    } 
                }
            }
        }
    )
    
    return response

def display_rag_results(response, show_citations=False, format_as_markdown=False):
    """
    Displays the results of a RAG query.
    
    Args:
        response (dict): The RAG response
        show_citations (bool): Whether to show citations
        format_as_markdown (bool): Whether to format the output as Markdown
        
    Returns:
        None
    """
    print('----------------- Answer ---------------------')
    
    output_text = response['output']['text']
    
    if format_as_markdown:
        # Replace $ with \$ to prevent Markdown interpretation issues
        output_text = output_text.replace("$", "\\$")
        display(Markdown(output_text))
    else:
        print(output_text, end='\n' * 2)
    
    if show_citations:
        print('----------------- Citations ------------------')
        print(json.dumps(response, indent=2))

def get_model_arn(base_model_id, account_number, region_name="us-west-2"):
    """
    Constructs the ARN for a Bedrock model.
    
    Args:
        base_model_id (str): Base model ID (e.g., 'us.amazon.nova-lite-v1:0')
        account_number (str): AWS account number
        region_name (str): AWS region name
        
    Returns:
        str: The complete model ARN
    """
    return f"arn:aws:bedrock:{region_name}:{account_number}:inference-profile/{base_model_id}"

def compare_rag_results(query, kb_ids, model_id, number_of_results=5, 
                       generation_config=None, prompt_template=None, region_name="us-west-2"):
    """
    Compares RAG results from different knowledge bases.
    
    Args:
        query (str): The user's query
        kb_ids (list): List of knowledge base IDs to compare
        model_id (str): Bedrock model ARN
        number_of_results (int): Number of relevant documents to retrieve
        generation_config (dict): Generation configuration
        prompt_template (str): Optional prompt template
        region_name (str): AWS region name
        
    Returns:
        list: List of RAG responses
    """
    results = []
    
    for kb_id in kb_ids:
        response = retrieve_and_generate(
            query=query,
            kb_id=kb_id,
            model_id=model_id,
            number_of_results=number_of_results,
            generation_config=generation_config,
            prompt_template=prompt_template,
            region_name=region_name
        )
        results.append(response)
    
    return results
######################################################################
#notebook 1.7
######################################################################
# SageMaker integration functions for advanced_rag_utils.py

import json
import time
import boto3
from botocore.client import Config
import warnings

def install_rag_dependencies():
    """
    Install dependencies required for RAG with SageMaker.
    """
    import subprocess
    warnings.filterwarnings("ignore")
    subprocess.check_call(["pip", "install", "-Uq", "sagemaker", "boto3", "langchain-aws"])

def get_or_deploy_sagemaker_endpoint(model_id, model_version, instance_type, region_name="us-west-2"):
    """
    Deploy a SageMaker endpoint or find an existing one.
    
    Args:
        model_id (str): JumpStart model ID
        model_version (str): Model version
        instance_type (str): SageMaker instance type
        region_name (str): AWS region name
        
    Returns:
        str: Endpoint name
    """
    # Initialize SageMaker client
    sagemaker_client = boto3.client('sagemaker', region_name=region_name)
    
    # Generate timestamp-based endpoint name
    timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
    endpoint_name = f"endpoint-llama-3-2-3b-instruct-{timestamp}"
    
    # First check for any existing endpoints
    llm_endpoint_name = None
    try:
        endpoints = sagemaker_client.list_endpoints()
        model_id_suffix = model_id.split('/')[-1]
        for endpoint in endpoints['Endpoints']:
            if model_id_suffix in endpoint['EndpointName']:
                llm_endpoint_name = endpoint['EndpointName']
                print(f"Found existing endpoint: {llm_endpoint_name}")
                break
    except Exception as e:
        print(f"Error checking for existing endpoints: {e}")
    
    # If no existing endpoint found, try to deploy a new one
    if not llm_endpoint_name:
        try:
            # Import JumpStartModel here to avoid loading it unless needed
            from sagemaker.jumpstart.model import JumpStartModel
            
            # Load the JumpStart model
            llm_model = JumpStartModel(model_id=model_id, model_version=model_version, instance_type=instance_type)
            
            # Deploy the model
            llm_endpoint = llm_model.deploy(
                accept_eula=True,
                initial_instance_count=1,
                endpoint_name=endpoint_name
            )
            llm_endpoint_name = llm_endpoint.endpoint_name
            print(f"Deployed new endpoint: {llm_endpoint_name}")
        except Exception as e:
            print(e)
            print("New endpoint cannot be created. Looking for any existing endpoints...")
            
            # Try again to find any existing endpoint if deployment failed
            try:
                endpoints = sagemaker_client.list_endpoints()
                model_id_suffix = model_id.split('/')[-1]
                for endpoint in endpoints['Endpoints']:
                    if model_id_suffix in endpoint['EndpointName']:
                        llm_endpoint_name = endpoint['EndpointName']
                        print(f"Using existing endpoint as fallback: {llm_endpoint_name}")
                        break
            except Exception:
                pass
    
    return llm_endpoint_name

def create_sagemaker_content_handler():
    """
    Create a content handler for SageMaker LLM endpoint.
    
    Returns:
        ContentHandler: The configured content handler
    """
    from langchain_aws.llms.sagemaker_endpoint import LLMContentHandler
    
    class ContentHandler(LLMContentHandler):
        # Specify content type for input and output
        content_type = "application/json"
        accepts = "application/json"

        # Method to transform user input into the format expected by SageMaker
        def transform_input(self, prompt: str, model_kwargs: dict) -> bytes:
            input_str = json.dumps({"inputs": prompt, "parameters": model_kwargs})  # Format input as JSON
            return input_str.encode("utf-8")  # Encode to bytes

        # Method to process the output from SageMaker
        def transform_output(self, output: bytes) -> str:
            response_json = json.loads(output.read().decode("utf-8"))  # Decode response JSON
            return response_json["generated_text"]  # Extract the generated text from response
    
    return ContentHandler()

def create_bedrock_retriever(kb_id, number_of_results=3, region_name="us-west-2"):
    """
    Create a retriever for Bedrock KB.
    
    Args:
        kb_id (str): Knowledge base ID
        number_of_results (int): Number of results to retrieve
        region_name (str): AWS region name
        
    Returns:
        AmazonKnowledgeBasesRetriever: The configured retriever
    """
    from langchain_aws.retrievers import AmazonKnowledgeBasesRetriever
    
    # Initialize the retriever to fetch relevant documents from the Amazon Knowledge Base
    retriever = AmazonKnowledgeBasesRetriever(
        knowledge_base_id=kb_id,  # Specify the Knowledge Base ID to retrieve data from
        region_name=region_name,  # Define the AWS region where the Knowledge Base is located
        retrieval_config={
            "vectorSearchConfiguration": {
                "numberOfResults": number_of_results  # Set the number of relevant documents to retrieve
            }
        },
    )
    
    return retriever

def get_llama_prompt_template(template_text=None):
    """
    Get a prompt template for Llama models.
    
    Args:
        template_text (str): Optional custom template text
        
    Returns:
        str: The prompt template
    """
    if template_text:
        return template_text
    
    return """
<|begin_of_text|>
<|start_header_id|>system<|end_header_id|>
You are an assistant for question-answering tasks. Answer the following question using the provided context. If you don't know the answer, just say "I don't know.".
<|start_header_id|>user<|end_header_id|>
Context: {context} 
Question: {question}
<|start_header_id|>assistant<|end_header_id|> 
Answer:
"""

def create_sagemaker_llm(endpoint_name, generation_config, content_handler=None, region_name="us-west-2"):
    """
    Create a SageMaker LLM.
    
    Args:
        endpoint_name (str): SageMaker endpoint name
        generation_config (dict): Generation configuration
        content_handler (ContentHandler): Optional content handler
        region_name (str): AWS region name
        
    Returns:
        SagemakerEndpoint: The configured LLM
    """
    from langchain_aws.llms import SagemakerEndpoint
    
    # Create a SageMaker runtime client
    sagemaker_runtime = boto3.client("sagemaker-runtime", region_name=region_name)
    
    # If no content handler is provided, create one
    if content_handler is None:
        content_handler = create_sagemaker_content_handler()
    
    # Initialize the LLM with the SageMaker endpoint
    llm = SagemakerEndpoint(
        endpoint_name=endpoint_name,  # Specify the SageMaker endpoint name
        client=sagemaker_runtime,  # Attach the SageMaker runtime client
        model_kwargs=generation_config,  # Pass the model configuration parameters
        content_handler=content_handler,  # Use the custom content handler for formatting
    )
    
    return llm

def create_rag_chain(retriever, llm, prompt_template):
    """
    Create a RAG chain.
    
    Args:
        retriever: Document retriever
        llm: Language model
        prompt_template: Prompt template
        
    Returns:
        The configured RAG chain
    """
    from langchain_core.prompts import PromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough
    
    prompt = PromptTemplate.from_template(prompt_template)
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    qa_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return qa_chain

def invoke_rag_chain(chain, query):
    """
    Invoke a RAG chain with a query.
    
    Args:
        chain: The RAG chain
        query (str): The query
        
    Returns:
        str: The generated response
    """
    return chain.invoke(query)

def retrieve_from_bedrock(query, kb_id, num_results=5, region_name="us-west-2"):
    """
    Retrieve relevant context from Bedrock Knowledge Base.
    
    Args:
        query (str): The query
        kb_id (str): Knowledge base ID
        num_results (int): Number of results to retrieve
        region_name (str): AWS region name
        
    Returns:
        list: List of retrieved document contents
    """
    # Initialize Bedrock client
    bedrock_agent_runtime = boto3.client("bedrock-agent-runtime", region_name=region_name)
    
    try:
        # Retrieve context based on the query using vector search configuration
        response = bedrock_agent_runtime.retrieve(
            knowledgeBaseId=kb_id,
            retrievalQuery={
                'text': query  # The query text to search in the knowledge base
            },
            retrievalConfiguration={
                'vectorSearchConfiguration': {
                    'numberOfResults': num_results  # Adjust based on the number of results required
                }
            }
        )
        # Extract the 'text' from the retrieval results and return as a list
        return [result['content']['text'] for result in response['retrievalResults']]
    except Exception as e:
        # Raise an error if the retrieval process fails
        raise RuntimeError(f"Bedrock retrieval failed: {str(e)}")

def format_prompt_for_llama(query, context):
    """
    Format prompt for Llama model.
    
    Args:
        query (str): The query
        context (str or list): The context (can be a string or a list of strings)
        
    Returns:
        str: The formatted prompt
    """
    # Convert context to string if it's a list
    if isinstance(context, list):
        context = "\n\n".join(context)
    
    # Format the complete prompt including system and user instructions
    return f"""
<|begin_of_text|>
<|start_header_id|>system<|end_header_id|>
You are an assistant for question-answering tasks. Answer the following question using the provided context. If you don't know the answer, just say "I don't know.".
<|start_header_id|>user<|end_header_id|>
Context: {context} 
Question: {query}
<|start_header_id|>assistant<|end_header_id|> 
Answer:
    """.strip()

def generate_response_from_sagemaker(prompt, endpoint_name, generation_config, region_name="us-west-2"):
    """
    Generate response from SageMaker endpoint.
    
    Args:
        prompt (str): The formatted prompt
        endpoint_name (str): SageMaker endpoint name
        generation_config (dict): Generation configuration
        region_name (str): AWS region name
        
    Returns:
        str: The generated response
    """
    # Initialize SageMaker runtime client
    runtime = boto3.client('sagemaker-runtime', region_name=region_name)
    
    # Prepare the payload with prompt and generation parameters
    payload = {
        "inputs": prompt,  # The formatted prompt to pass to the model
        "parameters": generation_config  # Additional parameters for the model
    }
    
    try:
        # Call the SageMaker endpoint to generate the response
        response = runtime.invoke_endpoint(
            EndpointName=endpoint_name,  # SageMaker endpoint name
            ContentType='application/json',  # Content type for the request
            Body=json.dumps(payload)  # Send the payload as JSON
        )

        # Parse the response body
        result = json.loads(response['Body'].read().decode("utf-8"))
        
        # Handle different response formats (list or dictionary)
        if isinstance(result, list):
            # If the result is a list, extract the generated text from the first element
            return result[0]['generated_text']
        elif 'generated_text' in result:
            # If the result is a dictionary with 'generated_text', return the generated text
            return result['generated_text']
        elif 'generation' in result:
            # Alternative format with 'generation' key
            return result['generation']
        else:
            # Raise an error if the response format is unexpected
            raise RuntimeError("Unexpected response format")
            
    except Exception as e:
        # Raise an error if the generation process fails
        raise RuntimeError(f"Generation failed: {str(e)}")

def setup_and_run_rag_with_sagemaker(query, kb_id, endpoint_name, generation_config, number_of_results=3, region_name="us-west-2"):
    """
    Setup and run RAG using boto3 directly.
    
    Args:
        query (str): The query
        kb_id (str): Knowledge base ID
        endpoint_name (str): SageMaker endpoint name
        generation_config (dict): Generation configuration
        number_of_results (int): Number of results to retrieve
        region_name (str): AWS region name
        
    Returns:
        tuple: (query, response)
    """
    # Retrieve context from Bedrock
    context = retrieve_from_bedrock(query, kb_id, number_of_results, region_name)
    
    # Format prompt
    prompt = format_prompt_for_llama(query, context)
    
    # Generate response
    response = generate_response_from_sagemaker(prompt, endpoint_name, generation_config, region_name)
    
    return query, response

def save_sagemaker_endpoint_to_variables(variables, endpoint_name, file_path="../variables.json"):
    """
    Save SageMaker endpoint name to variables JSON file.
    
    Args:
        variables (dict): Variables dictionary
        endpoint_name (str): SageMaker endpoint name
        file_path (str): Path to variables JSON file
        
    Returns:
        dict: Updated variables dictionary
    """
    updated_variables = {**variables, "sagemakerLLMEndpoint": endpoint_name}
    
    with open(file_path, "w") as f:
        json.dump(updated_variables, f, indent=2)
    
    return updated_variables

def get_default_sagemaker_generation_config(temperature=0, top_k=10, max_new_tokens=512, stop="<|eot_id|>"):
    """
    Get default generation configuration for SageMaker.
    
    Args:
        temperature (float): Temperature for generation
        top_k (int): Top-k for generation
        max_new_tokens (int): Maximum number of tokens to generate
        stop (str): Stop sequence
        
    Returns:
        dict: Generation configuration
    """
    return {
        "temperature": temperature,
        "top_k": top_k,
        "max_new_tokens": max_new_tokens,
        "stop": stop
    }

######################################################################
#notebook 1.8
######################################################################
# Query decomposition utilities to add to advanced_rag_utils.py

import boto3
import json
import warnings
from typing import List, Dict, Any, Optional, Tuple

def suppress_warnings():
    """Suppress all warnings."""
    warnings.filterwarnings('ignore')

def update_boto3():
    """Update boto3 to the latest version."""
    import subprocess
    subprocess.check_call(["pip", "install", "-U", "boto3"])

def get_kb_config(kb_id: str, account_number: str, region_name: str, number_of_results: int = 5) -> Dict[str, Any]:
    """
    Creates configuration for knowledge base retrievals.
    
    Args:
        kb_id (str): Knowledge base ID
        account_number (str): AWS account number
        region_name (str): AWS region name
        number_of_results (int): Number of results to retrieve
        
    Returns:
        dict: Configuration dictionary for knowledge base retrieval
    """
    model_id = f"arn:aws:bedrock:us-west-2:{account_number}:inference-profile/us.amazon.nova-lite-v1:0"
    
    generation_configuration = {
        'inferenceConfig': {
            'textInferenceConfig': {
                'maxTokens': 1024,
                'stopSequences': [],
                'temperature': 0.0,
                'topP': 0.2
            }
        }
    }
    
    return {
        "kb_id": kb_id,
        "model_id": model_id,
        "number_of_results": number_of_results,
        "generation_configuration": generation_configuration
    }

def retrieve_and_generate_basic(
    query: str, 
    kb_id: str, 
    model_id: str, 
    generation_configuration: Dict[str, Any], 
    number_of_results: int = 5,
    region_name: str = "us-west-2"
) -> Dict[str, Any]:
    """
    Perform basic RAG query without query decomposition.
    
    Args:
        query (str): The query to process
        kb_id (str): Knowledge base ID
        model_id (str): Model ARN
        generation_configuration (dict): Generation configuration
        number_of_results (int): Number of results to retrieve
        region_name (str): AWS region name
        
    Returns:
        dict: Response from Bedrock
    """
    bedrock_agent_runtime = boto3.client("bedrock-agent-runtime", region_name=region_name)
    
    response = bedrock_agent_runtime.retrieve_and_generate(
        input={
            "text": query
        },
        retrieveAndGenerateConfiguration={
            "type": "KNOWLEDGE_BASE",
            "knowledgeBaseConfiguration": {
                'knowledgeBaseId': kb_id,
                "modelArn": model_id,
                "generationConfiguration": generation_configuration,
                "retrievalConfiguration": {
                    "vectorSearchConfiguration": {
                        "numberOfResults": number_of_results
                    } 
                }
            }
        }
    )
    
    return response

def display_rag_results(response: Dict[str, Any], show_citations: bool = True) -> None:
    """
    Display RAG results from Bedrock.
    
    Args:
        response (dict): Response from Bedrock
        show_citations (bool): Whether to show citations
    """
    print('----------------- Answer ---------------------')
    print(response['output']['text'], end='\n'*2)
    
    if show_citations:
        print('----------------- Citations ------------------')
        print(json.dumps(response, indent=2))

def retrieve_and_generate_with_decomposition(
    query: str, 
    kb_id: str, 
    model_id: str, 
    generation_configuration: Dict[str, Any], 
    number_of_results: int = 5,
    region_name: str = "us-west-2"
) -> Dict[str, Any]:
    """
    Perform RAG query with query decomposition.
    
    Args:
        query (str): The query to process
        kb_id (str): Knowledge base ID
        model_id (str): Model ARN
        generation_configuration (dict): Generation configuration
        number_of_results (int): Number of results to retrieve
        region_name (str): AWS region name
        
    Returns:
        dict: Response from Bedrock
    """
    bedrock_agent_runtime = boto3.client("bedrock-agent-runtime", region_name=region_name)
    
    response = bedrock_agent_runtime.retrieve_and_generate(
        input={
            "text": query
        },
        retrieveAndGenerateConfiguration={
            "type": "KNOWLEDGE_BASE",
            "knowledgeBaseConfiguration": {
                'knowledgeBaseId': kb_id,
                "modelArn": model_id,
                "generationConfiguration": generation_configuration,
                "retrievalConfiguration": {
                    "vectorSearchConfiguration": {
                        "numberOfResults": number_of_results
                    } 
                },
                'orchestrationConfiguration': {
                    'queryTransformationConfiguration': {
                        'type': 'QUERY_DECOMPOSITION'
                    }
                }
            }
        }
    )
    
    return response

def create_langchain_content_handler():
    """
    Create a content handler for SageMaker LLM.
    
    Returns:
        ContentHandler: A content handler for SageMaker LLM
    """
    from langchain_aws.llms.sagemaker_endpoint import LLMContentHandler
    
    class ContentHandler(LLMContentHandler):
        content_type = "application/json"
        accepts = "application/json"
        
        def transform_input(self, prompt: str, model_kwargs: dict) -> bytes:
            input_str = json.dumps({"inputs": prompt, "parameters": model_kwargs})
            return input_str.encode("utf-8")

        def transform_output(self, output: bytes) -> str:
            response_json = json.loads(output.read().decode("utf-8"))
            return response_json["generated_text"]
    
    return ContentHandler()

def setup_langchain_components(
    endpoint_name: str, 
    kb_id: str, 
    region_name: str, 
    number_of_results: int = 5,
    generation_config: Optional[Dict[str, Any]] = None
) -> Tuple[Any, Any]:
    """
    Set up LangChain components for RAG.
    
    Args:
        endpoint_name (str): SageMaker endpoint name
        kb_id (str): Knowledge base ID
        region_name (str): AWS region name
        number_of_results (int): Number of results to retrieve
        generation_config (dict): Generation configuration
        
    Returns:
        tuple: (llm, retriever) - LangChain LLM and retriever components
    """
    from langchain_aws.llms import SagemakerEndpoint
    from langchain_aws.retrievers import AmazonKnowledgeBasesRetriever
    
    # Set default generation config if not provided
    if generation_config is None:
        generation_config = {
            "temperature": 0,
            "top_p": 0.3,
            "max_new_tokens": 512,
            "stop": ["<|eot_id|>"]
        }
    
    content_handler = create_langchain_content_handler()
    sagemaker_runtime = boto3.client("sagemaker-runtime", region_name=region_name)
    
    llm = SagemakerEndpoint(
        endpoint_name=endpoint_name,
        client=sagemaker_runtime,
        model_kwargs=generation_config,
        content_handler=content_handler,
    )
    
    retriever = AmazonKnowledgeBasesRetriever(
        knowledge_base_id=kb_id,
        region_name=region_name,
        retrieval_config={"vectorSearchConfiguration": {"numberOfResults": number_of_results}},
    )
    
    return llm, retriever

def create_qa_chain(retriever, llm, custom_prompt=None):
    """
    Create a QA chain using LangChain.
    
    Args:
        retriever: LangChain retriever
        llm: LangChain LLM
        custom_prompt (str): Optional custom prompt template
        
    Returns:
        chain: LangChain QA chain
    """
    from langchain_core.output_parsers import StrOutputParser
    from langchain.prompts import PromptTemplate
    from langchain_core.runnables import RunnablePassthrough
    
    if custom_prompt is None:
        prompt = PromptTemplate.from_template(
            """
            Human:

            You are an assistant who answers questions using following pieces of retrieved context only. 
            If you don't find the answer from the retrieved context, do not include it and just say you don't know about it.

            Question: {question}

            Context: {context}

            Answer: Based on the context given, my answer for your question is as following:
            """)
    else:
        prompt = PromptTemplate.from_template(custom_prompt)
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    qa_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return qa_chain

def setup_agentic_rag(retriever, llm):
    """
    Set up agentic RAG with query decomposition.
    
    Args:
        retriever: LangChain retriever
        llm: LangChain LLM
        
    Returns:
        agent: LangChain agent
    """
    from langchain.agents import AgentType, initialize_agent, Tool
    
    def fn_search(question):
        """
        Search the answer of the question from the knowledge base. 
        """
        chunks = [doc.page_content for doc in retriever.invoke(question)]
        return chunks
    
    def noop(input):
        """Use this when no action need to be taken for your thought."""
        return
    
    kb_tool_finance = Tool(
        name="SearchFinancialStatements",
        func=fn_search,
        description="Use this tool to find answers for financial data."
    )
    
    kb_tool_technology = Tool(
        name="SearchTechnologyDocuments",
        func=fn_search,
        description="Use this tool to find answers for technologies."
    )
    
    noop_tool = Tool(
        name="None",
        func=noop,
        description="Use this when no action need to be taken for your thought"
    )
    
    tools = [kb_tool_finance, kb_tool_technology, noop_tool]
    
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent_type=AgentType.SELF_ASK_WITH_SEARCH,
        verbose=True,
        handle_parsing_errors=True
    )
    
    return agent

def create_output_parser():
    """
    Create a custom output parser for LangChain.
    
    Returns:
        parser: Custom output parser
    """
    from langchain_core.exceptions import OutputParserException
    from langchain_core.output_parsers import BaseOutputParser
    
    class CustomOutputParser(BaseOutputParser):
        """Custom parser."""
        
        def parse(self, text: str):
            print(text)
            return text
        
        @property
        def _type(self) -> str:
            return "custome_output_text"
    
    return CustomOutputParser()
