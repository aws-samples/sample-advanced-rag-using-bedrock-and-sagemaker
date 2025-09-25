# Deploy CloudFormation template to create S3 bucket and IAM role
# This is optional if you already have the required resources

import boto3
import time
import uuid
from datetime import datetime

# Configuration
# Random generated string 
SUFFIX = "workshop"
TEMPLATE_FILE = "big-data-agent-infrastructure.yaml"

def check_stack_exists(cf_client, stack_name):
    """Check if a CloudFormation stack exists"""
    try:
        cf_client.describe_stacks(StackName=stack_name)
        return True
    except cf_client.exceptions.ClientError as e:
        if 'does not exist' in str(e):
            return False
        raise e

def deploy_infrastructure(stack_name):
    """Deploy or update the CloudFormation stack"""
    cf_client = boto3.client('cloudformation')
    
    try:
        # Read the template file
        with open(TEMPLATE_FILE, 'r') as f:
            template_body = f.read()
        
        print(f"🚀 Deploying stack: {stack_name}")
        print(f"🏷️  Suffix: {SUFFIX}")
        
        # Check if stack exists
        stack_exists = check_stack_exists(cf_client, stack_name)
        
        if stack_exists:
            print(f"📝 Stack {stack_name} already exists. Updating...")
            
            # Update the stack
            try:
                response = cf_client.update_stack(
                    StackName=stack_name,
                    TemplateBody=template_body,
                    Parameters=[
                        {
                            'ParameterKey': 'Suffix',
                            'ParameterValue': SUFFIX
                        }
                    ],
                    Capabilities=['CAPABILITY_NAMED_IAM'],
                    Tags=[
                        {'Key': 'Purpose', 'Value': 'BigDataAgent'},
                        {'Key': 'Environment', 'Value': SUFFIX}
                    ]
                )
                
                print(f"✅ Stack update initiated. Stack ID: {response['StackId']}")
                
                # Wait for stack update to complete
                print("⏳ Waiting for stack update to complete...")
                waiter = cf_client.get_waiter('stack_update_complete')
                waiter.wait(
                    StackName=stack_name,
                    WaiterConfig={'Delay': 30, 'MaxAttempts': 20}
                )
                
                print("🎉 Stack updated successfully!")
                
            except cf_client.exceptions.ClientError as e:
                if 'No updates are to be performed' in str(e):
                    print("ℹ️  No updates needed - stack is already up to date")
                else:
                    raise e
        else:
            print(f"🆕 Creating new stack: {stack_name}")
            
            # Create the stack
            response = cf_client.create_stack(
                StackName=stack_name,
                TemplateBody=template_body,
                Parameters=[
                    {
                        'ParameterKey': 'Suffix',
                        'ParameterValue': SUFFIX
                    }
                ],
                Capabilities=['CAPABILITY_NAMED_IAM'],
                Tags=[
                    {'Key': 'Purpose', 'Value': 'BigDataAgent'},
                    {'Key': 'Environment', 'Value': SUFFIX}
                ]
            )
            
            print(f"✅ Stack creation initiated. Stack ID: {response['StackId']}")
            
            # Wait for stack creation to complete
            print("⏳ Waiting for stack creation to complete...")
            waiter = cf_client.get_waiter('stack_create_complete')
            waiter.wait(
                StackName=stack_name,
                WaiterConfig={'Delay': 30, 'MaxAttempts': 20}
            )
            
            print("🎉 Stack created successfully!")
        
        # Get stack outputs
        stack_info = cf_client.describe_stacks(StackName=stack_name)
        outputs = stack_info['Stacks'][0].get('Outputs', [])
        
        #print("\n📊 Stack Outputs:")
        #for output in outputs:
        #    print(f"  {output['OutputKey']}: {output['OutputValue']}")
            
        return outputs
        
    except Exception as e:
        print(f"❌ Error deploying/updating stack: {e}")
        return None

# Uncomment the line below to deploy the infrastructure
# stack_outputs = deploy_infrastructure()

print("💡 To deploy the infrastructure, uncomment the last line and run this cell.")
print(f"💡 This will create an S3 bucket and IAM role with suffix '{SUFFIX}'")