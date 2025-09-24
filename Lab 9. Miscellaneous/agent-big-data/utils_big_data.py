
# Import required libraries
import os, time, boto3, json
from strands import Agent, tool
from strands.models import BedrockModel
from datetime import datetime
from pydantic import BaseModel, Field
from typing import Optional, List
from pprint import pprint

# Bypass tool consent for automated execution
os.environ["BYPASS_TOOL_CONSENT"] = "true"
# Specify that if python_repl tool is used, it shouldnt wait for user interaction
os.environ["PYTHON_REPL_INTERACTIVE"] = "False"

# model3 = "us.anthropic.claude-3-7-sonnet-20250219-v1:0
# model4 = "us.anthropic.claude-sonnet-4-20250514-v1:0"

# Let's define a helper function that will help us print input and output tokens to LLM
def print_tokens_costs(agent_response):
    
    pprint(agent_response.metrics.accumulated_metrics)
    pprint(agent_response.metrics.accumulated_usage)

    seconds = agent_response.metrics.accumulated_metrics['latencyMs']/1000

    inputTokens = agent_response.metrics.accumulated_usage['inputTokens']
    inputCosts_per_M = 3.00
    inputToken_costs = (inputTokens/1000000)*inputCosts_per_M

    outputTokens = agent_response.metrics.accumulated_usage['outputTokens']
    outputCosts_per_M = 15.00
    outputToken_costs = (outputTokens/1000000)*outputCosts_per_M

    totalTokenCosts = inputToken_costs+outputToken_costs

    print(f"Time to research = {seconds} seconds")
    print(f"Input Token Costs = ${inputToken_costs};\nOutput Token Costs = ${outputToken_costs}\nTotal Token Costs = ${totalTokenCosts}")

def print_tokens_costs2(agent_response, aws_region = "us-west-2", model_id = "us.anthropic.claude-3-7-sonnet-20250219-v1:0"):
    #  Create MCP client for AWS Pricing MCP  server
    pricing_mcp_client = MCPClient(lambda: stdio_client(
        StdioServerParameters(
            command="uvx",  # Use uvx to run the MCP server
            args= [
                "awslabs.aws-pricing-mcp-server@latest",
                "--allow-write",  # Enable write operations
            ],
            env= {
                "FASTMCP_LOG_LEVEL": "ERROR",  # Minimize logging noise
                "AWS_REGION": aws_region    # Set AWS region
        }
        )
    ))

    with pricing_mcp_client :
        tools = pricing_mcp_client.list_tools_sync()    
        cost_agent = Agent(model="openai.gpt-oss-120b-1:0", tools=tools)
        response = cost_agent(f"""
        Calculate the cost of input tokens and output tokens for the Bedrock Model {model_id} in AWS region {aws_region} using the information in: {agent_response.metrics.accumulated_usage}.
        You can find latency information in {agent_response.metrics.accumulated_metrics}
        You can find tools information in {agent_response.metrics.tool_metrics.keys}
        Return the results in structured form below:
        input tokens (int): number of input tokens
        output tokens (int):  number of output tokens
        input costs (float):  cost of input tokens
        output costs (float):  cost of output tokens        
        total costs (float):  total costs which is a sum of input and output token costs
        total costs for 1000 such queries (float): total costs for 1000 such queries
        latency (seconds): latency in seconds
        tool count (int): number of tools used
        """)
        return response

def read_config_data() :
    # We could have stored the info in a json file and read it into a dictionary.
    # But let's show how to store configuration info in natural language and extract required information from it to a dictionary using an agent.

    class ConfigurationData(BaseModel):
        """configuration info for S3, region, role, folders"""
        s3_bucket_name: str = Field(description="The S3 bucket name")
        s3_folder_name: str = Field(description="The S3 folder path where data files are stored")
        aws_region: str = Field(description="AWS region name")
        aws_role_name: str = Field(description="AWS role name")        
        s3_bucket_for_athena_output: str = Field(description="The S3 bucket name for athena output")
        kb_id: Optional[str] = Field(default=None, description="The Bedrock knowledge base id")

    # Initialize Claude 3.7 Sonnet model via Bedrock
    model = BedrockModel(model_id="us.anthropic.claude-3-7-sonnet-20250219-v1:0", temperature=0.1)  # Low temperature for consistent structured output

    # Create the Agent.
    config_agent = Agent(model=model)

    #read the contents of the file
    input_file_name = "inputs.txt"
    # read the file content using with
    with open(input_file_name, 'r') as f:
        input_file_contents = f.read()

    #Pass the file content to the agent and ask the agent to extract the structured info.
    config_data = config_agent.structured_output(
        ConfigurationData, 
        f"Extract the information strictly in structured format from {input_file_contents}"
    )
    # The above pattern is very popular when you need th agent to return data in a structured format.
    # Many customers use this while extracting specific entities from resumes, medical, legal, insurance, or financial documents.

    print(config_data)
    # config_data now has the info that we need.

    #covert this into a dictionary
    config_dict = config_data.model_dump()
    return config_dict

def load_system_prompt_from_file(file_path: str, **variables) -> str:
    """
    Load system prompt from a text file and substitute variables.
    
    Args:
        file_path (str): Path to the text file in the curremt folder containing the prompt template
        **variables: Keyword arguments for variable substitution
        
    Returns:
        str: The formatted system prompt with variables substituted
        
    Example:
        kb_system_prompt = load_system_prompt_from_file(
            "kb_system_prompt.txt", 
            config_dict=config_dict
        )
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            prompt_template = file.read()
        
        # Use format() to substitute variables
        formatted_prompt = prompt_template.format(**variables)
        return formatted_prompt
        
    except FileNotFoundError:
        raise FileNotFoundError(f"Prompt file not found: {file_path}")
    except KeyError as e:
        raise KeyError(f"Missing variable for prompt template: {e}")
    except Exception as e:
        raise Exception(f"Error loading prompt from file: {e}")