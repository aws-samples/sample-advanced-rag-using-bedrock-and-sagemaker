
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