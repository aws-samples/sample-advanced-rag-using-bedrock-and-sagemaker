import pandas as pd
from typing import Dict, Any, List
from decimal import Decimal
from dataclasses import dataclass

MILLION = 1_000_000
THOUSAND = 1_000
SECONDS_IN_MINUTE = 60
MINUTES_IN_HOUR = 60
# Read the CSV file into a pandas DataFrame
df = pd.read_csv('./data/bedrock_limits_small.csv')

@dataclass
class MetricsData:
    """Data class to store metrics information"""
    cost: Decimal
    latency: float
    input_tokens: int
    output_tokens: int

def extract_metadata_metrics(metadata: Dict[str, Any]) -> MetricsData:
    """
    Extract metrics from metadata dictionary
    
    Args:
        metadata: Dictionary containing metadata information
    Returns:
        MetricsData object with extracted metrics
    """
    return MetricsData(
        input_tokens=metadata.get("inputTokens", 0),
        output_tokens=metadata.get("outputTokens", 0),
        latency=float(metadata.get("latencyMs", 0)),
        cost=Decimal('0.0000')
    )

def calculate_bedrock_inference_cost(input_tokens,output_tokens, inference_model, aws_region):

    input_price = df[
        (df["model"] == inference_model) & (df["Region"] == aws_region)
        ]["input_price"]

    output_price = df[
        (df["model"] == inference_model) & (df["Region"] == aws_region)
        ]["output_price"]

    input_price_per_million_tokens = float(input_price.values[0])  # Price per million tokens
    output_price_per_million_tokens = float(output_price.values[0])  # Price per million tokens

    input_actual_cost = (input_price_per_million_tokens * float(input_tokens)) / MILLION
    output_actual_cost = (output_price_per_million_tokens * float(output_tokens)) / MILLION
    return input_actual_cost + output_actual_cost

def calculate_cost_and_latency_metrics(inference_data, inference_model, aws_region):
    if isinstance(inference_data, list):
        total_cost = Decimal('0.0000')
        total_latency = 0.0
        item_count = 0
    
        for item_data in inference_data:
            if not isinstance(item_data, dict) or "metadata" not in item_data:
                continue
    
            metrics = extract_metadata_metrics(item_data["metadata"])
            
            # Calculate cost for this item
            item_cost = calculate_bedrock_inference_cost(
                metrics.input_tokens,
                metrics.output_tokens,
                inference_model,
                aws_region
            )
            
            total_cost += Decimal(str(item_cost))
            total_latency += metrics.latency
            item_count += 1
    
        if item_count > 0:
            return {
                'inference_cost': float(total_cost),
                'average_inference_cost': float(total_cost / item_count),
                'latency': total_latency,
                'average_latency': total_latency / item_count,
                'processed_items': item_count
            }
        else:
            return {
                'cost': float(total_cost),
                'average_cost': float(total_cost / item_count),
                'latency': total_latency,
                'average_latency': total_latency / item_count,
                'processed_items': item_count
            }



