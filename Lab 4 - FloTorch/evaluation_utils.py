from typing import List, Optional, Dict
from flotorch_core.evaluator.evaluation_item import EvaluationItem

def convert_to_evaluation_dict(data: Dict) -> Dict[str, List[EvaluationItem]]:
    """
    Converts the given dictionary into a dictionary where the key is the KB type
    and the value is a list of EvaluationItem objects.  This version handles
    dynamic KB keys.

    Args:
        data: The input dictionary.

    Returns:
        A dictionary where the key is the KB type and the value is a list of EvaluationItem.
    """

    evaluation_dict: Dict[str, List[EvaluationItem]] = {}

    for kb_type, items in data.items():
        if isinstance(items, list):  # Ensure we're processing a list of items
            evaluation_items: List[EvaluationItem] = []
            for item_data in items:
                if isinstance(item_data, dict) and "question" in item_data and "expected_answer" in item_data and "generated_answer" in item_data and "retrieved_contexts" in item_data:
                    evaluation_item = EvaluationItem(
                        question=item_data["question"],
                        generated_answer=item_data["generated_answer"],
                        expected_answer=item_data["expected_answer"],
                        context=[context["text"] for context in item_data["retrieved_contexts"]]
                    )
                    evaluation_items.append(evaluation_item)
            evaluation_dict[kb_type] = evaluation_items

    return evaluation_dict