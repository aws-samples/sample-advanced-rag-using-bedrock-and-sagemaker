from pydantic import BaseModel
from flotorch_core.chunking.chunking import Chunk

class Question(BaseModel):
    question: str
    answer: str

    def get_chunk(self) -> Chunk:
        return Chunk(data=self.question)

def rag_with_flotorch(exp_config_data, vector_storage, reranker, inferencer, questions: list[Question]):
    """
        Process a list of questions through the RAG pipeline.

        Args:
            vector_storage: Initialized vector storage 
            questions: List of Question objects to process

        Returns:
            List[Dict[str, Any]]: List of responses containing metadata and answers
            
        Raises:
            ValueError: If vector storage search fails
            RuntimeError: If text generation fails
    """
    responses_list = []
    for question in questions:
        try:
            inference_response = {}
            # fetch documents from bedrock knowledge bases
            question_chunk = question.get_chunk()
            response = vector_storage.search(question_chunk, int(exp_config_data.get("knn_num")))
            vector_response = response.to_json()['result']
            
            # rerank if selected
            if reranker:
                vector_response = reranker.rerank_documents(question_chunk.data, vector_response)
            
            # send for inferencing
            metadata, answer = inferencer.generate_text(question.question, vector_response)

            inference_response["metadata"] = metadata
            inference_response["generated_answer"] = answer
            inference_response["expected_answer"] = question.answer
            inference_response["question"] = question.question
            inference_response["retrieved_contexts"] = vector_response
            responses_list.append(inference_response)
        except Exception as e:
            # Log the error and continue with next question
            print(f"Error processing question: {question.question}. Error: {str(e)}")
            responses_list.append({
                "metadata": {"error": str(e)},
                "answer": "Failed to process question"
            })

    return responses_list