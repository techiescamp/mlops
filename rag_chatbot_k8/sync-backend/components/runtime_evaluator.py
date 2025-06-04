from typing import List, Dict
from .utils import compute_semantic_similarity, compute_text_overlap

class RuntimeEvaluator:
    def __init__(self, embedding_model, threshold: float=0.6):
        self.embedding_model = embedding_model
        self.threshold = threshold

    def _get_relevant_documents(self, query: str, docs: List[Dict]) -> List[str]:
        """ Identify relevant documents based on semantic similarity. """
        relevant_sources = []
        for doc in docs:
            similarity = compute_semantic_similarity(query, doc.page_content, self.embedding_model)
            if similarity > self.threshold:
                relevant_sources.append(doc.metadata.get('source', ''))
        print(f"Relevant documents: {relevant_sources}")
        return relevant_sources
    
    def _get_query_doc_similarity(self, query: str, docs: List[Dict]) -> float:
        """ claculate average query-document similarity. """
        similarities = []
        for doc in docs:
            similarity = compute_semantic_similarity(query, doc.page_content, self.embedding_model)
            similarities.append(similarity)
        avg_similarity = sum(similarities) / len(similarities) if similarities else 0.0
        print(f"Average query-document similarity: {avg_similarity}")
        return avg_similarity

    def _get_document_diversity(self, docs: List[Dict]) -> float:
        """ Calculate document diversity based on unique filenames in metadata. """
        if len(docs) < 2:
            return 1.0  # No diversity if less than 2 documents
        similarites = []
        for i in range(len(docs)):
            for j in range(i+1, len(docs)):
                sim = compute_semantic_similarity(docs[i].page_content, docs[j].page_content, self.embedding_model)
                similarites.append(sim)
        avg_similarity = sum(similarites) / len(similarites) if similarites else 0.0
        return 1.0 - avg_similarity  # Diversity is 1 - average similarity
    

    def evaluate_retrieval(self, query: str, retrieved_docs: List[Dict], selected_docs: List[Dict]) -> Dict:
        """ Evaluate retrieval quality using query-document similarity and diversity. """
        k = len(retrieved_docs)
        relevant_sources = self._get_relevant_documents(query, selected_docs)
        retrieved_sources = [doc.metadata.get('source', '') for doc in retrieved_docs]
        relevant_set = set(relevant_sources)
        print('relevant_set: ', len(relevant_set))
        retrived_relevant = [source for source in retrieved_sources if source in relevant_set]

        # metrics
        precision = len(retrived_relevant) / k if k > 0 else 0
        recall = len(retrived_relevant) / len(relevant_set) if len(relevant_set) > 0 else 0
        mrr = 0
        for i, source in enumerate(retrieved_sources, 1):
            if source in relevant_set:
                mrr = 1 / i
                break
        # custom metrics
        avg_query_doc_similarity = self._get_query_doc_similarity(query, retrieved_docs)
        document_diversity = self._get_document_diversity(retrieved_docs)

        return {
            "retrieved_count_from_rag": k,
            "relevant_sources_count": len(relevant_sources),
            "relevant_sources": relevant_sources,
            "precision": precision,
            "recall": recall,
            "mrr": mrr,
            "avg_query_doc_similarity": avg_query_doc_similarity,
            "document_diversity": document_diversity
        }

        
    def evaluate_generation(self, query: str, answer: str, context: str) -> Dict:
        """Evaluate generation quality using context adherence and answer-query similarity."""
        out_of_context = "🌟" in answer
        context_adherence = compute_semantic_similarity(answer, context, self.embedding_model)
        answer_query_similarity = compute_semantic_similarity(answer, query, self.embedding_model)
        text_overlap = compute_text_overlap(answer, context)

        return {
            "context_adherence": context_adherence,
            "answer_query_similarity": answer_query_similarity,
            "text_overlap": text_overlap,
            "out_of_context": out_of_context
        }
    
    def evaluate(self, query: str, retrieved_docs: List[Dict], selected_docs: List[Dict], answer: str, context: str) -> Dict:
        """Evaluate the entire retrieval and generation process."""
        retrieval_metrics = self.evaluate_retrieval(query, retrieved_docs, selected_docs)
        generation_metrics = self.evaluate_generation(query, answer, context)

        return {
            "query": query,
            "retrieval": retrieval_metrics,
            "generation": generation_metrics
        }