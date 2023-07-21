# Ensemble Question Generation for FiD-GAR Question Answering

## Code Implementation of the paper

For the QG part, we present a novel ensemble question generation (EQG) technique that overcomes the performance and flexibility constraints of existing methods. EQG consists of two synergistic channels, answer-agnostic and answer-aware, each contributing unique advantages. The answer-agnostic channel enhances our model's performance on conventional metrics, while the answer-aware channel enables users to specify the sentence and answer on which the question is based. By integrating these two channels, our system not only surpasses baseline performance but also offers flexibility for manual answer specification.

For the QA part, we proposed an approach to improve the performance of question answering by combining the idea in Generation-Augmented Retrieval (GAR) to Fusion-in-Decoder (FiD) models. We augmented the query by leveraging contextual word embeddings to retrieve highly relevant related passages. We fine-tuned the reader model using the retrieval results from GAR. We implemented query filtering to discard less relevant passages, avoid duplication, and provide more comprehensive contexts. We compared our model with an extractive baseline model, and our approach achieves higher accuracy in generating answers and provides more comprehensive contexts.
