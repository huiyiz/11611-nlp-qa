import re
import torch
import numpy as np
import random

#Utils for Answer generation
def extract_and_tokenize(context, answers, tokenizer, encoder):
    embeddings, tokenized_context, _ = tokenize_and_encode_context(context, tokenizer, encoder)
    binary_labels = generate_binary_labels(answers, tokenized_context, tokenizer)
    embeddings, binary_labels = extract_answer_token_and_nonanswer_token(embeddings, binary_labels)
    return embeddings, binary_labels

def tokenize_and_encode_context(context, tokenizer, encoder):
    tokenized_context = []
    embeddings = []
    for text in context:
        tokenization = tokenizer(text, return_tensors="pt", max_length = 512, 
                                 truncation = True, padding='max_length')
        context_token = tokenization['input_ids'][0].detach().numpy()
        tokenized_context.append(context_token)
        out = encoder(**tokenization, output_hidden_states=True)
        embedding = out.last_hidden_state.detach().numpy()
        assert(embedding.shape[1] == len(context_token))
        embeddings.append(embedding[0])
    return embeddings, tokenized_context, tokenization


def generate_binary_labels(answers, tokenized_context, tokenizer):
    binary_labels = []
    for k, answer in enumerate(answers):
        answer_texts = answer['text']
        context = tokenized_context[k]
        binary_label = np.zeros(len(context))
        for text in answer_texts:
            tokenization = tokenizer(text, return_tensors="pt")
            answer_token = tokenization['input_ids'][0].detach().numpy()[1:-1]
            answer_length = len(answer_token)
            
            for i in range(len(context) - answer_length + 1):
                if np.all(context[i:i+answer_length] == answer_token):
                    binary_label[i:i+answer_length] = 1
        binary_labels.append(binary_label)

    return binary_labels

def extract_answer_token_and_nonanswer_token(embeddings, binary_labels):
    single_embeddings = []
    single_labels = []
    for i, embedding in enumerate(embeddings):
        for k, token in enumerate(embedding):
            if binary_labels[i][k] == 1:
                single_embeddings.append(token)
                single_labels.append(1)
            else:
                if np.random.randint(0, 100, 1)[0] == 0:
                    single_embeddings.append(token)
                    single_labels.append(0)
      
    return single_embeddings, single_labels

def predict_answer(model, context, tokenizer, encoder, device):
    words_list, words_ids_list, words_mapping = context_to_words_inputids(context, tokenizer)
    embeddings, _, tokenization = tokenize_and_encode_context([context], tokenizer, encoder)
    predictions = model_predict(model, embeddings, device)
    input_ids = tokenization['input_ids'][0].detach().numpy().tolist()
    answer_ids_list = get_answer_ids(input_ids, predictions)
    answers = map_answer_ids_to_words(answer_ids_list, words_ids_list, words_mapping, words_list)
    return sorted(set(answers), key=answers.index)

def map_answer_ids_to_words(answer_ids_list, words_ids_list, words_mapping, words_list):
    text_answer_list = []
    for ans in answer_ids_list:
        ans_length = len(ans)
        for i in range(len(words_ids_list) - ans_length + 1):
            if words_ids_list[i:i+len(ans)] == ans:
                corresponding_words = words_mapping[i:i+len(ans)]
                corresponding_words = sorted(set(corresponding_words), key=corresponding_words.index)
                text_ans = ""
                for w in corresponding_words:
                    text_ans += ' ' + words_list[w]
                text_answer_list.append(text_ans[1:])
    return text_answer_list

def model_predict(model, embeddings, device):
    embeddings = torch.tensor(embeddings[0])
    embeddings = embeddings.to(device)

    predictions = model(embeddings).flatten()

    predictions[predictions >= 0.5] = 1
    predictions[predictions < 0.5] = 0
    predictions = predictions.cpu().detach().numpy().tolist()
    return predictions

def get_answer_ids(input_ids, predictions):
    assert(len(input_ids) == len(predictions))
    all_answers_ids = []
    
    i = 0
    while i < len(predictions) - 1:
        if predictions[i] == 1:
            ans_span = [input_ids[i]]
            j = i + 1
            while j < len(predictions):
                if predictions[j] == 1:
                    ans_span.append(input_ids[j])
                    j += 1
                else:
                    all_answers_ids.append(ans_span)
                    i = j
                    break
        else:
            i += 1
            
    return all_answers_ids

def context_to_words_inputids(context,  tokenizer):
    word_ids_list = []
    word_mapping = []
    words_list = context_to_words(context)
    for k, word in enumerate(words_list):
        word_id = tokenizer(word)['input_ids'][1:-1]
        word_ids_list.extend(word_id)
        word_mapping.extend([k for i in range(len(word_id))])
    return words_list, word_ids_list, word_mapping

def context_to_words(context):
    return re.sub("[^\w]", " ",  context).split()

#Utils for question generation
def get_question(answer, context, tokenizer2, QG_model, device, max_length=256):
    input_text = "answer: %s  context: %s" % (answer, context)
    features = tokenizer2([input_text], return_tensors='pt')

    QG_model.to(device)
    output = QG_model.generate(input_ids=features['input_ids'].to(device), 
                attention_mask=features['attention_mask'].to(device),
                max_length=max_length)
    question = tokenizer2.decode(output[0], skip_special_tokens=True)
    return question

def generate_questions(answers, context, tokenizer, model, device, max_len=256, N = 10):
    questions = []
    sub_answers = random.sample(answers, N)
    for i in range(len(sub_answers)):
        this_answer = sub_answers[i]
        pred_quesiton = get_question(this_answer, context, tokenizer, model, device, max_len)
        questions.append(pred_quesiton)
    if len(questions) > N:
        return questions[:N]
    else:
        return questions