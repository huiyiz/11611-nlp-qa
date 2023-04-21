import re
import torch
import numpy as np
import random
import nltk
from nltk import word_tokenize
import os

nltk.data.path.append('/QG/nltk_data')

def split_into_sentences(text):
    alphabets= "([A-Za-z])"
    prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
    suffixes = "(Inc|Ltd|Jr|Sr|Co)"
    starters = "(Mr|Mrs|Ms|Dr|Prof|Capt|Cpt|Lt|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
    acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
    websites = "[.](com|net|org|io|gov|edu|me)"
    digits = "([0-9])"
    text = " " + text + "  "
    text = text.replace("\n"," ")
    text = re.sub(prefixes,"\\1<prd>",text)
    text = re.sub(websites,"<prd>\\1",text)
    text = re.sub(digits + "[.]" + digits,"\\1<prd>\\2",text)
    if "..." in text: text = text.replace("...","<prd><prd><prd>")
    if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
    text = re.sub("\s" + alphabets + "[.] "," \\1<prd> ",text)
    text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>",text)
    text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
    text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
    text = re.sub(" " + alphabets + "[.]"," \\1<prd>",text)
    if "”" in text: text = text.replace(".”","”.")
    if "\"" in text: text = text.replace(".\"","\".")
    if "!" in text: text = text.replace("!\"","\"!")
    if "?" in text: text = text.replace("?\"","\"?")
    text = text.replace(".",".<stop>")
    text = text.replace("?","?<stop>")
    text = text.replace("!","!<stop>")
    text = text.replace("<prd>",".")
    sentences = text.split("<stop>")
    sentences = sentences[:-1]
    sentences = [s.strip() for s in sentences]
    return sentences
    
def answer_agnostic_model_postprocess_output(output):
    output = output[0][0]
    if output[-1] != '?':
        output = output + '?'
    output = output.split('? ')
    for i in range(len(output)):
        if output[i][-1] != '?':
            output[i] = output[i] + '?'
    return output

def run_answer_agnostic_model(context, N, answer_agnositic_tokenizer, answer_agnositic_model, device):
    #将原文裁剪到只剩前N句话
    context_sentences = split_into_sentences(context)
    context_sentences = context_sentences[:N]
    context_short = ' '.join(context_sentences)

    generator_args = {
    "max_length": 2048,
    "num_beams": 4,
    "length_penalty": 1.5,
    "no_repeat_ngram_size": 10,
    "early_stopping": True,
    }
    context = "generate questions: " + context_short + " </s>"
    input_ids = answer_agnositic_tokenizer.encode(context, return_tensors="pt")
    input_ids = input_ids.to(device)
    res = answer_agnositic_model.generate(input_ids, **generator_args)
    output = answer_agnositic_tokenizer.batch_decode(res, skip_special_tokens=True)
    output = [item.split("<sep>") for item in output]
    output = answer_agnostic_model_postprocess_output(output)
    return output

def catagorize_answers(answers_sentences, question_content):
    cate_answers = dict()
    cate_answers['entity'] = []
    cate_answers['event'] = []
    cate_answers['number'] = []
    cate_answers['any'] = []

    for pair in answers_sentences:
        answer = pair[0]
        sentence = pair[1]
        tokens = word_tokenize(answer)
        tags = nltk.pos_tag(tokens, tagset = "universal")

        NOUN = False
        VERB = False
        NUM = None
        NUM_sentence = None

        for tag in tags:
            word = tag[0]
            POS = tag[1]

            if POS == 'NOUN':
                NOUN = True
            if POS == 'VERB':
                VERB = True
            if POS == 'NUM':
                NUM = word
                NUM_sentence = sentence
          
        if NOUN or VERB or (NUM != None):
            cate_answers['any'].append(pair)
        
        if VERB:
            cate_answers['event'].append(pair)
        elif NUM != None:
            cate_answers['number'].append((NUM, sentence))
        elif NOUN:
            cate_answers['entity'].append(pair)
            
    return cate_answers[question_content]

def format_type_inputs(answers_sentences, question_type):
    model_input = []
    if question_type == "Wh-question":
        for pair in answers_sentences:
            model_input.append(f"{pair[0]} \\n {pair[1]}")
    elif question_type == "TF-question":
        for pair in answers_sentences:
            model_input.append(f"{'yes'} {pair[0]} \\n {pair[1]}")
    elif question_type == "any":
        for pair in answers_sentences:
            model_input.append(f"{pair[0]} \\n {pair[1]}")
            model_input.append(f"{'yes'} {pair[0]} \\n {pair[1]}")
    return model_input

def sample_input(model_input, N):
    if len(model_input) > N:
        return list(np.random.choice(model_input, size = N, replace = False))
    else:
        return model_input

def align_answers_with_sentences(answers, sentences):
    answers_sentences = []
    for answer in answers:
        for i, sentence in enumerate(sentences):
            if answer in sentence:
                if i < 4:
                    answers_sentences.append((answer, " ".join(sentences[:i + 1])))
                else:
                    answers_sentences.append((answer, " ".join(sentences[i - 4 : i + 1]))) 
                break
    return answers_sentences

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

def tokenize_and_encode_context(context, tokenizer, encoder, device):
    tokenized_context = []
    embeddings = []
    for text in context:
        tokenization = tokenizer(text, return_tensors="pt", max_length = 512, 
                                 truncation = True, padding='max_length')
        context_token = tokenization['input_ids'][0].detach().numpy()
        tokenized_context.append(context_token)
        tokenization = tokenization.to(device)
        out = encoder(**tokenization, output_hidden_states=True)
        embedding = out.last_hidden_state.cpu().detach().numpy()
        assert(embedding.shape[1] == len(context_token))
        embeddings.append(embedding[0])
    return embeddings, tokenized_context, tokenization

def model_predict(model, embeddings, device):
    embeddings = torch.tensor(embeddings[0]).to(device)
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
            if (j >= (len(predictions) - 1)):
                i = j
        else:
            i += 1
            
    return all_answers_ids

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

def predict_answer(model, context, tokenizer, encoder, device):
    words_list, words_ids_list, words_mapping = context_to_words_inputids(context, tokenizer)
    embeddings, _, tokenization = tokenize_and_encode_context([context], tokenizer, encoder, device)
    predictions = model_predict(model, embeddings, device)
    input_ids = tokenization['input_ids'][0].cpu().detach().numpy().tolist()
    answer_ids_list = get_answer_ids(input_ids, predictions)
    answers = map_answer_ids_to_words(answer_ids_list, words_ids_list, words_mapping, words_list)
    answers = sorted(set(answers), key=answers.index)
    sentences = split_into_sentences(context)
    answers_sentences = align_answers_with_sentences(answers, sentences)
    return answers_sentences

def process_questions(questions):
    questions_m = []
    for question in questions:
        if question[-1] != '?':
            question = question[0].upper() + question[1:] + '?'
        questions_m.append(question)
    return questions_m

def get_question(MixQG_tokenizer, MixQG_model, model_input, device):
    input_ids = MixQG_tokenizer(model_input, return_tensors="pt", padding = True).input_ids
    input_ids = input_ids.to(device)
    generated_ids = MixQG_model.generate(input_ids, max_length=32, num_beams=4) 
    questions = MixQG_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    questions = process_questions(questions)
    return questions

def run_answer_aware_model(context, device, N , QE_model, tokenizer1, encoder1, MixQG_tokenizer, MixQG_model, question_type = 'any', question_content = 'any'):
    answers_sentences = predict_answer(QE_model, context, tokenizer1, encoder1, device)
    answers_sentences = catagorize_answers(answers_sentences, question_content)
    if len(answers_sentences) == 0:
        print("Cannot generate required questions!")
        return None

    assert question_type == 'any'
    answers_sentences = sample_answers(answers_sentences, N // 2 + 1)

    model_input = format_type_inputs(answers_sentences, question_type)
    model_input = sample_input(model_input, N)
    pred_quesitons = get_question(MixQG_tokenizer, MixQG_model, model_input, device)
    return pred_quesitons

def sample_answers(answers_sentences, N):
    inds = np.random.choice(len(answers_sentences), N, replace = False)
    rand_vals = [answers_sentences[ind] for ind in inds]   
    return rand_vals