# import external packages
from transformers import BertModel, AutoTokenizer, AutoModelForSeq2SeqLM, T5ForConditionalGeneration, T5TokenizerFast
import transformers
import torch
import argparse

# import self-implemented
from src.network import Network
from src.utils import *


transformers.logging.set_verbosity(transformers.logging.CRITICAL)


parser = argparse.ArgumentParser()
parser.add_argument("file_path", type=str, help="File path")
parser.add_argument("N_question", type=int, help="Number of questions")
args = parser.parse_args()
file_path = args.file_path
N_question = args.N_question

with open(file_path, "r") as f:
    context = f.read()

# initialize
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# print(device)
device = 'cpu'

# tokenizer1 = AutoTokenizer.from_pretrained(
#     "pretrained/tok1", local_files_only=True)
# encoder1 = BertModel.from_pretrained("pretrained/model1", local_files_only=True)

# tokenizer2 = AutoTokenizer.from_pretrained("pretrained/tok2", local_files_only=True)
# QG_model = AutoModelForSeq2SeqLM.from_pretrained("pretrained/model2", local_files_only=True)

# model config
config = {
    'batch_size': 512,
    'dropout_rate': 0.5,
    'learning_rate': 1e-1,
    'epochs': 10
}


# Answer Extraction
model_path = './trained_model/answer_extractor.pth'


# pred_answer = predict_answer(QE_model, context, tokenizer1, encoder1, device)

# Question Generation based on the answer
answer_agnositic_model = T5ForConditionalGeneration.from_pretrained("ThomasSimonini/t5-end2end-question-generation")
answer_agnositic_tokenizer = T5TokenizerFast.from_pretrained('t5-small', model_max_length=2048)

question_part1 = run_answer_agnostic_model(context, 2*N_question, answer_agnositic_tokenizer, answer_agnositic_model)

question_part2 = []

#balance the 2 types of questions
if len(question_part1) > N_question // 3:
    question_part1 = question_part1[ : N_question // 3]

tokenizer1 = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny", do_lower_case=True)
encoder1 = BertModel.from_pretrained("prajjwal1/bert-tiny")

MixQG_tokenizer = AutoTokenizer.from_pretrained('Salesforce/mixqg-base')
MixQG_model = AutoModelForSeq2SeqLM.from_pretrained('Salesforce/mixqg-base')
QE_model = Network(dropout=config['dropout_rate']).to(device)
QE_model.load_state_dict(torch.load(model_path)['model_state_dict'])
QE_model.eval()

question_part2 = run_answer_aware_model(context, device, N_question - len(question_part1) , QE_model, tokenizer1, encoder1, MixQG_tokenizer, MixQG_model)

questions = question_part1 + question_part2

for question in questions:
    print(question)