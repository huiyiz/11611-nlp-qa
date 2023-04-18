from transformers import T5TokenizerFast, T5ForConditionalGeneration, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModel


answer_agnositic_model = T5ForConditionalGeneration.from_pretrained("ThomasSimonini/t5-end2end-question-generation")
answer_agnositic_tokenizer = T5TokenizerFast.from_pretrained('t5-small', model_max_length=2048)

answer_agnositic_model.save_pretrained('./pretrained_model/answer_agnostic_model')
answer_agnositic_tokenizer.save_pretrained('./pretrained_model/answer_agnostic_tokenizer')



tokenizer1 = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny", do_lower_case=True)
encoder1 = AutoModel.from_pretrained("prajjwal1/bert-tiny")

tokenizer1.save_pretrained('./pretrained_model/tokenizer1')
encoder1.save_pretrained('./pretrained_model/encoder1')



MixQG_tokenizer = AutoTokenizer.from_pretrained('Salesforce/mixqg-base')
MixQG_model = AutoModelForSeq2SeqLM.from_pretrained('Salesforce/mixqg-base')

MixQG_tokenizer.save_pretrained('./pretrained_model/MixQG_tokenizer')
MixQG_model.save_pretrained('./pretrained_model/MixQG_model')