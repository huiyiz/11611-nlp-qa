# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import sys
import torch
import transformers
import numpy as np
from pathlib import Path
import torch.distributed as dist
from torch.utils.data import DataLoader, SequentialSampler

import src.slurm
import src.util
from src.options import Options
import src.data
import src.evaluation
import src.model

import argparse
import csv
import json
import logging
import pickle
import time
import glob
from pathlib import Path

import numpy as np
import torch
import transformers

import src.slurm
import src.util
import src.model
import src.data
import src.index

from torch.utils.data import DataLoader

from src.evaluation import calculate_matches


# used imports
from argparse import Namespace
import csv
import json
import logging
import pickle
import time
import glob
from pathlib import Path

import numpy as np
import torch
import transformers

import src.slurm
import src.util
import src.model
import src.data
import src.index

from torch.utils.data import DataLoader

from src.evaluation import calculate_matches



################################################################################
# Retrieval functions                                                          #
################################################################################
def embed_questions(opt, data, model, tokenizer):
    batch_size = opt.per_gpu_batch_size * opt.world_size
    dataset = src.data.Dataset(data)
    collator = src.data.Collator(opt.question_maxlength, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, drop_last=False, num_workers=10, collate_fn=collator)
    model.eval()
    embedding = []
    with torch.no_grad():
        for k, batch in enumerate(dataloader):
            (idx, _, _, question_ids, question_mask) = batch
            output = model.embed_text(
                text_ids=question_ids.to(opt.device).view(-1, question_ids.size(-1)), 
                text_mask=question_mask.to(opt.device).view(-1, question_ids.size(-1)), 
                apply_mask=model.config.apply_question_mask,
                extract_cls=model.config.extract_cls,
            )
            embedding.append(output)

    embedding = torch.cat(embedding, dim=0)
    logger.info(f'Questions embeddings shape: {embedding.size()}')

    return embedding.cpu().numpy()


def index_encoded_data(index, embedding_files, indexing_batch_size):
    allids = []
    allembeddings = np.array([])
    for i, file_path in enumerate(embedding_files):
        logger.info(f'Loading file {file_path}')
        with open(file_path, 'rb') as fin:
            ids, embeddings = pickle.load(fin)

        allembeddings = np.vstack((allembeddings, embeddings)) if allembeddings.size else embeddings
        allids.extend(ids)
        while allembeddings.shape[0] > indexing_batch_size:
            allembeddings, allids = add_embeddings(index, allembeddings, allids, indexing_batch_size)

    while allembeddings.shape[0] > 0:
        allembeddings, allids = add_embeddings(index, allembeddings, allids, indexing_batch_size)
        
    logger.info('Data indexing completed.')

def add_embeddings(index, embeddings, ids, indexing_batch_size):
    end_idx = min(indexing_batch_size, embeddings.shape[0])
    ids_toadd = ids[:end_idx]
    embeddings_toadd = embeddings[:end_idx]
    ids = ids[end_idx:]
    embeddings = embeddings[end_idx:]
    index.index_data(ids_toadd, embeddings_toadd)
    return embeddings, ids


def validate(data, workers_num):
    match_stats = calculate_matches(data, workers_num)
    top_k_hits = match_stats.top_k_hits

    logger.info('Validation results: top k documents hits %s', top_k_hits)
    top_k_hits = [v / len(data) for v in top_k_hits] 
    logger.info('Validation results: top k documents hits accuracy %s', top_k_hits)
    return match_stats.questions_doc_hits


def add_passages(data, passages, top_passages_and_scores):
    # add passages to original data
    merged_data = []
    assert len(data) == len(top_passages_and_scores)
    for i, d in enumerate(data):
        results_and_scores = top_passages_and_scores[i]
        docs = [passages[doc_id] for doc_id in results_and_scores[0]]
        scores = [str(score) for score in results_and_scores[1]]
        ctxs_num = len(docs)
        d['ctxs'] =[
                {
                    'id': results_and_scores[0][c],
                    'title': docs[c][1],
                    'text': docs[c][0],
                    'score': scores[c],
                } for c in range(ctxs_num)
            ] 

def add_hasanswer(data, hasanswer):
    # add hasanswer to data
    for i, ex in enumerate(data):
        for k, d in enumerate(ex['ctxs']):
            d['hasanswer'] = hasanswer[i][k]

def retriever(opt): 
    src.util.init_logger(is_main=True)
    tokenizer = transformers.BertTokenizerFast.from_pretrained('bert-base-uncased')
    data = src.data.load_data(opt.data)
    model_class = src.model.Retriever
    model = model_class.from_pretrained(opt.model_path)

    model.cuda()
    model.eval()
    if not opt.no_fp16:
        model = model.half()

    index = src.index.Indexer(model.config.indexing_dimension, opt.n_subquantizers, opt.n_bits)

    # index all passages
    input_paths = glob.glob(args.passages_embeddings)
    input_paths = sorted(input_paths)
    embeddings_dir = Path(input_paths[0]).parent
    index_path = embeddings_dir / 'index.faiss'
    if args.save_or_load_index and index_path.exists():
        src.index.deserialize_from(embeddings_dir)
    else:
        logger.info(f'Indexing passages from files {input_paths}')
        start_time_indexing = time.time()
        index_encoded_data(index, input_paths, opt.indexing_batch_size)
        logger.info(f'Indexing time: {time.time()-start_time_indexing:.1f} s.')
        if args.save_or_load_index:
            src.index.serialize(embeddings_dir)

    questions_embedding = embed_questions(opt, data, model, tokenizer)

    # get top k results
    start_time_retrieval = time.time()
    top_ids_and_scores = index.search_knn(questions_embedding, args.n_docs) 
    logger.info(f'Search time: {time.time()-start_time_retrieval:.1f} s.')

    passages = src.util.load_passages(args.passages)
    passages = {x[0]:(x[1], x[2]) for x in passages}

    add_passages(data, passages, top_ids_and_scores)
    hasanswer = validate(data, args.validation_workers)
    add_hasanswer(data, hasanswer)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_path, 'w') as fout:
        json.dump(data, fout, indent=4)
    logger.info(f'Saved results to {args.output_path}')


if __name__ == "__main__":
    opt = {}
    

    # passage retrieval parameters
    opt['data'] = "./experiment_dataset/NQ/test.json" # .json file containing question and answers, similar format to reader data
    opt['passages'] = "./experiment_dataset/processed_wikipedia_passages_both.tsv" # Path to passages (.tsv file)
    opt['passages_embeddings'] = "./experiment_dataset" # Glob path to encoded passages
    opt['output_path'] = "./retriever_output/output.json"
    opt['n-docs'] = 100 # Number of documents to retrieve per questions
    opt['validation_workers'] = 16 # Number of parallel processes to validate results
    opt['per_gpu_batch_size'] = 64 # Batch size for question encoding
    opt['save_or_load_index'] = True # If enabled, save index and load index if it exists
    opt['model_path'] = "pretrained_models/nq_retriever"
    opt['no_fp16'] = False
    opt['passage_maxlength'] = 200 # Maximum number of tokens in a passage
    opt['question_maxlength'] = 40 # Maximum number of tokens in a question
    opt['indexing_batch_size'] = 50000 # Batch size of the number of passages indexed
    opt['n-subquantizers'] = 0
    opt['n-bits'] = 8
    
    # passage retrieval
    args = Namespace(**opt)
    src.slurm.init_distributed_mode(args)
    retriever(args)







    # get raw questions
    article_path = sys.argv[1]
    question_path = sys.argv[2]

    with open(question_path, "r", encoding="utf-8") as f: 
        raw_questions = [line.strip().lower() for line in f]

    
    # run retrieval model







    # run reader model



    # write output to file
    glob_path = Path(opt.checkpoint_dir) / opt.name / 'test_results'
    write_path = Path(opt.checkpoint_dir) / opt.name / 'final_output.txt'
    src.util.write_output(glob_path, write_path) 








    options = Options()
    options.add_reader_options()
    options.add_eval_options()
    opt = options.parse()
    src.slurm.init_distributed_mode(opt)
    src.slurm.init_signal_handler()
    opt.train_batch_size = opt.per_gpu_batch_size * max(1, opt.world_size)

    dir_path = Path(opt.checkpoint_dir)/opt.name
    directory_exists = dir_path.exists()
    if opt.is_distributed:
        torch.distributed.barrier()
    dir_path.mkdir(parents=True, exist_ok=True)
    if opt.write_results:
        (dir_path / 'test_results').mkdir(parents=True, exist_ok=True)
    logger = src.util.init_logger(opt.is_main, opt.is_distributed, Path(opt.checkpoint_dir) / opt.name / 'run.log')
    if not directory_exists and opt.is_main:
        options.print_options(opt)


    tokenizer = transformers.T5Tokenizer.from_pretrained('t5-base', return_dict=False)

    collator_function = src.data.Collator(opt.text_maxlength, tokenizer)
    eval_examples = src.data.load_data(
        opt.eval_data, 
        global_rank=opt.global_rank, #use the global rank and world size attibutes to split the eval set on multiple gpus
        world_size=opt.world_size
    )
    eval_dataset = src.data.Dataset(
        eval_examples, 
        opt.n_context, 
    )

    eval_sampler = SequentialSampler(eval_dataset) 
    eval_dataloader = DataLoader(
        eval_dataset, 
        sampler=eval_sampler, 
        batch_size=opt.per_gpu_batch_size,
        num_workers=20, 
        collate_fn=collator_function
    )
    
    model_class = src.model.FiDT5
    model = model_class.from_pretrained(opt.model_path)
    model = model.to(opt.device)

    logger.info("Start eval")
    exactmatch, total = evaluate(model, eval_dataset, eval_dataloader, tokenizer, opt)

    logger.info(f'EM {100*exactmatch:.2f}, Total number of example {total}')

    if opt.write_results and opt.is_main:
        glob_path = Path(opt.checkpoint_dir) / opt.name / 'test_results'
        write_path = Path(opt.checkpoint_dir) / opt.name / 'final_output.txt'
        src.util.write_output(glob_path, write_path) 
    if opt.write_crossattention_scores:
        src.util.save_distributed_dataset(eval_dataset.data, opt)

