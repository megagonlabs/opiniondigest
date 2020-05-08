# Copyright 2019 Megagon Labs, Inc. and the University of Edinburgh. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import argparse
import os
import sys
from typing import List

import dill
import numpy as np
import pandas as pd
import sentencepiece as spm
from sumeval.metrics.rouge import RougeCalculator
from sumeval.metrics.bleu import BLEUCalculator
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchtext.vocab import Vocab
from torchtext.data import Field, RawField, TabularDataset, BucketIterator

# from beam_search import Search, BeamSearch
import time

from models import LabelSmoothingLoss, TransformerModel, SumEvaluator, denumericalize
from utils import Config

if __name__ == "__main__":

    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--individual',
                        action='store_true')
    args = parser.parse_args()
    print("Option: --individual={}".format(args.individual))
    individual = args.individual
    """
    individual = False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if len(sys.argv) < 5:
        print("Config file(s) are missing")
        print("Usage: {} <prepare_conf> <train_conf> <aggregate_conf> <generate_conf>")
        sys.exit(1)

    p_conf = Config(sys.argv[1])
    t_conf = Config(sys.argv[2])
    a_conf = Config(sys.argv[3])  # can be CSV
    g_conf = Config(sys.argv[4])

    assert p_conf.conf_type == "prepare"
    assert t_conf.conf_type == "train"
    assert a_conf.conf_type == "aggregate"
    assert g_conf.conf_type == "generate"

    verbose = 0

    # Check if the method is valid
    assert g_conf["method"] in ["greedy", "beam"]

    # Basepath
    if "BASEPATH" not in os.environ:
        basepath = "."
    else:
        basepath = os.environ["BASEPATH"]

    # model filepath / output filepath
    model_filepath = os.path.join(basepath,
                                  "model",
                                  "{}_op2text_{}.pt".format(p_conf.conf_name,
                                                            t_conf.conf_name))
    output_filepath = os.path.join(basepath,
                                   "output",
                                   "{}_op2text_{}_{}_{}.csv".format(p_conf.conf_name,
                                                                    t_conf.conf_name,
                                                                    a_conf.conf_name,
                                                                    g_conf.conf_name))
    output_dirpath = os.path.dirname(output_filepath)
    if not os.path.exists(output_dirpath):
        os.makedirs(output_dirpath)

    # Load Fields
    with open(model_filepath.replace(".pt", "_IN_TEXT.field"), "rb") as fin:
        IN_TEXT = dill.load(fin)
    with open(model_filepath.replace(".pt", "_OUT_TEXT.field"), "rb") as fin:
        OUT_TEXT = dill.load(fin)
    with open(model_filepath.replace(".pt", "_ID.field"), "rb") as fin:
        ID = dill.load(fin)

    # Data file
    data_dirpath = os.path.join(basepath,
                                "data",
                                "{}".format(p_conf.conf_name))
    train_filepath = os.path.join(data_dirpath,
                                  "train.csv")
    valid_filepath = os.path.join(data_dirpath,
                                  "dev.csv")
    test_filepath = os.path.join(data_dirpath,
                                 "test.csv")
    agg_test_filepath = os.path.join(data_dirpath,
                                     "test{}.csv".format(a_conf.get_agg_name()))
    
    assert os.path.exists(train_filepath)
    assert os.path.exists(valid_filepath)
    assert os.path.exists(test_filepath)
    assert os.path.exists(agg_test_filepath)

    agg_test_df = pd.read_csv(agg_test_filepath)

    # We can use a different batch size for generation
    batch_size = g_conf["batch_size"]

    # Load dataset
    # ================================================================
    fields = {"eid": ("eid", ID),
              "rid": ("rid", ID),
              "review": ("out_text", OUT_TEXT),
              "input_text": ("in_text", IN_TEXT)}

    TEST_EID = RawField()
    agg_fields = {"eid": ("eid", TEST_EID),
                  "input_text": ("in_text", IN_TEXT)}

    train = TabularDataset(path=train_filepath,
                           format="csv",
                           fields=fields)
    valid = TabularDataset(path=valid_filepath,
                           format="csv",
                           fields=fields)
    test = TabularDataset(path=test_filepath,
                          format="csv",
                          fields=fields)
    agg_test = TabularDataset(path=agg_test_filepath,
                              format="csv",
                              fields=agg_fields)

    train_iterator = BucketIterator(train,
                                    batch_size=batch_size,
                                    device=device,
                                    sort=False,
                                    sort_within_batch=False)
    valid_iterator = BucketIterator(valid,
                                    batch_size=batch_size,
                                    device=device,
                                    sort=False,
                                    sort_within_batch=False)
    test_iterator = BucketIterator(test,
                                   batch_size=batch_size,
                                   device=device,
                                   sort=False,
                                   sort_within_batch=False)
    agg_test_iterator = BucketIterator(agg_test,
                                       batch_size=batch_size,
                                       device=device,
                                       sort=False,
                                       sort_within_batch=False)
    # ================================================================

    # Load model
    model = TransformerModel(len(IN_TEXT.vocab),
                             t_conf["model"]["params"]["d_model"],  # emb_size
                             len(OUT_TEXT.vocab),
                             pretrained_vectors=None,
                             nhead=t_conf["model"]["params"]["nhead"],
                             num_encoder_layers=t_conf["model"]["params"]["num_encoder_layer"],
                             num_decoder_layers=t_conf["model"]["params"]["num_decoder_layer"],
                             dim_feedforward=t_conf["model"]["params"]["dim_feedforward"],
                             dropout=t_conf["model"]["params"]["dropout"]).to(device)
    model.load_state_dict(torch.load(model_filepath,
                                     map_location=device))

    # sumeval evaluator
    evaluator = SumEvaluator(metrics=t_conf["metrics"],
                             stopwords=False,
                             lang="en")

    # Old script used t_conf["training"]["gen_maxlen"]
    gen_maxlen = g_conf["gen_maxtoken"]

    ## 1. Generation for each entity in "aggregated" test_{}.csv
    all_pred_gens = []
    all_eids = []
    for batch_idx, batch in enumerate(agg_test_iterator):
        tgt_seqlen, b_size = batch.in_text.shape

        start_time = time.time()
        # TODO: Switch generation ==================================
        if g_conf["method"] == "greedy":
            tgt = model.generate(batch.in_text,
                                 gen_maxlen,
                                 bos_index=OUT_TEXT.vocab.stoi["<BOS>"],
                                 pad_index=OUT_TEXT.vocab.stoi["<pad>"])
        elif g_conf["method"] == "beam":
            if "beam_width" in g_conf["params"]:
                beam_width = g_conf["params"]["beam_width"]
            else:
                beam_width = 3

            if "no_repeat_ngram_size" in g_conf["params"]:
                no_repeat_ngram_size = g_conf["params"]["no_repeat_ngram_size"]
            else:
                no_repeat_ngram_size = 0

            tgt = model.generate_beamsearch(
                batch.in_text,
                gen_maxlen,
                bos_index=OUT_TEXT.vocab.stoi["<BOS>"],
                pad_index=OUT_TEXT.vocab.stoi["<pad>"],
                unk_index=OUT_TEXT.vocab.stoi["<unk>"],
                eos_index=OUT_TEXT.vocab.stoi["<EOS>"],
                vocab_size=len(OUT_TEXT.vocab),
                beam_size=beam_width,
                no_repeat_ngram_size=no_repeat_ngram_size)
        else:
            raise ValueError("Invalid decoding method: {}".format(g_conf["method"]))

        pred_gens = denumericalize(tgt,
                                   OUT_TEXT.vocab,
                                   join=" ")
        all_pred_gens += pred_gens
        all_eids += batch.eid

    agg_pred_df = pd.DataFrame({"eid": all_eids,
                                "pred": all_pred_gens})

    merge_df = pd.merge(agg_pred_df,
                        agg_test_df[["eid", "input_text"]],
                        left_on="eid", right_on="eid")
    merge_df = merge_df[["eid", "input_text", "pred"]]
    #agg_pred_df.to_csv(
    merge_df.to_csv(
        output_filepath.replace(".csv", "_agg.csv"),
        index=False)

    ## 2. Generation for each review in test.csv
    if individual:
        print("Process individual revidws")
        eval_df_list = []
        beam_eval_df_list = []
        start_time = time.time()
        for batch_idx, batch in enumerate(test_iterator):
            tgt_seqlen, b_size = batch.out_text.shape

            start_time = time.time()
            # TODO: Switch generation ==================================
            if g_conf["method"] == "greedy":
                tgt = model.generate(batch.in_text,
                                     gen_maxlen,
                                     bos_index=OUT_TEXT.vocab.stoi["<BOS>"],
                                     pad_index=OUT_TEXT.vocab.stoi["<pad>"])
            elif g_conf["method"] == "beam":
                if "beam_width" in g_conf["params"]:
                    beam_width = g_conf["params"]["beam_width"]
                else:
                    beam_width = 3

                if "no_repeat_ngram_size" in g_conf["params"]:
                    no_repeat_ngram_size = g_conf["params"]["no_repeat_ngram_size"]
                else:
                    no_repeat_ngram_size = 0

                tgt = model.generate_beamsearch(
                    batch.in_text,
                    gen_maxlen,
                    bos_index=OUT_TEXT.vocab.stoi["<BOS>"],
                    pad_index=OUT_TEXT.vocab.stoi["<pad>"],
                    unk_index=OUT_TEXT.vocab.stoi["<unk>"],
                    eos_index=OUT_TEXT.vocab.stoi["<EOS>"],
                    vocab_size=len(OUT_TEXT.vocab),
                    beam_size=beam_width,
                    no_repeat_ngram_size=no_repeat_ngram_size)
            else:
                raise ValueError("Invalid decoding method: {}".format(g_conf["method"]))

            true_gens = denumericalize(batch.out_text,
                                       OUT_TEXT.vocab,
                                       join=" ")
            pred_gens = denumericalize(tgt,
                                       OUT_TEXT.vocab,
                                       join=" ")

            # Generation evaluation 
            eval_df = evaluator.eval(true_gens, pred_gens)
            eval_df_list.append(eval_df)

            if verbose == 1 and batch_idx % 100 == 0:
                end_time = time.time()
                elapsed_time = end_time - start_time
                start_time = end_time
                print("{} done ({:.2f} sec.)".format((batch_idx + 1) * batch_size,
                                                     elapsed_time))

        # Save results
        all_eval_df = pd.concat(eval_df_list, axis=0).reset_index(drop=True)
        all_eval_df.to_csv(output_filepath,
                           index=False)
        all_eval_df.mean(axis=0).to_csv(
            output_filepath.replace(".csv", ".eval"))
