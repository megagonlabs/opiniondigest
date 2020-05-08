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

import os
import sys

import dill
import pandas as pd
import sentencepiece as spm
import torch
import torch.nn as nn
from models import LabelSmoothingLoss, TransformerModel, SumEvaluator, denumericalize
from torch.optim.lr_scheduler import StepLR
from torch.optim import SGD
from torchtext.data import Field, TabularDataset, BucketIterator
from utils import Config

# from beam_search import Search, BeamSearch

if __name__ == "__main__":

    if len(sys.argv) < 3:
        print("Config file is missing")
        print("Usage: {} <prepare_conf> <train_conf>")
        sys.exit(1)

    p_conf = Config(sys.argv[1])
    t_conf = Config(sys.argv[2])

    assert p_conf.conf_type == "prepare"
    assert t_conf.conf_type == "train"

    verbose = 1    

    data_split_ratio = [0.8, 0.1, 0.1]
    pretrained_vectors = None  # "glove.6B.100d"
    
    loss_func = t_conf["training"]["loss_func"] # cross_entropy_avgsum
    sp_model_filepath = None  # "model/hm_model.model"

    batch_size = t_conf["training"]["batch_size"]
    num_epoch = t_conf["training"]["num_epoch"]
    if "clipping" in t_conf["training"]:
        clipping = t_conf["training"]["clipping"]
    else:
        clipping = None

    gen_maxlen = t_conf["training"]["gen_maxlen"]
    metrics = t_conf["metrics"]
    
    if "BASEPATH" not in os.environ:
        basepath = "."
    else:
        basepath = os.environ["BASEPATH"]
    model_filepath = os.path.join(basepath,
                                  "model",
                                  "{}_op2text_{}.pt".format(p_conf.conf_name,
                                                            t_conf.conf_name))

    # Dirpath 
    model_dirname = os.path.dirname(model_filepath)
    if not os.path.exists(model_dirname):
        os.makedirs(model_dirname)

    # Data file
    data_dirpath = os.path.join(basepath,
                                "data",
                                "{}".format(p_conf.conf_name))
    train_filepath = os.path.join(data_dirpath,
                                  "train.csv")
    valid_filepath = os.path.join(data_dirpath,
                                  "dev.csv")

    assert os.path.exists(train_filepath)
    assert os.path.exists(valid_filepath)
    
    # config
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if sp_model_filepath is None:
        sp = None
    else:
        sp = spm.SentencePieceProcessor()
        sp.Load(sp_model_filepath)

    ID = Field(sequential=False,
               is_target=False)
    
    if sp is None:
        IN_TEXT = Field(sequential=True,
                        lower=True,
                        use_vocab=True,
                        init_token="<BOS>",
                        eos_token="<EOS>")
        OUT_TEXT = Field(sequential=True,
                         lower=True,
                         use_vocab=True,
                         init_token="<BOS>",
                         eos_token="<EOS>")
    else:
        IN_TEXT = Field(sequential=True,
                        lower=True,
                        use_vocab=True,
                        init_token="<BOS>",
                        eos_token="<EOS>",
                        tokenize=sp.EncodeAsPieces)
        OUT_TEXT = Field(sequential=True,
                         lower=True,
                         use_vocab=True,
                         init_token="<BOS>",
                         eos_token="<EOS>",
                         tokenize=sp.EncodeAsPieces)

    # Load dataset
    fields = {"eid": ("eid", ID),
              "rid": ("rid", ID),
              "review": ("out_text", OUT_TEXT),
              "input_text": ("in_text", IN_TEXT)}
    
    train = TabularDataset(path=train_filepath,
                           format="csv",
                           fields=fields)
    valid = TabularDataset(path=valid_filepath,
                           format="csv",
                           fields=fields)
    
    # Build vocabulary
    if "vocab" in t_conf and "max_size" in t_conf["vocab"]:
        vocab_max_size = t_conf["vocab"]["max_size"]
    else:
        vocab_max_size = None
        
    ID.build_vocab(train)
    if pretrained_vectors:
        IN_TEXT.build_vocab(train, vectors=pretrained_vectors, max_size=vocab_max_size)
        OUT_TEXT.build_vocab(train, vectors=pretrained_vectors, max_size=vocab_max_size)        
    else:
        IN_TEXT.build_vocab(train, max_size=vocab_max_size)
        OUT_TEXT.build_vocab(train, max_size=vocab_max_size)
        
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

    # sumeval evaluator
    evaluator = SumEvaluator(metrics=metrics,
                             stopwords=False,
                             lang="en")

    # Transformer model
    model = TransformerModel(len(IN_TEXT.vocab),
                             t_conf["model"]["params"]["d_model"], # emb_size
                             len(OUT_TEXT.vocab),
                             pretrained_vectors=None,
                             nhead=t_conf["model"]["params"]["nhead"],
                             num_encoder_layers=t_conf["model"]["params"]["num_encoder_layer"],
                             num_decoder_layers=t_conf["model"]["params"]["num_decoder_layer"],
                             dim_feedforward=t_conf["model"]["params"]["dim_feedforward"],
                             dropout=t_conf["model"]["params"]["dropout"]).to(device)

    # Optimizer
    # General template to make an optimzier instance
    # e.g.,)
    #        optimizer = optim.SGD(model.parameters(),
    #                              lr=0.1,
    #                              momentum=0.9,
    #                              nesterov=True)
    optimizer = eval("{}(model.parameters(), **{})".format(t_conf["training"]["optimizer"]["cls"],
                                                           str(t_conf["training"]["optimizer"]["params"])))

    scheduler = StepLR(optimizer,
                       step_size=1,
                       gamma=0.1)
    
    # Loss function
    if loss_func == "cross_entropy":
        criterion = nn.CrossEntropyLoss(ignore_index=OUT_TEXT.vocab.stoi["<pad>"],
                                        reduction="none").to(device) # mean
        
    elif loss_func == "label_smoothing":
        criterion = LabelSmoothingLoss(label_smoothing=0.1,
                                       tgt_vocab_size=len(OUT_TEXT.vocab),
                                       device=device,
                                       ignore_index=OUT_TEXT.vocab.stoi["<pad>"])

    log_data_list = []
    # Training
    for epoch in range(num_epoch):
        training_loss = 0.0
        valid_loss = 0.0
        model.train()
        scheduler.step()
        print("Epoch:", epoch, "LR:", scheduler.get_lr())
        for batch_idx, batch in enumerate(train_iterator):
            if verbose == 1 and batch_idx % 100 == 0:
                print(batch_idx)
            src_seqlen, b_size = batch.in_text.shape
            tgt_seqlen, _ = batch.out_text.shape
            optimizer.zero_grad()

            # Mask preparation
            src_key_padding_mask = (batch.in_text == IN_TEXT.vocab.stoi["<pad>"]).T  # b_size x srclen
            tgt_key_padding_mask = (batch.out_text == OUT_TEXT.vocab.stoi["<pad>"]).T  # b_size x tgtlen
            tgt_mask = model.transformer.generate_square_subsequent_mask(tgt_seqlen).to(device)

            pred = model(batch.in_text,
                         batch.out_text,
                         tgt_mask=tgt_mask,
                         src_key_padding_mask=src_key_padding_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask)

            # Shift one token for prediction
            labels = torch.cat([batch.out_text[1:, :],
                                torch.LongTensor(OUT_TEXT.vocab.stoi["<pad>"],
                                                 b_size).fill_(1).to(device)], axis=0)
            if loss_func == "cross_entropy":
                loss_vals = criterion(pred.transpose(1, 2),
                                      labels)
                
                # Avg of sum of token loss (after ignoring padding tokens)
                # loss = loss_vals
                loss = loss_vals.sum(axis=0).mean()

            elif loss_func == "label_smoothing":
                loss = criterion(pred, labels)

            loss.backward()
            # Clipping
            if clipping:
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               clipping)
            optimizer.step()        
            
            loss_val = loss.data.item() # * batch.in_text.size(0)
            if verbose >= 2:
                if batch_idx % 500 == 0:
                    print("Train: {} loss={}".format(batch_idx, loss_val))
                    print("Input: {}".format(denumericalize(batch.in_text,
                                                            OUT_TEXT.vocab)[0]))
                    print("True: {}".format(denumericalize(batch.out_text,
                                                           OUT_TEXT.vocab)[0]))
                    print("Pred: {}".format(denumericalize(pred.argmax(dim=2),
                                                           OUT_TEXT.vocab)[0]))

            training_loss += loss_val
            
        training_loss /= len(train_iterator)

        # Validation
        model.eval()
        eval_df_list = []
        beam_eval_df_list = []
        greedy_time = 0.
        beamsearch_time = 0.
        for batch_idx, batch in enumerate(valid_iterator):
            if verbose == 1 and batch_idx % 100 == 0:
                print(batch_idx)

            tgt_seqlen, b_size = batch.out_text.shape

            # Mask preparation
            src_key_padding_mask = (batch.in_text == IN_TEXT.vocab.stoi["<pad>"]).T  # b_size x srclen
            tgt_key_padding_mask = (batch.out_text == OUT_TEXT.vocab.stoi["<pad>"]).T  # b_size x tgtlen
            tgt_mask = model.transformer.generate_square_subsequent_mask(tgt_seqlen).to(device)

            pred = model(batch.in_text,
                         batch.out_text,
                         tgt_mask=tgt_mask,
                         src_key_padding_mask=src_key_padding_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask)
            
            labels = torch.cat([batch.out_text[1:, :],
                                torch.LongTensor(OUT_TEXT.vocab.stoi["<pad>"],
                                                 b_size).fill_(1).to(device)], axis=0)
            if loss_func == "cross_entropy":
                loss_vals = criterion(pred.transpose(1, 2),
                                      labels)
                
                # Avg of sum of token loss (after ignoring padding tokens)
                loss = loss_vals.sum(axis=0).mean()

            elif loss_func == "label_smoothing":
                loss = criterion(pred,
                                 labels)

            valid_loss_val = loss.data.item() * batch.in_text.size(0)
            valid_loss += valid_loss_val

            # TODO(Yoshi): How to save log?
            if verbose >= 2 and batch_idx % 500 == 0:
                tgt = model.generate(batch.in_text,
                                     gen_maxlen,
                                     bos_index=OUT_TEXT.vocab.stoi["<BOS>"],
                                     pad_index=OUT_TEXT.vocab.stoi["<pad>"])

                print("Valid: {} loss={}".format(batch_idx, loss_val))                
                print("Input: {}".format(denumericalize(batch.in_text,
                                                        OUT_TEXT.vocab)[0]))
                print("True: {}".format(denumericalize(batch.out_text,
                                                       OUT_TEXT.vocab)[0]))
                print("Pred: {}".format(denumericalize(pred.argmax(dim=2),
                                                       OUT_TEXT.vocab)[0]))
                print(" Gen: {}".format(denumericalize(tgt,
                                                       OUT_TEXT.vocab)[0]))
            
        valid_loss /= len(valid_iterator)
        print('Epoch: {}, Training loss: {:.2f}, Valid loss: {:.2f}'.format(
            epoch, training_loss, valid_loss))

        # log
        log_data_list.append([epoch,
                              training_loss,
                              valid_loss])

        # TODO(Yoshi): The last model is redundant
        torch.save(model.state_dict(),
                   model_filepath.replace(".pt", "_epoch-{}.pt".format(epoch)))

    torch.save(model.state_dict(),
               model_filepath)
    
    with open(model_filepath.replace(".pt", "_IN_TEXT.field"), "wb") as fout:
        dill.dump(IN_TEXT, fout)
    with open(model_filepath.replace(".pt", "_OUT_TEXT.field"), "wb") as fout:
        dill.dump(OUT_TEXT, fout)
    with open(model_filepath.replace(".pt", "_ID.field"), "wb") as fout:
        dill.dump(ID, fout)

    # Write out log
    df = pd.DataFrame(log_data_list,
                      columns=["epoch", "training_loss", "valid_loss"])
    df.to_csv(model_filepath.replace(".pt", "_loss.csv"))
    
