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

import math
from typing import List

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from beam_search import BeamSearch, NgramBlocking
from sumeval.metrics.bleu import BLEUCalculator
from sumeval.metrics.rouge import RougeCalculator
from torchtext.vocab import Vocab


# Modification of the following OpenNMT-code
# https://github.com/OpenNMT/OpenNMT-py/blob/e8622eb5c6117269bb3accd8eb6f66282b5e67d9/onmt/utils/loss.py#L186
class LabelSmoothingLoss(nn.Module):
    """
    With label smoothing,
    KL-divergence between q_{smoothed ground truth prob.}(w)
    and p_{prob. computed by model}(w) is minimized.
    """
    def __init__(self,
                 label_smoothing: float,
                 tgt_vocab_size: int,
                 device: torch.device,
                 ignore_index: int = -100,
                 reduction: str = "none"):
        assert 0.0 < label_smoothing <= 1.0
        assert reduction in ["sum", "mean", "batchmean", "none"]
        self.device = device
        self.ignore_index = ignore_index
        self.reduction = reduction
        super(LabelSmoothingLoss, self).__init__()

        smoothing_value = label_smoothing / (tgt_vocab_size - 2)
        one_hot = torch.full((tgt_vocab_size,), smoothing_value)
        one_hot[self.ignore_index] = 0
        
        self.register_buffer('one_hot', one_hot.unsqueeze(0))
        self.confidence = 1.0 - label_smoothing

    def forward(self,
                output: torch.FloatTensor,
                target: torch.LongTensor):
        """
        output (FloatTensor): batch_size x n_classes
        target (LongTensor): batch_size
        """
        # model_prob = self.one_hot.repeat(target.size(0), 1)
        # model_prob.scatter_(1, target.unsqueeze(1), self.confidence)
        # model_prob.masked_fill_((target == self.ignore_index).unsqueeze(1), 0)
        # return F.kl_div(output, model_prob, reduction=self.reduction)
        model_prob = self.one_hot.repeat(target.size(0), target.size(1), 1).to(device)
        model_prob.scatter_(2, target.unsqueeze(2), self.confidence)
        model_prob.masked_fill((labels == self.ignore_index).unsqueeze(2), 0)

        if self.reduction == "none":
            return F.kl_div(F.log_softmax(output),
                            model_prob,
                            reduction="none").sum(2).sum(0).mean()
        else:
            return F.kl_div(F.log_softmax(output),
                            model_prob,
                            reduction=self.reduction)

    
class PositionalEncoding(nn.Module):
    """OpenNMT-py"""
    """Sinusoidal positional encoding for non-recurrent neural networks.

    Implementation based on "Attention Is All You Need"
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`

    Args:
       dropout (float): dropout parameter
       dim (int): embedding size
    """

    def __init__(self, dim, dropout=0.1, max_len=5000):
        if dim % 2 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dim (got dim={:d})".format(dim))
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) *
                             -(math.log(10000.0) / dim)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(1)
        super(PositionalEncoding, self).__init__()
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout)
        self.dim = dim

    def forward(self, emb, step=None):
        """Embed inputs.

        Args:
            emb (FloatTensor): Sequence of word vectors
                ``(seq_len, batch_size, self.dim)``
            step (int or NoneType): If stepwise (``seq_len = 1``), use
                the encoding for this position.
        """

        emb = emb * math.sqrt(self.dim)
        if step is None:
            emb = emb + self.pe[:emb.size(0)]
        else:
            emb = emb + self.pe[step]
        emb = self.dropout(emb)
        return emb

    
class TransformerModel(nn.Module):
    def __init__(self,
                 in_vocab_size: int,
                 emb_size: int,
                 out_vocab_size: int,
                 pretrained_vectors: torch.Tensor = None,
                 shared_word_embedding: bool = False,
                 *args, **kwargs):
        super(TransformerModel, self).__init__()
        self.in_vocab_size = in_vocab_size
        self.out_vocab_size = out_vocab_size
        self.emb_size = emb_size
        self.in_emb = nn.Embedding(self.in_vocab_size,
                                   self.emb_size)
        self.in_pos = PositionalEncoding(self.emb_size)

        # Shared word&pos embedding
        if shared_word_embedding:
            if in_vocab_size != out_vocab_size:
                raise ValueError("Unable to use shared word/pos embeddings ",
                                 "if in_vocab_size != out_vocab_size.")
            self.out_emb = self.in_emb
            self.out_pos = self.in_pos
        else:
            self.out_emb = nn.Embedding(self.out_vocab_size,
                                        self.emb_size)
            self.out_pos = PositionalEncoding(self.emb_size)
            
        if pretrained_vectors is not None:
            self.in_emb.weight.data.copy_(pretrained_vectors)
            self.out_emb.weight.data.copy_(pretrained_vectors)

        self.transformer = nn.Transformer(self.emb_size,
                                          **kwargs)
        self.linear = nn.Linear(self.emb_size,
                                self.out_vocab_size)

    def in_embed(self,
                 src):
        return self.in_pos(self.in_emb(src))

    def out_embed(self,
                  tgt):
        return self.out_pos(self.out_emb(tgt))
    
    def encode(self,
               src,
               src_mask=None,
               src_key_padding_mask=None):
        memory = self.transformer.encoder(self.in_embed(src),
                                          mask=src_mask,
                                          src_key_padding_mask=src_key_padding_mask)
        return memory
    
    def decode(self,
               tgt,
               memory,
               tgt_mask=None,
               memory_mask=None,
               tgt_key_padding_mask=None,
               memory_key_padding_mask=None):
        output = self.transformer.decoder(self.out_embed(tgt),
                                          memory,
                                          tgt_mask=tgt_mask,
                                          memory_mask=memory_mask,
                                          tgt_key_padding_mask=tgt_key_padding_mask,
                                          memory_key_padding_mask=memory_key_padding_mask)
        return output
    
    def forward(self,
                src,
                tgt,
                *args, **kwargs):
        out = self.transformer(self.in_embed(src),
                               self.out_embed(tgt),
                               *args,
                               **kwargs)
        out = self.linear(out)
        return out

    def generate(self,
                 src: torch.Tensor,
                 maxlen: int,
                 bos_index: int,
                 pad_index: int):
        # Obtain device information
        device = next(self.parameters()).device
        _, batch_size = src.shape
        src_key_padding_mask = (src == pad_index).T  # batch_size x srclen
        memory = self.encode(src,
                             src_key_padding_mask=src_key_padding_mask)
        
        # <BOS> tgt seq for generation
        tgt = torch.LongTensor(maxlen, batch_size).fill_(pad_index).to(device)
        tgt[0, :] = torch.LongTensor(batch_size).fill_(bos_index).to(device)

        for i in range(1, maxlen):
            tgt_key_padding_mask = (tgt[:i, :] == pad_index).T  # batch_size x len(tgt)
            tgt_mask = self.transformer.generate_square_subsequent_mask(i).to(device)
            decode_prob = self.decode(tgt[:i, :],
                                      memory,
                                      tgt_mask=tgt_mask,
                                      tgt_key_padding_mask=tgt_key_padding_mask)
            pred_prob = self.linear(decode_prob)
            decode_output = pred_prob.argmax(2)
            tgt[i, :] = decode_output[-1, :]
        return tgt

    def generate_beamsearch(self,
                            src: torch.Tensor,
                            maxlen: int,
                            bos_index: int,
                            pad_index: int,
                            unk_index: int,
                            eos_index: int,
                            vocab_size: int,
                            beam_size: int=3,
                            no_repeat_ngram_size: int=0):
        # Obtain device information
        device = next(self.parameters()).device
        _, batch_size = src.shape        
        src_key_padding_mask = (src == pad_index).T  # batch_size x srclen
        memory = self.encode(src,
                             src_key_padding_mask=src_key_padding_mask)
        
        # <BOS> tgt seq for generation
        tgt = torch.LongTensor(maxlen, batch_size, beam_size).fill_(pad_index).to(device)
        tgt[0, :, :] = torch.LongTensor(batch_size, beam_size).fill_(bos_index).to(device)
        scores = torch.zeros(batch_size, beam_size, maxlen).to(device)
        scores[:, :, 0] = torch.ones(batch_size, beam_size).to(device)
        active_beams = [0]  # up to beam_size beams.
        search = BeamSearch(vocab_size, pad_index, unk_index, eos_index)
        ngram_blocking = NgramBlocking(no_repeat_ngram_size)

        # After eos
        log_probs_after_eos = torch.FloatTensor(batch_size, beam_size, self.out_vocab_size).fill_(float("-inf")).cpu()
        log_probs_after_eos[:, :, eos_index] = 0.
        best_n_indices = tgt.new_full((batch_size, len(active_beams)), bos_index)

        for i in range(1, maxlen):
            if (best_n_indices == eos_index).all():  # if all of last prediction is eos, we can leave the loop
                break

            # Generate probability for all beams, update probability for all beams (lprobs).
            lprobs = torch.zeros(batch_size, len(active_beams), vocab_size).to(device)
            for j in range(len(active_beams)):
                tgt_key_padding_mask = (tgt[:i, :, active_beams[j]] == pad_index).T  # batch_size x len(tgt)
                tgt_mask = self.transformer.generate_square_subsequent_mask(i).to(device)
                decode_prob = self.decode(tgt[:i, :, active_beams[j]],
                                          memory,
                                          tgt_mask=tgt_mask,
                                          tgt_key_padding_mask=tgt_key_padding_mask)
                pred_prob = self.linear(decode_prob)
                lprobs[:, j, :] = pred_prob[-1, :]

            # Update lprobs for n-gram blocking
            if no_repeat_ngram_size > 0:
                for batch_idx in range(batch_size):
                    for beam_idx in range(len(active_beams)):
                        lprobs[batch_idx, beam_idx] = ngram_blocking.update(i-1, 
                            tgt[:i, batch_idx, beam_idx], lprobs[batch_idx, beam_idx])

            expanded_indices = best_n_indices.detach().cpu().unsqueeze(-1).expand(
                (batch_size, len(active_beams), self.out_vocab_size))
            clean_lprobs = torch.where(expanded_indices == eos_index, log_probs_after_eos[:, :len(active_beams)],
                                       F.log_softmax(lprobs.detach().cpu(), dim=-1))
            # Run the beam search step and select the top-k beams.
            best_n_scores, best_n_indices, best_n_beams = search.step(i, clean_lprobs,
                scores.index_select(1, torch.tensor(active_beams, device=device)).detach().cpu(),
                beam_size)
            
            # Take the top results, more optimization can be done here, e.g., avoid <eos> beams.
            best_n_scores = best_n_scores[:, :beam_size]
            best_n_indices = best_n_indices[:, :beam_size]
            best_n_beams = best_n_beams[:, :beam_size]

            # update results
            tgt = tgt.gather(2, best_n_beams.unsqueeze(0).expand(maxlen, batch_size, -1).to(device))
            tgt[i, :, :] = best_n_indices
            scores[:, :, i] = best_n_scores
            active_beams = range(beam_size)

        return tgt[:, :, 0]

    
def denumericalize(id_tensor: torch.Tensor,
                   vocab: Vocab,
                   ignore_pad: str = "<pad>",
                   ignore_bos: str = "<BOS>",
                   ignore_eos: str = "<EOS>",
                   join: str = None):
    sentences = id_tensor.t().tolist()
    denum_sentences = []
    for sentence in sentences:
        denum_sentence = []
        for token_id in sentence:
            if ignore_bos and token_id == vocab.stoi[ignore_bos]:
                continue
            elif ignore_eos and token_id == vocab.stoi[ignore_eos]:
                break
            elif ignore_pad and token_id == vocab.stoi[ignore_pad]:
                break
            denum_sentence.append(vocab.itos[token_id])
        if join is None:
            denum_sentences.append(denum_sentence)
        else:
            denum_sentences.append(join.join(denum_sentence))

    return denum_sentences



class SumEvaluator:
    """Evaluator class for generation.
    A wrapper class of sumeval library
    """
    def __init__(self,
                 metrics: List[str] = ["rouge_1",
                                       "rouge_2",
                                       "rouge_l",
                                       "rouge_be",
                                       "bleu"],
                 lang: str = "en",
                 stopwords: bool = True,
                 stemming: bool = True,
                 use_porter = True):
        if use_porter:
            self.rouge = RougeCalculator(stopwords=stopwords,
                                         stemming=stemming,
                                         lang="en-porter")
        else:
            self.rouge = RougeCalculator(stopwords=stopwords,
                                         stemming=stemming,
                                         lang="en")
        self.bleu = BLEUCalculator(lang=lang)
        self.metrics = sorted(metrics)
        
    def eval(self,
             true_gens: List[str],
             pred_gens: List[str]):
        
        assert len(true_gens) == len(pred_gens)
        
        eval_list = []
        colnames = []
        for i, (true_gen, pred_gen) in enumerate(zip(true_gens, pred_gens)):
            evals = []
            
            # BLEU
            if "bleu" in self.metrics:
                bleu_score = self.bleu.bleu(pred_gen,
                                            true_gen) / 100.0  # align scale
                evals.append(bleu_score)
            
            # ROUGE
            if "rouge_1" in self.metrics:
                rouge_1 = self.rouge.rouge_n(
                    summary=pred_gen,
                    references=[true_gen],
                    n=1)
                evals.append(rouge_1)                

            if "rouge_2" in self.metrics:
                rouge_2 = self.rouge.rouge_n(
                    summary=pred_gen,
                    references=[true_gen],
                    n=2)
                evals.append(rouge_2)

            if "rouge_be" in self.metrics:
                rouge_be = self.rouge.rouge_be(
                    summary=pred_gen,
                    references=[true_gen])
                evals.append(rouge_be)                
                    
            if "rouge_l" in self.metrics:
                rouge_l = self.rouge.rouge_l(
                    summary=pred_gen,
                    references=[true_gen])
                evals.append(rouge_l)                

            eval_list.append([pred_gen, true_gen] + evals)
        eval_df = pd.DataFrame(eval_list,
                               columns=["pred",
                                        "true"] + self.metrics)
        return eval_df
