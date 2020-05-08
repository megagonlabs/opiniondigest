# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch


class Search(object):

    def __init__(self, vocab_size, pad, unk, eos):
        self.pad = pad
        self.unk = unk
        self.eos = eos
        self.vocab_size = vocab_size
        self.scores_buf = None
        self.indices_buf = None
        self.beams_buf = None

    def _init_buffers(self, t):
        if self.scores_buf is None:
            self.scores_buf = t.new()
            self.indices_buf = torch.LongTensor().to(device=t.device)
            self.beams_buf = torch.LongTensor().to(device=t.device)

    def step(self, step, lprobs, scores):
        """Take a single search step.
        Args:
            step: the current search step, starting at 0
            lprobs: (bsz x input_beam_size x vocab_size)
                the model's log-probabilities over the vocabulary at the current step
            scores: (bsz x input_beam_size x step)
                the historical model scores of each hypothesis up to this point
        Return: A tuple of (scores, indices, beams) where:
            scores: (bsz x output_beam_size)
                the scores of the chosen elements; output_beam_size can be
                larger than input_beam_size, e.g., we may return
                2*input_beam_size to account for EOS
            indices: (bsz x output_beam_size)
                the indices of the chosen elements
            beams: (bsz x output_beam_size)
                the hypothesis ids of the chosen elements, in the range [0, input_beam_size)
        """
        raise NotImplementedError

    def set_src_lengths(self, src_lengths):
        self.src_lengths = src_lengths


class BeamSearch(Search):

    def __init__(self, vocab_size, pad, unk, eos):
        super().__init__(vocab_size, pad, unk, eos)

    def step(self, step, lprobs, scores, output_beam_size):
        super()._init_buffers(lprobs)
        bsz, beam_size, vocab_size = lprobs.size()

        if step == 0:
            # at the first step all hypotheses are equally likely, so use
            # only the first beam
            lprobs = lprobs[:, ::beam_size, :].contiguous()
        else:
            # make probs contain cumulative scores for each hypothesis
            lprobs.add_(scores[:, :, step - 1].unsqueeze(-1))

        torch.topk(
            lprobs.view(bsz, -1),
            k=min(
                # Take the best 2 x beam_size predictions. We'll choose the first
                # beam_size of these which don't predict eos to continue with.
                output_beam_size * 2,
                lprobs.view(bsz, -1).size(1) - 1,  # -1 so we never select pad
            ),
            out=(self.scores_buf, self.indices_buf),
        )
        torch.div(self.indices_buf, vocab_size, out=self.beams_buf)
        self.indices_buf.fmod_(vocab_size)
        return self.scores_buf, self.indices_buf, self.beams_buf

class BeamSearchNMT(Search):
    """Google's Neural Machine Translation beam search implementation:
    https://arxiv.org/pdf/1609.08144.pdf
    """
    def __init__(self, vocab_size, pad, unk, eos, alpha=0.6):
        super().__init__(vocab_size, pad, unk, eos)
        self.alpha = alpha

    def step(self, step, lprobs, scores, output_beam_size):
        super()._init_buffers(lprobs)
        bsz, beam_size, vocab_size = lprobs.size()

        # Calculate 
        if step == 0:
            # at the first step all hypotheses are equally likely, so use
            # only the first beam
            lprobs = lprobs[:, ::beam_size, :].contiguous()
        else:
            # make probs contain cumulative scores for each hypothesis
            lprobs.add_(scores[:, :, step - 1].unsqueeze(-1))
        
        # Update by length penalty
        length_penalty = self._length_penalty(step)
        scores = lprobs / length_penalty

        torch.topk(
            scores.view(bsz, -1),
            k=min(
                # Take the best 2 x beam_size predictions. We'll choose the first
                # beam_size of these which don't predict eos to continue with.
                output_beam_size * 2,
                scores.view(bsz, -1).size(1) - 1,  # -1 so we never select pad
            ),
            out=(self.scores_buf, self.indices_buf),
        )
        torch.div(self.indices_buf, vocab_size, out=self.beams_buf)
        self.indices_buf.fmod_(vocab_size)
        return self.scores_buf, self.indices_buf, self.beams_buf

    def _length_penalty(self, step):
        """Length penalty:
            lp(Y) = ((5+|Y|)^alpha)/(5+1)^alpha
        Args:
            step: the current search step, starting at 0
        Returns:
            length_penalty: float
                length penalty of current step.
        """
        return ((step+1)/6.0)**self.alpha

    def _coverage_penalty(self, attn):
        """Coverage penalty:
            cp(X;Y) =beta * sum(log(min(attn_i_j, 1.0)))
        Args:
            attn: (bsz x beam_size x src_seqlen)
                the total attention prob of i-th source word.
        Return:
            coverage_penalty: (bsz x beam_size) or 0.0
                the coverage penalty for each beam.
        TOOD (@xiaolan): finish implementation
        """
        return 0.0

class DiverseBeamSearch(Search):
    """Diverse Beam Search.
    See "Diverse Beam Search: Decoding Diverse Solutions from Neural Sequence
    Models" for details.
    We only implement the Hamming Diversity penalty here, which performed best
    in the original paper.
    Recommended setting in original paper:
    num_groups: "Setting G=beam_size allows for the maximum exploration 
        of the space,"
    diversity_strength: "We find a wide range of λ values (0.2 to 0.8) work well 
        for most tasks and datasets."
    """

    def __init__(self, vocab_size, pad, unk, eos, num_groups, diversity_strength=0.5):
        super().__init__(vocab_size, pad, unk, eos)
        self.num_groups = num_groups
        self.diversity_strength = -diversity_strength
        self.diversity_buf = None
        self.beam = BeamSearch(vocab_size, pad, unk, eos)

    def step(self, step, lprobs, scores, output_beam_size):
        super()._init_buffers(lprobs)
        bsz, beam_size, vocab_size = lprobs.size()
        num_groups = self.num_groups if beam_size > 1 else 1
        if beam_size % num_groups != 0:
            raise ValueError(
                'DiverseBeamSearch requires --beam to be divisible by the number of groups'
            )
        # initialize diversity penalty
        if self.diversity_buf is None:
            self.diversity_buf = lprobs.new()
        torch.zeros(lprobs[:, 0, :].size(), out=self.diversity_buf)

        scores_G, indices_G, beams_G = [], [], []
        for g in range(num_groups):
            lprobs_g = lprobs[:, g::num_groups, :]
            scores_g = scores[:, g::num_groups, :] if step > 0 else None

            # apply diversity penalty
            if g > 0:
                lprobs_g = torch.add(lprobs_g, self.diversity_strength, self.diversity_buf.unsqueeze(1))
            else:
                lprobs_g = lprobs_g.contiguous()

            scores_buf, indices_buf, beams_buf = self.beam.step(step, lprobs_g, scores_g, output_beam_size)
            beams_buf.mul_(num_groups).add_(g)

            scores_G.append(scores_buf.clone())
            indices_G.append(indices_buf.clone())
            beams_G.append(beams_buf.clone())

            # update diversity penalty
            self.diversity_buf.scatter_add_(
                1,
                indices_buf,
                self.diversity_buf.new_ones(indices_buf.size())
            )

        # interleave results from different groups
        self.scores_buf = torch.stack(scores_G, dim=2, out=self.scores_buf).view(bsz, -1)
        self.indices_buf = torch.stack(indices_G, dim=2, out=self.indices_buf).view(bsz, -1)
        self.beams_buf = torch.stack(beams_G, dim=2, out=self.beams_buf).view(bsz, -1)
        return self.scores_buf, self.indices_buf, self.beams_buf


class NgramBlocking:
    def __init__(self, no_repeat_ngram_size):
        """ Ngram blocking: avoid generating sequences with repetitive n-grams.
        """
        self.no_repeat_ngram_size = no_repeat_ngram_size

    def update(self, step, sequence, lprobs):
        """ Update lprobs: set token probability as -math.inf for repetitive n-grams
        """
        blocked_ngrams = self._gen_blocked_ngram(sequence[:step+1])
        banned_tokens = self._gen_banned_tokens(step, sequence, blocked_ngrams)
        lprobs[banned_tokens] = -math.inf
        return lprobs

    def _gen_blocked_ngram(self, sequence):
        """ Generate a dict of ngrams that already exist in previous sequence. 
        e.g., 
        Given a sequence of: [0, 1, 2, 3, 4]
        And no_repeat_ngram_size = 3,

        The blocked ngrams are: 
        {
            (0, 1): [2]
            (1, 2): [3]
            (2, 3): [4]
        }
        Modified from https://github.com/pytorch/fairseq/sequence_generator.py#L338-L450
        """
        blocked_ngrams = {}
        for ngram in zip(*[sequence[i:].tolist() for i in range(self.no_repeat_ngram_size)]):
            blocked_ngrams[tuple(ngram[:-1])] = blocked_ngrams.get(tuple(ngram[:-1]), []) + [ngram[-1]]
        return blocked_ngrams

    def _gen_banned_tokens(self, step, sequence, blocked_ngrams):
        """ Generate tokens that should be banned for (step+1).
        """
        banned_tokens = []
        if step+2-self.no_repeat_ngram_size < 0:
            return banned_tokens
        ngram_index = tuple(sequence[step+2-self.no_repeat_ngram_size:step+1].tolist())
        return blocked_ngrams.get(ngram_index, [])
