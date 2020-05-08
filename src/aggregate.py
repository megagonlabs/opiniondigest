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

from collections import Counter
import commentjson
import csv
import gensim.downloader as api
import os
from nltk import word_tokenize as tokenizer
import random
from random import shuffle
import torch
import torch.nn.functional as F
from tqdm import tqdm
import sys


class Extraction:
    def __init__(self, opn, asp, att, pol, w2v, threshold=0.9):
        self.opn = opn
        self.asp = asp
        self.att = att
        self.pol = pol
        self.emb = self._get_emb(w2v)
        self.threshold = threshold
        self.is_valid = True

    def is_duplicate(self, other):
        """ Determine if two extractions are the same or not
		Args:
			other (Extraction object)
		Returns:
			True or False
		Rule:
			Consider two extractions as the same if their w2v cosine similarity
			is above the specified threshold:
				ext1 == ext2, if cosine(ext1.emb, ext2.emb) >= threshold
		"""
        if self.att != other.att or self.pol != other.pol:
            return False
        similarity = F.cosine_similarity(self.emb.unsqueeze(0),
                                         other.emb.unsqueeze(0))
        return similarity >= self.threshold

    def to_string(self):
        """ Extraction to string: used for extraction column"""
        info = [self.opn, self.asp, self.att, self.pol]
        return ",".join(info)

    def to_key(self):
        """ Extraction to key: used for input_text column"""
        return self.opn + " " + self.asp

    def _get_emb(self, w2v):
        """ Get the avgerage w2v embedding """
        toks = tokenizer(self.opn.lower()) + tokenizer(self.asp.lower())
        self.is_valid = len(toks) > 0
        if not self.is_valid:
            return None
        embs = torch.stack([self._get_tok_emb(tok, w2v) for tok in toks], dim=0)
        return torch.mean(embs, dim=0)

    @staticmethod
    def _get_tok_emb(tok, w2v):
        """ Get the w2v embedding of a token """
        if tok not in w2v.vocab:
            return torch.zeros(w2v.vectors.shape[1])
        return torch.tensor(w2v.vectors[w2v.vocab[tok].index])


class Entity:
    def __init__(self, eid):
        self.eid = eid
        self.rids = []
        self.reviews = []
        self.exts = []

    def add_review(self, rid, review, exts, w2v, threshold):
        """ Add review into current entity
		Args:
			rid (int): review id
			review (str): review content
			exts (list(str)): extraction sequence
			w2v (glove2word2vec object): w2v object
			threhold (float): threshold for determine duplicate extraction
		"""
        self.rids.append(rid)
        self.reviews.append(review)
        cur_exts = []
        for ext in exts:
            if len(ext.strip()) < 1:
                continue
            opn, asp, att, pol = ext.split(",")
            ext_obj = Extraction(opn, asp, att, pol, w2v, threshold)
            if ext_obj.is_valid and ext_obj.emb is not None:
                cur_exts.append(ext_obj)
        self.exts.append(cur_exts)

    def select(self, n, k, att, pol):
        """ Select the top-k extraction.
		Args:
			n (int): number of reviews to summarize, 
			k (int): top-k 
			attr (str): attribute name or all
			pol (str): sentiment polarity or all
		Return:
			list of integer/string: output of the summary.
		"""
        # Select reviews
        rids_local = [i for i in range(len(self.rids))]
        shuffle(rids_local)
        selected = [self.rids[i] for i in rids_local[:n]]
        groups = self._deduplicate(rids_local[:n])
        # Further filter by selection rule
        filtered = []
        for exts in groups:
            # is_att = (att == "all") or (att == exts["att"])
            # is_pol = (pol == "all") or (pol == exts["pol"])
            is_att = (att == "all") or (att in exts["att"])            
            is_pol = (pol == "all") or (pol in exts["pol"])
            if is_att and is_pol:
                filtered.append(exts["exts"])
        # Sort by frequency
        k = min(k, len(filtered))
        filtered = sorted(filtered, key=lambda x: -len(x))
        exts = [self._select_repr(group) for group in filtered[:k]]
        return self._sel_to_row(n, selected, rids_local[:n], exts)

    def get_size(self):
        """ Return number of reviews. """
        return len(self.reviews)

    def _deduplicate(self, rids):
        """ Deduplicate extractions in greedy fashion.
		Args:
			rids (list[ int ]): list of selected review ids.
		"""
        # Deduplication
        filtered = []
        for rid in rids:
            for ext in self.exts[rid]:
                find_merge = False
                for exts_other in filtered:
                    if self._do_merge(ext, exts_other):
                        exts_other.append(ext)
                        find_merge = True
                        break
                if not find_merge:
                    filtered.append([ext])
        # Update attr/pol information for each group
        groups = []
        for exts in filtered:
            att = self._find_majority([ext.att for ext in exts])
            pol = self._find_majority([ext.pol for ext in exts])
            groups.append({"exts": exts, "att": att, "pol": pol})
        return groups

    @staticmethod
    def _do_merge(ext, exts_other):
        """ Validate wheter to merge two groups of extractins. """
        for ext_other in exts_other:
            if not ext.is_duplicate(ext_other):
                return False
        return True

    @staticmethod
    def _find_majority(values):
        """ Find the most frequent value in an array. """
        counter = Counter(values)
        return counter.most_common(1)[0][0]

    @staticmethod
    def _select_repr(exts, do_random=True):
        """ Select the extraction as the representative """
        # Randomly select one member as the representative.
        if do_random:
            rand_idx = random.randint(0, len(exts) - 1)
            return exts[rand_idx]
        # Select the representative as the centroid.
        embs = torch.stack([ext.emb for ext in exts], dim=0)
        centroid = torch.mean(embs, dim=0)
        similarities = F.cosine_similarity(embs, centroid.unsqueeze(0).expand(embs.size()))
        idx = torch.topk(similarities, k=1).indices[0].item()
        return exts[idx]

    def _sel_to_row(self, n, rids, local_rids, exts):
        """ Return output row of selected extractions
		Args:
			rids (list[int]): list of reviews
			exts (list(Extraction object)): list of selected extraction object.
		Returns:
			list of integer/string: output of the summary.
		"""
        row = [self.eid, ",".join([str(rid) for rid in rids]), str(len(rids))]
        for i in range(n):
            if i >= len(local_rids):
                row.append("")
            else:
                row.append(self.reviews[local_rids[i]])
        row.append(";".join([ext.to_string() for ext in exts]))
        row.append(" [SEP] ".join([ext.to_key() for ext in exts]))
        return row


class Input:
    def __init__(self, source_file, w2v, threshold):
        self.source_file = source_file
        self.threshold = threshold
        self.entities = self._read_file(source_file, w2v, threshold)

    def _read_file(self, source_file, w2v, threshold):
        """ Read reviews from file, compute w2v embedding for extractions.
		Args:
			source_file (str): source file path
			w2v (glove2word2vec object): w2v object
			threshold (float): threshold for determine duplicate extraction.
		"""
        num_lines = sum(1 for _ in open(source_file, "r"))
        pbar = tqdm(total=num_lines)
        entities = {}
        with open(source_file, "r") as file:
            reader = csv.reader(file, delimiter=",")
            next(reader)
            for row in reader:
                eid = row[0]
                rid = row[1]
                exts = row[3].split(";")
                if eid not in entities:
                    entities[eid] = Entity(eid)
                entities[eid].add_review(rid, row[2], exts, w2v, threshold)
                pbar.update(1)
        return list(entities.values())

    def select(self, n, k, att, pol, is_exact, emb_dim, gold_summary):
        """ Select entities/reviews/extractions according to selection rule;
		write selected entities/reviews/extraction into file.
		Args:
			n (int): number of reviews per entity
			k (int): top-k frequent extractions
			att (str): extraction selection rule -- attribute of the summary
			pol (str): extraction selection rule -- sentiment of the summary
		"""
        gold = None if len(gold_summary) == 0 else self._read_gold(gold_summary)
        selected = []
        for entity in self.entities:
            if is_exact and entity.get_size() < n:
                continue
            row = entity.select(n, k, att, pol)
            selected.append(row)
            if len(selected) >= 200:
                break
        print("*****Select {} out of {} entities*****".format(len(selected),
                                                              len(self.entities)))
        self._write_file(n, k, att, pol, selected, emb_dim, gold)

    def _get_target_name(self, n, k, att, pol, emb_dim):
        """ Generate target file name. """
        threshold = str(int(self.threshold * 10))
        agg_name = "_{}_{}_{}_{}_{}_{}".format(n, k, att, pol, emb_dim, threshold)
        target_file = self.source_file[:-4] + agg_name + ".csv"
        return target_file

    @staticmethod
    def _read_gold(file):
        gold = {}
        csvreader = csv.reader(open(file, "r"), delimiter=",")
        next(csvreader)
        for row in csvreader:
            gold[row[0]] = row[9]
        return gold

    def _write_file(self, n, k, att, pol, selected, emb_dim, gold):
        """ Write selected entities/reviews/extractions into file. """
        target_file = self._get_target_name(n, k, att, pol, emb_dim)
        writer = csv.writer(open(os.path.join(target_file), "w"))
        header = ["eid", "rids", "n"]
        header = header + ["gold_summary"] if gold is not None else header
        header += ["review_{}".format(i) for i in range(n)]
        header += ["extraction", "input_text"]
        writer.writerow(header)
        for row in selected:
            row = row[:3] + [gold[row[0]] if row[0] in gold else ""] + row[3:] if gold is not None else row
            writer.writerow(row)


if __name__ == "__main__":
    assert len(sys.argv) > 1, "Please specify prepare configuration file!"
    config_file = sys.argv[1]
    with open(config_file, "r") as file:
        configs = commentjson.loads(file.read())

    # Get all files
    source_files = [os.path.join("data", configs["p_name"], f) for f in configs["files"]]
    gold_file = os.path.join("data", configs["p_name"], configs["gold"])

    # Initialize w2v
    print("*****Load w2v ({}) matrix*****".format(configs["embedding"]))
    w2v = api.load(configs["embedding"])
    emb_dim = configs["embedding"][-3:]

    is_exact = True if configs["is_exact"] == "True" else False

    for source_file in source_files:
        print("*****Processing {}*****".format(source_file))
        inputs = Input(source_file, w2v, configs["threshold"])
        print("*****Generating {}*****".format(source_file))
        inputs.select(configs["num_review"], configs["top_k"], configs["attribute"], configs["sentiment"], is_exact,
                      emb_dim, gold_file)
