"""
Adapted from BertSenClu:
  https://github.com/JohnTailor/BertSenClu

Original implementation licensed under the MIT License.
Copyright (c) 2023 Johannes Schneider.
"""

# Python libs
import multiprocessing
import os
import pickle
from multiprocessing import Pool

import numpy as np

# PyTorch
import pandas as pd
import pysbd  # Sentence segmenter https://github.com/nipunsadvilkar/pySBD #pip install pysbd
import torch
import torch.nn as nn

# Other libs
from gensim import utils as gut  # tokenizer
from nltk.stem import WordNetLemmatizer

from src.utils.encoder import SentenceEncoder

SentenceTransformer = SentenceEncoder

# import sys
# sys.path.append('bertsenclu') #for visutils import
# from bertsenclu import visUtils


class SenClu:
    def __init__(
        self,
        device="auto",
        encoder_model_name="sentence-transformers/all-mpnet-base-v2",
        encode_prefix=None,
        encoder_backend="auto",
        pooling=None,
        encode_prompt=None,
        encode_prompt_name=None,
        encode_batch_size=128,
        model_kwargs=None,
        tokenizer_kwargs=None,
        normalize_embeddings=None,
        truncate_dim=None,
        encoder_fit_tokenized=None,
    ):
        """
        Inputs:
        device...where to do computations : 'cpu' or 'cuda' (= on GPU) or 'auto' (= choose cuda if available else cpu)
        """
        super(SenClu, self).__init__()
        self.maxTopWords = 50  # Top words per Topic computed
        self.device = device
        self.encoder_model_name = encoder_model_name
        self.encode_prefix = encode_prefix if encode_prefix else None
        self.encoder_backend = encoder_backend
        self.pooling = pooling
        self.encode_prompt = encode_prompt
        self.encode_prompt_name = encode_prompt_name
        self.encode_batch_size = encode_batch_size
        self.model_kwargs = {} if model_kwargs is None else dict(model_kwargs)
        self.tokenizer_kwargs = (
            {} if tokenizer_kwargs is None else dict(tokenizer_kwargs)
        )
        self.normalize_embeddings = normalize_embeddings
        self.truncate_dim = truncate_dim
        self.encoder_fit_tokenized = (
            None
            if encoder_fit_tokenized is None
            else [list(tokens) for tokens in encoder_fit_tokenized]
        )
        self.verbose = True

    def _apply_encode_prefix(self, sentences):
        if not self.encode_prefix:
            return sentences
        prefix = self.encode_prefix
        return [
            text if text.startswith(prefix) else f"{prefix}{text}" for text in sentences
        ]

    def computeTopicModel_SenClu(
        self, docs, device, ntopics, nepoch=20, alpha=1, epsilon=0.001
    ):  # dimension of sentence embeddings # initial offset to add to topic to smoothen p(t|d) distribution; # alpha = final offset, serves also as prior to determine how many topics per document exist
        """
        Compute topic model given docs with sentence embeddings
        ---------------------------------------------
        Inputs:
        docs = documents as string
        device = cuda (GPU) or cpu
        ntopics = number of topics
        nepoch = number of training epochs
        alpha = parameter giving the preference to have many or few topics per document, i.e., we used p(w|topic t) = p(w,vec topic t) *(alpha+ probability(topic t |document) , 0...means tend towards having only 1 topic, alpha...probability of topic in document is proportional to alpha even if no sentence of document belongs to topic t
        epsilon = Convergence criterion: Stop computation if topic distribution changes by less than this for an epoch (even before nepoch have been executed)
        ----------------------
        Returns:
        ptd... probability topic given document (for each doc in input) ;Dimensions: #topics x #docs
        vec_t... topic vectors; Dimensions: #topics x embedding dim of pretrained vecs:
        assign_t ... topic assignments to each sentences;  ~ #docs x #sentences in doc
        """
        if self.verbose:
            print("Training topic model")
        nd = len(docs)  # number of docs
        istart = ioff = (
            8  # initial smoothening of p(t|d), not very critical, if doc is very long, it should also be a bit larger
        )
        nWarmUp = 0  # number of warm Up epochs -> cluster more documents and later split them; (not needed)

        # Initialize topic vectors v_t,
        embDims = len(
            docs[0][0]
        )  # embDims = dimensions of pre-trained sentence encodings, usually 384
        vec_t = (
            np.random.random((embDims, ntopics)) - 0.5
        )  # Initialize topic vectors randomly
        sum_vt = np.sqrt(np.sum(vec_t**2, axis=0, keepdims=True))
        vec_t = vec_t / (sum_vt + 1e-10)
        vec_t = vec_t.astype(dtype=np.float32)

        closs = 1  # loss
        assignSen_ptd = [
            [] for _ in range(nd)
        ]  # topic assignments of each sentence account for prob(t|d)
        # assignSen = [[] for _ in range(nd)]  # topic assignments of each sentence without prob(t|d)
        prob_t = [[] for _ in range(nd)]  # topic prob of assigned sentence
        # Use new torch.amp.autocast to avoid deprecation warning
        with torch.amp.autocast("cuda" if device.startswith("cuda") else "cpu"):
            ptd = torch.ones((ntopics, nd)) / ntopics  # p(t,d) uniform
            ptd = ptd.to(device)
            vec_t = torch.from_numpy(vec_t)
            vec_t = vec_t.to(device)
            docs_torch = [torch.from_numpy(x.astype(np.float32)) for x in docs]
            docs_torch = [x.to(device) for x in docs_torch]

        # Start training
        for epoch in range(nepoch):

            # cbatches = np.random.permutation(len(docs_torch))  # we don't use the dataloader but rather permute the batch ids
            cbatches = range(len(docs_torch))
            nAssign_t = torch.zeros(ntopics).to(
                device
            )  # number of assigned sentences to topic
            newvec_t = torch.zeros_like(vec_t).to(
                device
            )  # This will become the new topic vector v_t after the epoch

            for ib, i in enumerate(cbatches):  # go through all docs
                dx = docs_torch[i]  # current document
                with torch.amp.autocast("cuda" if device.startswith("cuda") else "cpu"):
                    with torch.no_grad():
                        # Compute similarity of document, i.e., its sentences, and topic vectors
                        pts = torch.matmul(
                            dx, vec_t
                        )  # Dot product of sentence vectors in doc and topic vectors
                        prod = pts * (
                            ptd[:, i] + ioff
                        )  # Compute p(s|t)*p(t|d) , (mean is not needed) #* torch.mean(pts[pts > 0]
                        vals, mind = torch.max(
                            prod, dim=1
                        )  # Compute most likely topic and take it as assignment

                        # Update p(t|d)
                        cntd = nn.functional.one_hot(mind, ntopics)
                        sntd = torch.sum(cntd, dim=0)
                        ptd[:, i] = sntd / torch.sum(sntd)

                        # Update topic vectors
                        td = torch.transpose(dx, 0, 1)
                        prod = torch.matmul(td, cntd.float())
                        sntd = sntd * ptd[:, i]
                        prod = prod * ptd[:, i]
                        newvec_t = (
                            newvec_t + prod
                        )  # use only some of sentence vectors -> could weight by ptd and do rolling update
                        nAssign_t += sntd

                        # Other stuff:
                        assignSen_ptd[i] = (
                            mind.cpu().numpy()
                        )  # Remember assignment (only needed at the end)
                        prob_t[i] = vals.cpu().numpy()
                        closs = (
                            0.95 * closs + 0.05 * torch.mean(vals).item()
                            if epoch > 0 or i > 10
                            else 0.8 * closs + 0.2 * torch.mean(vals).item()
                        )  # update loss

            if self.verbose:
                print("  Epoch:", epoch, " Loss", np.round(closs, 4))
            if (
                epoch >= nWarmUp
            ):  # warmup epochs, where we do more of a document clustering with small alpha; this helps to distribute frequent words
                ioff = (
                    istart if epoch == nWarmUp else max(ioff / 2, alpha)
                )  # Update the smoothing constant
            else:
                ioff = alpha
            with torch.no_grad():  # Update the topic vector
                newvec_t = newvec_t / (nAssign_t + 1e-8)
                diff = torch.sqrt(torch.sum((newvec_t - vec_t) ** 2))
                vec_t = newvec_t
                if diff < epsilon and epoch > 4:
                    break  # Stop if vectors don't change significantly anymore
        return ptd.cpu().numpy(), vec_t, assignSen_ptd, prob_t

    def inferTopicModel_SenClu(
        self,
        docs,
        device,
        vec_t,
        ntopics,
        nepoch=20,
        alpha=1,
        epsilon=0.001,
    ):
        """
        Infer p(topic|document) for documents with fixed topic vectors.

        This is intentionally separate from computeTopicModel_SenClu: it updates
        only document-local topic proportions and sentence assignments, never the
        learned global topic vectors.
        """
        nd = len(docs)
        if nd == 0:
            return (
                np.zeros((ntopics, 0), dtype=np.float32),
                [],
                [],
            )
        istart = ioff = 8
        assignSen_ptd = [[] for _ in range(nd)]
        prob_t = [[] for _ in range(nd)]
        autocast_device = "cuda" if device.startswith("cuda") else "cpu"
        with torch.amp.autocast(autocast_device):
            ptd = torch.ones((ntopics, nd), device=device) / ntopics
            if not torch.is_tensor(vec_t):
                vec_t = torch.from_numpy(np.asarray(vec_t, dtype=np.float32))
            vec_t = vec_t.to(device)
            docs_torch = [
                torch.from_numpy(x.astype(np.float32)).to(device) for x in docs
            ]

        for epoch in range(nepoch):
            previous_ptd = ptd.clone()
            for i, dx in enumerate(docs_torch):
                with torch.amp.autocast(autocast_device):
                    with torch.no_grad():
                        pts = torch.matmul(dx, vec_t)
                        prod = pts * (ptd[:, i] + ioff)
                        vals, mind = torch.max(prod, dim=1)
                        cntd = nn.functional.one_hot(mind, ntopics)
                        sntd = torch.sum(cntd, dim=0)
                        total = torch.sum(sntd)
                        if total > 0:
                            ptd[:, i] = sntd / total
                        assignSen_ptd[i] = mind.cpu().numpy()
                        prob_t[i] = vals.cpu().numpy()
            ioff = istart if epoch == 0 else max(ioff / 2, alpha)
            diff = torch.sqrt(torch.sum((ptd - previous_ptd) ** 2))
            if diff < epsilon and epoch > 4:
                break
        return ptd.cpu().numpy(), assignSen_ptd, prob_t

    """
    Lemmatize topics
    -----------
    Inputs
    lemDict...Dictionary of word -> lemma
    topics... list of list of words resembling top words of topics
    Returns
    lemmatized topics... list of list of words resembling top lemmatized words of topics
    """

    def lemTopics(self, topics, lemDict):
        nt = []
        for t in topics:
            ct = []
            for w in t:
                cw = w
                if cw in lemDict:
                    cw = lemDict[w]
                elif cw.lower() in lemDict:
                    cw = lemDict[cw.lower()]
                if cw not in ct:
                    ct.append(cw)
            nt.append(ct)
        return nt

    def get_topics(self, sendocs, assign_t):
        """
        Compute topics (list of words)
        -------------------
        Input
        ptd... probability topic given document (for each doc in input) ;Dimensions: #topics x #docs
        vec_t... topic vectors; Dimensions: #topics x embedding dim of pretrained vecs:
        assign_t ... topic assignments to each sentences;  ~ #docs x #sentences in doc
        ntopw ... number of words to return per topic (words with largest score are returned)
        -------------------
        Return
        topics ... list of list of words; each list of words contains ntopw words
        """
        # Preprocess corpus -> We need to tokenize documents into words, we also lower case and lemmatize
        # NOTE: Other methods use their own preprocessing, we also lemmatize the final topics of other methods
        dic = {}  # dictionary word to id
        nw = 2000  # number of initial words (this is grown)
        nfw = 0
        nd = len(sendocs)
        nt = self.ntopics
        occ_wt = np.zeros(
            (nw, nt), dtype=np.int32
        )  # matrix with number of occ of word in topic (it is grown as needed)
        occ_wd = np.zeros(
            (nw, nd), dtype=np.int32
        )  # matrix with number of occ of word in document
        occ_dt = np.zeros(
            (nd, nt), dtype=np.int32
        )  # matrix with number of occ of topic per document
        # from nltk.corpus import stopwords
        # stop_words = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()
        corpus = []
        lems = {}  # build map of words to lemmas
        for id in range(nd):  # go through all docs
            dass = assign_t[id]  # assignments of sentences of doc
            doc = sendocs[id]  # sentences in doc
            if self.verbose and len(dass) != len(doc):
                print(
                    "Failed not all sentences assigned",
                    id,
                    len(dass),
                    len(doc),
                    " NDoc",
                    nd,
                    len(sendocs),
                    doc,
                )
            cdoc = []
            for it, s in enumerate(doc):  # go through all sentences
                toks = gut.tokenize(s, lower=True)
                toks = [ctok for ctok in toks if len(ctok) > 1]  # ignore single chars
                # toks = [ctok for ctok in toks if not ctok in stop_words]  # Optional: Igore stop words
                for w in toks:
                    if w not in lems:
                        lems[w] = lemmatizer.lemmatize(w)
                toks = self.lemTopics([toks], lems)[0]
                ctopic = dass[it]
                for w in toks:  # go through all tokens
                    cdoc.append(w)
                    if w not in dic:  # add new word to dic
                        dic[w] = nfw
                        nfw += 1
                        if (
                            nfw > occ_wt.shape[0]
                        ):  # if found max number of words expand matrix
                            occ_wt = np.concatenate(
                                [occ_wt, np.zeros((nw, nt), dtype=np.int32)], axis=0
                            )
                            occ_wd = np.concatenate(
                                [occ_wd, np.zeros((nw, nd), dtype=np.int32)], axis=0
                            )
                    occ_wt[dic[w], ctopic] += 1  # add count of words in topic
                    occ_wd[dic[w], id] += 1
                    occ_dt[id, ctopic] += 1
            corpus.append(cdoc)

        denom = np.sum(occ_wt, axis=1) + 1e-8
        denom = denom.reshape(-1, 1)
        # scores_word_topic = (occ_wt-denom/self.ntopics) ** 0.5 * np.clip(occ_wt / denom-1/self.ntopics,0,1) #This is an extenstion from the paper: Essentially, by look at how much more the probability is above "chance", the advantage is that otherwise we have strong dependency o nthe number of topics, i.e. , say we have ntopics=2 and the word "a" occurs 1000 times, by chance it occurs very frequent everywhere, byt for ntopics=200. by chance it occurs a factor 100 less # * np.sum(awt, axis=0,keepdims=True)/np.sum(awt)
        scores_word_topic = np.clip(
            (occ_wt - denom / self.ntopics), 0, 1e10
        ) ** 0.5 * np.clip(occ_wt / denom - 1 / self.ntopics, 0, 1)

        # Get indexes of words with top scores
        sw = np.zeros((self.maxTopWords, nt), dtype=np.int32)
        swscores = np.zeros((self.maxTopWords, nt), dtype=np.float32)
        index_top_scores = np.copy(scores_word_topic)
        for i in range(self.maxTopWords):
            wind = np.argmax(index_top_scores, axis=0)
            mscore = np.max(index_top_scores, axis=0)
            sw[i] = wind
            swscores[i] = mscore
            index_top_scores[wind, np.arange(nt)] = (
                -1
            )  # set to maximum so that wont be taken again

        # Get words per topic from indexes
        idic = {dic[w]: w for w in dic}
        topicsAsWordScoreList = []
        for t in range(nt):
            ws = sw[:, t]
            scores = swscores[:, t]
            # lws=[idic[w] for w in ws if w in idic]
            lws = [(idic[w], s) for w, s in zip(ws, scores)]
            topicsAsWordScoreList.append(lws)
        return topicsAsWordScoreList

    def segmenter(self, docs):
        """
        Segment docs into sentences
        Inputs: list of documents, each document being a string
        Returns: list of documens, each document being a list of sentences
        """
        pid, docs = docs
        seg = pysbd.Segmenter(language="en", clean=False)
        adocs = []
        out = 300
        for id, d in enumerate(docs):
            segs = seg.segment(d)
            adocs.append(segs)
            if self.verbose and id == out:
                print("   ProcessID:", pid, " Segmented by process:", id)
                out *= 4
        return adocs

    # Split list a into n parts
    def split(self, a, n):
        k, m = divmod(len(a), n)
        return (a[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n))

    def getEncodedSentences(self, docs, device):
        """
        Segment docs into sentences
        Input: list of documents, each document being a string
        Returns: list of documens, each document being a list of sentences
        """
        if self.verbose:
            print("Segmentation of docs into sentences; Total Docs: ", len(docs))
        # Segment documents into sentences
        nseg = min(8, multiprocessing.cpu_count() - 1)
        # Avoid spawning children from a daemon process (category-level parallel)
        if multiprocessing.current_process().daemon:
            nseg = 1
        cpool = Pool(nseg) if nseg > 1 else None
        # We already have segmented documents.
        # docs_segmented = cpool.map(self.segmenter, zip(np.arange(len(px)), px))
        # docs_segmented = sum(docs_segmented, [])
        docs_segmented = docs
        filtered_docs = []
        nFailed = 0
        for i, d in enumerate(docs_segmented):
            if len(d) == 0:
                if self.verbose and nFailed < 5:
                    print(
                        " Removing empty sentence embedded doc; Original doc ID",
                        i,
                        " Original length:",
                        len(docs[i]),
                        (
                            ""
                            if len(docs[i]) == 0
                            else "Content enclosed in '--' (e.g., it might contain only 'return characters'):  --"
                            + docs[i]
                            + "--"
                        ),
                    )
                nFailed += 1
            else:
                filtered_docs.append(d)
        if nFailed > 0:
            print(
                " Number of empty sentenced embedded docs:",
                nFailed,
                (
                    "  (Use verbose option to see first 5 docs)"
                    if not self.verbose
                    else ""
                ),
            )
        if cpool is not None:
            cpool.close()
            cpool.join()
        if self.verbose:
            print(
                "Embed sentences using a sentence embedder; Total Docs: ",
                len(filtered_docs),
            )
        # get Sentence Embeddings -> this can be parallelized
        if SentenceTransformer is SentenceEncoder:
            model = SentenceEncoder(
                self.encoder_model_name,
                device=device,
                encode_prefix=self.encode_prefix,
                backend=self.encoder_backend,
                pooling=self.pooling,
                encode_prompt=self.encode_prompt,
                encode_prompt_name=self.encode_prompt_name,
                encode_batch_size=self.encode_batch_size,
                model_kwargs=self.model_kwargs,
                tokenizer_kwargs=self.tokenizer_kwargs,
                normalize_embeddings=self.normalize_embeddings,
                truncate_dim=self.truncate_dim,
            )
            if getattr(model, "requires_fit", False):
                model.fit_tokenized(self.encoder_fit_tokenized or [])
            apply_prefix = False
        else:
            model = SentenceTransformer(self.encoder_model_name, device=device)
            model.eval()
            apply_prefix = True
        docs_sentenceEncoded = []
        out = 300
        for id, d in enumerate(filtered_docs):
            with torch.no_grad():
                encode_input = self._apply_encode_prefix(d) if apply_prefix else d
                segs = model.encode(encode_input)
            docs_sentenceEncoded.append(segs)
            if self.verbose and id == out:
                print("  Encoded Docs:", id)
                out *= 4
        return filtered_docs, docs_sentenceEncoded

    def fit_transform(
        self,
        train_docs,
        test_docs,
        nTopics=50,
        alpha=None,
        nEpochs=40,
        loadAndStoreInFolder=None,
        verbose=True,
    ):
        """
        Compute topic_model given raw text docs, optional: stores partial sentence embeddings for recomputation with other parameters
        --------------------
        Input:
        docs...documents
        alpha = parameter giving the preference to have many or few topics per document, i.e., we used p(w|topic t) = p(w,vec topic t) *(alpha+ probability(topic t |document) , 0...means tend towards having only 1 topic, alpha...probability of topic in document is proportional to alpha even if no sentence of document belongs to topic t
        ntopwords...number of words returned per topic (they are sorted in descending order by relevance)
        storeInFolder...path to a folder, where docs (split into sentences) and sentence embeddings of docs are stored; if they exist, these are loaded and used for topic modeling ; this speeds up comutation of different models on same corpus
        verbose...Print state of computation during topic modeling process
        Returns:
        topics...list of topics, each topic contains a list of "ntopwords" words sorted in descending order by relevance
        probs...topic probabilities
        """

        train_index = len(train_docs)
        docs = train_docs + test_docs
        if alpha is None:
            alpha = 1 / np.sqrt(nTopics)
        device = str(self.device)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.verbose = verbose
        cache_path = (
            None
            if loadAndStoreInFolder is None
            else os.path.join(loadAndStoreInFolder, "topicDat.pic")
        )
        # get encoded Sentences
        if cache_path is None or not os.path.exists(cache_path):
            self.sendocs, embdocs = self.getEncodedSentences(docs, device)
            if loadAndStoreInFolder is not None:
                os.makedirs(loadAndStoreInFolder, exist_ok=True)
                with open(cache_path, "wb") as f:
                    pickle.dump([self.sendocs, embdocs], f)
        else:
            if self.verbose:
                print(
                    "Loading docs split into sentences and embedded sentences for faster topic modeling"
                )
            with open(cache_path, "rb") as f:
                self.sendocs, embdocs = pickle.load(f)
        self.ntopics = nTopics
        self.alpha = alpha
        self.embdocs = embdocs
        train_embdocs = embdocs[:train_index]
        test_embdocs = embdocs[train_index:]
        # Topic modeling is fitted only on train documents. Test documents are
        # inferred with fixed topic vectors to avoid train/test leakage.
        train_ptd, self.vec_t, train_assign, train_prob = self.computeTopicModel_SenClu(
            train_embdocs, device, nTopics, nepoch=nEpochs, alpha=alpha
        )
        test_ptd, test_assign, test_prob = self.inferTopicModel_SenClu(
            test_embdocs,
            device,
            self.vec_t,
            nTopics,
            nepoch=nEpochs,
            alpha=alpha,
        )
        self.ptd = np.concatenate([train_ptd, test_ptd], axis=1)
        self.assignSen_ptd = list(train_assign) + list(test_assign)
        self.prob_t = list(train_prob) + list(test_prob)
        # We don't need word-level topics
        # self.topicsAsWordScoreList = self.get_topics(self.sendocs, self.assignSen_ptd)
        # self.pt=np.sum(self.ptd,axis=1)/self.ptd.shape[1]
        # return self.getTopics(),self.pt
        return train_ptd.transpose(), test_ptd.transpose()

    def get_sentence_topic_distribution_soft(self, temperature=1.0):
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        if (
            not hasattr(self, "embdocs")
            or not hasattr(self, "ptd")
            or not hasattr(self, "vec_t")
        ):
            raise RuntimeError(
                "Sentence-topic soft distribution requires fit_transform to be run first."
            )

        vec_t = self.vec_t
        if torch.is_tensor(vec_t):
            vec_t = vec_t.detach().cpu().numpy()
        vec_t = np.asarray(vec_t, dtype=np.float64)
        ptd = np.asarray(self.ptd, dtype=np.float64)  # (K, D)
        alpha = float(getattr(self, "alpha", 1.0 / np.sqrt(self.ntopics)))

        sentence_topic_soft = []
        for d, embdoc in enumerate(self.embdocs):
            dx = np.asarray(embdoc, dtype=np.float64)
            if dx.size == 0:
                sentence_topic_soft.append(
                    np.zeros((0, self.ntopics), dtype=np.float32)
                )
                continue
            scores = (dx @ vec_t) * (ptd[:, d] + alpha)[None, :]
            if temperature != 1.0:
                scores = scores / temperature
            scores -= scores.max(axis=1, keepdims=True)
            probs = np.exp(scores)
            row_sums = probs.sum(axis=1, keepdims=True)
            bad_rows = (~np.isfinite(row_sums)) | (row_sums <= 0.0)
            if np.any(bad_rows):
                probs[bad_rows[:, 0]] = 1.0 / float(self.ntopics)
                row_sums = probs.sum(axis=1, keepdims=True)
            probs /= row_sums
            sentence_topic_soft.append(probs.astype(np.float32))
        return sentence_topic_soft

    def getTopics(self, wordsPerTopic=10):
        return [[w for w, s in t[:wordsPerTopic]] for t in self.topicsAsWordScoreList]

    def getTopicsWitchScores(self):
        return self.topicsAsWordScoreList

    def getTopicDocumentDistribution(self):
        return self.ptd

    def getTopicDistribution(self):
        return self.pt

    def getTopDocsPerTopic(self, ntop=20):
        sortDocs = np.argsort(self.ptd, axis=1)  # ascending order!
        topDocs = []
        for i in range(self.ntopics):
            topd = sortDocs[i][-ntop:]
            topd = topd[::-1]  # descending order
            scores = self.ptd[i][topd]
            csen = [self.sendocs[itop] for itop in topd]
            topDocs.append(list(zip(csen, scores, topd)))
        return topDocs  # return document ID, doc score and doc

    def getTopSentencesPerTopic(self, ntop=20):
        perTop = [[] for _ in range(self.ntopics)]
        for i in range(len(self.sendocs)):
            vals = self.prob_t[i]
            tops = self.assignSen_ptd[i]
            for j in range(len(vals)):
                perTop[tops[j]].append((vals[j], i, j))
        for t in range(self.ntopics):
            s = sorted(perTop[t], reverse=True)
            ct = []
            for ctop in range(ntop):
                if len(s) <= ctop:
                    break
                v, i, j = s[ctop]
                ct.append((self.sendocs[i][j], v))
            perTop[t] = ct
        return perTop

    def saveOutputs(
        self,
        folder="Bert-SenClu",
        topWordPerTopic=10,
        topSenPerTopic=10,
        topDocsPerTopic=10,
        maxSenPerDoc=50,
        addTopicMarkup=True,
        createVisual=True,
    ):
        """
        Produce csv and visualization files and store in given folder; The csv can also be used for exploration and visualization
        A topic contains a probability, top words (each with probability), top sentences (each with score), top documents for that topic (full docs, only topic sentences, topic sentences with context); each doch has a score and an ID of the original document in the dataset fed into the topic model)
        For top documents, three representations are output:
        - "FullDoc" The full document up to specified maximum number of sentences maxSenPerDoc
        - "ContextDoc"  The document always contains the first 5 and the last 2 sentences and otherwise only sentences belonging to the topic (plus 1 sentence before/after to give context)
        - "TopicOnlyDoc" Keep only sentences belonging to the document
        The csv file contains 1 row for each topic and each column corresponds to a single item, e.g., a topic word, a topic probability, a top sentence etc.
        -------------------------------------------------
        Input
        fileName ... name of csv file
        topSenPerTopic=15... number of top sentences per topic
        topDocsPerTopic=20 ... number of top documents per topic
        maxSenPerDoc=100 ... maximum number of displayed sentences per document
        addTopicMarkup=True ... for a top document of a topic, highlight sentences belonging to topic
        createVisual=True ... create and store visualization files
        Returns:
        None (stores files in given folder)
        """
        if self.verbose:
            print("Creating and storing Outputs")
        # get data
        senPerTop = self.getTopSentencesPerTopic(topSenPerTopic)
        docPerTop = self.getTopDocsPerTopic(topDocsPerTopic)

        # Compute  CSV
        header = ["Topic_ID", "Topic_Prob"]

        def r(x):
            return str(np.round(x, 4))

        if topWordPerTopic > self.maxTopWords:
            print(
                "Exceeded max words per Topic. Fixing to allowed maximum of ",
                self.maxTopWords,
            )
            topWordPerTopic = self.maxTopWords
        headW = sum(
            [["Word_" + str(i), "Prob_" + str(i)] for i in range(topWordPerTopic)], []
        )
        headS = sum(
            [["Sentence_" + str(i), "Prob_" + str(i)] for i in range(topSenPerTopic)],
            [],
        )
        headD = sum(
            [
                [
                    "FullDoc_" + str(i),
                    "ContextDoc_" + str(i),
                    "TopicOnlyDoc_" + str(i),
                    "Prob_" + str(i),
                    "DocID_" + str(i),
                ]
                for i in range(topDocsPerTopic)
            ],
            [],
        )
        rows = []

        def getFixedLenLi(li, nEle):
            conLi = [[w[0], r(w[1])] for w in li[:nEle]]
            conLi = sum(conLi, [])
            return conLi

        topicColor = "#D84141"
        for i in range(self.ntopics):
            topdat = [str(i), r(self.pt[i])]
            topW = getFixedLenLi(self.topicsAsWordScoreList[i], topWordPerTopic)
            topS = getFixedLenLi(
                [
                    (
                        (
                            (
                                "<span style='color:"
                                + topicColor
                                + "'>"
                                + s[0]
                                + "</span>"
                            )
                            if addTopicMarkup
                            else s[0]
                        ),
                        s[1],
                    )
                    for s in senPerTop[i]
                ],
                topSenPerTopic,
            )

            topD = docPerTop[i]
            conDoc = []
            # Add Docs - keep only sentences of the topic i and for each topic sentence one before and after as well as first and last sentences

            for d, dsc, did in topD:
                topSen = (np.array(self.assignSen_ptd[did]) == i).astype(np.int)
                senToKeep = np.copy(topSen)
                senToKeep[:5] = 1  # keep first and last sentences
                senToKeep[-2:] = 1
                senToKeep[:-1] += topSen[
                    1:
                ]  # for each topic sentence add one before and after of a different topic
                senToKeep[1:] += topSen[:-1]
                consens, topsens, alls = [], [], []
                thres = np.median(
                    self.prob_t[did][topSen > 0]
                )  # highlight top 50% of a sentence more
                skipSen = 0
                for inds, s in enumerate(d):
                    if topSen[inds]:  # Topic sentence?
                        if addTopicMarkup:  # add markup if topic sentence
                            bold = self.prob_t[did][inds] > thres
                            s = (
                                "<span style='color:"
                                + topicColor
                                + "'>"
                                + ("<b>" if bold else "")
                                + s
                                + ("</b>" if bold else "")
                                + "</span>"
                            )  # redish color
                        if len(topsens) < maxSenPerDoc:
                            topsens.append(s)
                    elif addTopicMarkup:  # add markup if topic sentence
                        s = "<span style='color:#7D88C7'>" + s + "</span>"  # blue color
                    if len(alls) < maxSenPerDoc:
                        alls.append(s)
                    if senToKeep[inds]:
                        if skipSen:  # if left out a sentence add dots
                            s = " ...>>" + str(skipSen) + "... " + s
                        if len(consens) < maxSenPerDoc:
                            consens.append(s)
                        skipSen = 0
                    else:
                        skipSen += 1
                conDoc.append(
                    [
                        " ".join(alls),
                        " ".join(consens),
                        " ".join(topsens if len(topsens) else [" "]),
                        r(dsc),
                        str(did),
                    ]
                )
            conDoc = sum(conDoc, [])
            topdat += topW + topS + conDoc
            rows.append(topdat)

        df = rows
        df = pd.DataFrame(df, columns=header + headW + headS + headD)
        colW = [c for c in df.columns if c.startswith("Word_")]
        df["TopicShort"] = df.apply(
            lambda r: "_".join([str(r["Topic_ID"])] + list(r[colW].values[:5]))[:60],
            axis=1,
        )
        os.makedirs(folder, exist_ok=True)
        df.to_csv(folder + "/topic_all_info.csv")

        # Create visualization of topics
        # if createVisual:
        #     if self.verbose: print("Creating Visualizations")
        #     visUtils.hierarchy(folder, nvec, df["TopicShort"].values)
        #     visUtils.tsne(self, nvec, folder, df["TopicShort"].values)
