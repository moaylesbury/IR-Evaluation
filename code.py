import math
import re
import numpy as np
from stemming.porter2 import stem
import random

from scipy.sparse import dok_matrix

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.datasets import make_multilabel_classification

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline

from gensim.test.utils import common_texts, common_corpus
from gensim.corpora.dictionary import Dictionary
from gensim.models import LdaModel


def format_number(n):
    # formats output numbers
    # all integers are cast to a string
    # all floats have two decimal places and are cast to a string

    # if sys/qrn num or already two decimal float
    # note query number goes to 10, so need to check for length 1 and 2
    if len(str(n)) == 1 or len(str(n)) == 2 or len(str(n)) == 4:
        return str(n)
    # if less than two decimal places add zeroes
    if len(str(n)) < 4:
        diff = 4-len(str(n))
        return str(n) + '0'*diff
    # if more than two decimal places then round
    if len(str(n)) > 4:
        to_round = int(str(n)[2:5])
        final_digit = int(str(to_round)[-1])
        # if the last digit is less than five use floor function
        if final_digit < 5:
            return str(math.floor(to_round/10)/100)
        # ese use ceil function
        else:
            return str(math.ceil(to_round/10)/100)


class System:
    def __init__(self, num):
        self.num = num
        self.queries = []

    def add_query_result(self, query_result):
        self.queries.append(query_result)

    def get_query(self, qry_num):
        # get list of QueryResults for qry_num
        queries = []
        for query in self.queries:
            if query.qry_num == qry_num:
                queries.append(query)
        return queries


class QueryResult:
    def __init__(self, sys_num, qry_num, doc_num, doc_rank, score):
        # has form system_number, query_number, doc_number, rank_of_doc, score
        self.sys_num  = sys_num
        self.qry_num  = qry_num
        self.doc_num  = doc_num
        self.doc_rank = doc_rank
        self.score    = score
    def pretty_print(self):
        print("system:", self.sys_num, " query:", self.qry_num, " doc:", self.doc_num, " doc rank:",
              self.doc_rank, " score: ", self.score)


class SystemResults:
    def __init__(self):
        self.systems = []

    def add_systems(self, systems):
        # systems is a list of System objects
        [add_system(sys) for sys in systems]

    def add_system(self, sys):
        self.systems.append(sys)

    def get_system(self, num):
        for sys in self.systems:
            if sys.num == num:
                return sys


class Qrel:
    # list of Results for query x in 1-6
    # there will be 6 objects
    def __init__(self, qry_num):
        self.qry_num = qry_num
        self.res_list = []

    def add_result(self, res):
        self.res_list.append(res)

    def get_doc_rel(self, doc):
        # get relevance of document
        for res in self.res_list:
            if res.doc_num == doc:
                return res.rel
        # if document not present return 0
        return 0


class Result:
    # individual result
    def __init__(self, qry_num, doc_num, rel):
        self.qry_num = qry_num
        self.doc_num = doc_num
        self.rel = rel

    def pretty_print(self):
        print("query: ", self.qry_num, " doc: ", self.doc_num, " rel: ", self.rel)


class Qrels:
    def __init__(self):
        self.qrels = []

    def add_qrel(self, qrel):
        self.qrels.append(qrel)

    def get_qrel(self, qry_num):
        # get qrel from query number
        for qrel in self.qrels:
            if qrel.qry_num == qry_num:
                return qrel


class EVAL:
    def __init__(self):
        pass

    def read_in_sys_res(self, filename):

        curr_sys = 1
        system_results = SystemResults()
        sys = System(curr_sys)
        # don't want the first line as it is just format
        for line in open(filename, 'r').readlines()[1:]:
            # split input string on commas
            split = line.split(',')
            # cast values to appropriate type and add to list
            vals = [int(val) for val in split[:-1]] + [float(split[-1])]
            # create
            qry = QueryResult(vals[0], vals[1], vals[2], vals[3], vals[4])
            if qry.sys_num == curr_sys:
                # add query result to system
                sys.add_query_result(qry)
            else:
                # add system to system results
                system_results.add_system(sys)
                # increment sys number
                curr_sys += 1
                # create new system
                sys = System(curr_sys)
                # add qrt
                sys.add_query_result(qry)
        # catch the last sys
        system_results.add_system(sys)
        # for query in system_results.get_system(1).queries:
        #     query.pretty_print()
        return system_results


    def read_in_qrels(self, filename):

        # final object - list of qrels. one for each query
        qrels = Qrels()

        curr_query = 1
        # queries 1 to 6
        # list of Result objects for query
        qrel = Qrel(curr_query)

        for line in open(filename, 'r').readlines()[1:]:
            split = line.split(',')
            vals = [int(val) for val in split]

            res = Result(vals[0], vals[1], vals[2])
            if res.qry_num == curr_query:
                qrel.add_result(res)
            else:
                # add qrel to qrels
                qrels.add_qrel(qrel)
                # increment query number
                curr_query += 1
                # create new qrel
                qrel = Qrel(curr_query)
                # add res
                qrel.add_result(res)
        # catch the last qrel
        qrels.add_qrel(qrel)
        # for res in qrels.get_qrel(4).res_list:
        #     res.pretty_print()
        return qrels


    def get_system_results(self):
        # reads in system_results.csv
        # has form system_number, query_number, doc_number, rank_of_doc, score

        # get system_results
        return self.read_in("system_results.csv")


    def get_qrels(self):
        # reads in qrels.csv
        # has form query_id, doc_id, relevance

        # get qrels
        return self.read_in("qrels.csv")


    def evaluate(self):
        # gets top n results
        n = 10
        # uses precision or recall metric

        # fetch the system results and qrels
        # only want the top 10 results as cutoff at 10 and ranked linearly

        file = open("ir_eval.csv", 'w')

        # TODO: change this, they don't need to pass the filename
        qrels = self.read_in_qrels("qrels.csv")
        sys_res = self.read_in_sys_res("system_results.csv")

        # TODO: first beginning with system1, need top 10 results for each query

        # output to be written, updated every iteration
        # of form [system_number,query_number,P@10,R@50,r-precision,AP,nDCG@10,nDCG@20]
        output = ""
        # IR systems are numbered 1-6
        for sys_num in range(1, 7):
            # print("=-=-=-=-=-=-=-=-=-=-=-=-=- IR SYSTEM ", sys_num, " =-=-=-=-=-=-=-=-=-=-=-=-=-")
            # queries are numbered 1-10
            for qry_num in range(1, 11):
                output = ""
                print("-----")
                if sys_num == 1 and qry_num == 10:
                    for a in sys_res.get_system(1).get_query(10):
                        a.pretty_print()
                print("-----")
                # print("~~~~~~~~~~~~~ system:1 , query:", qry_num, " ~~~~~~~~~~~~~")
                a = self.precision(sys_res.get_system(sys_num).get_query(qry_num)[:10], qrels)
                b = self.recall(sys_res.get_system(sys_num).get_query(qry_num)[:50], qrels)
                c = self.ap(sys_res.get_system(sys_num).get_query(qry_num), qrels)
                d = self.r_precison(sys_res.get_system(sys_num).get_query(qry_num), qrels)
                e = self.nDCG(sys_res.get_system(sys_num).get_query(qry_num)[:10], qrels)
                f = self.nDCG(sys_res.get_system(sys_num).get_query(qry_num)[:20], qrels)

                # do formatting
                for count, i in enumerate([sys_num, qry_num, a, b, c, d, e, f]):
                    # calculating output string for line
                    # for the sys_num and qry_num (count < 1) just append the number cast to a string
                    formatted = format_number(i)
                    # print("formatted: ", formatted)
                    output += formatted
                    output += " "
                # print(output)
                file.write(output + '\n')
        file.close()

    def get_relevant_for_query(self, qrels, qry_num):
        relevant = []
        for rel in qrels.get_qrel(qry_num).res_list:
            relevant.append(rel.doc_num)
        return relevant

    def get_retrieved_for_query(self, sys):
        retrieved = []
        for result in sys:
            retrieved.append(result.doc_num)
        return retrieved

    def precision(self, sys, qrels):
        # precision at cutoff 10
        # using precision = relevant and retrieved / retrieved
        # that is P = TP / (TP + FP)

        # get the query number
        qry_num = sys[0].qry_num

        # get all retreived documents for the query
        retrieved = self.get_retrieved_for_query(sys)

        # get all relevant documents for the query
        relevant = self.get_relevant_for_query(qrels, qry_num)

        # true positives are in retrieved and relevant
        TP = len([r for r in retrieved if r in relevant])
        # false positives are in retrived but not relevant
        # FP = len([r for r in retrieved if r not in relevant])

        return TP/len(retrieved)

    def recall(self, sys, qrels):
        # precision at cutoff 10
        # using precision = relevant and retrieved / retrieved
        # that is P = TP / (TP + FP)
        # # exit()
        # for i in range(1,11):
        #     print(i, "==========================================================================")
        #     for rel in qrels.get_qrel(i).res_list:
        #         print(rel.doc_num)
        # exit()


        # get the query number
        qry_num = sys[0].qry_num

        # get all retreived documents for the query
        retrieved = self.get_retrieved_for_query(sys)

        # get all relevant documents for the query
        relevant = self.get_relevant_for_query(qrels, qry_num)
        # print(qry_num)
        # print(relevant)

        # true positives are in retrieved and relevant
        TP = len([r for r in retrieved if r in relevant])
        # false positives are in retrived but not relevant
        # FP = len([r for r in retrieved if r not in relevant])

        return TP / len(relevant)

    def ap(self, sys, qrels):
        # r: number of relevant docs for q
        # n: number of docs retrieved
        # P(k): p@k
        # Rel(k): 1 if retrieved doc@k is relevant otherwise 0

        # ap = 1/r * sum from k=1 to n: p@k * rel(k)
        qry_num = sys[0].qry_num

        # get all relevant documents for the query to calculate r

        relevant = self.get_relevant_for_query(qrels, qry_num)
        r = len(relevant)

        # get all retreived documents for the query to calculate n
        retrieved = self.get_retrieved_for_query(sys)
        n = len(retrieved)

        # summation loop
        sum = 0
        for k in range(1, n):
            sum += self.precision(sys[:k], qrels) * self.relevant(retrieved[k], relevant)
        return sum / r

    def r_precison(self, sys, qrels):
        # get the query number
        # r-precision is defined as r/R

        # where R is the total number of relevant documents
        # and r is the number of relevant documents out of the R fetched
        qry_num = sys[0].qry_num
        relevant = self.get_relevant_for_query(qrels, qry_num)
        R = len(relevant)

        retrieved = self.get_retrieved_for_query(sys[:R])

        r = len([r for r in retrieved if r in relevant])

        return r/R

    def relevant(self, doc, rels):
        # returns binary value; 1 if document is relevant to query, 0 otherwise
        return 1 if doc in rels else 0

    def nDCG(self, sys, qrels):
        # TODO: make sure it's top n
        # nDCG = rel_1 + sum_{i=2 to rank n} rel_i / log_2(i)

        # get the query number
        qry_num = sys[0].qry_num

        # get qrel for qry_num
        qrel = qrels.get_qrel(qry_num)

        # get all retreived documents for the query
        retrieved = self.get_retrieved_for_query(sys)

        # get all relevant documents for the query
        relevant = self.get_relevant_for_query(qrels, qry_num)

        # instansiate (actual) DCG to 0
        DCG = 0
        # instansiate (ideal) iDCG to 0
        iDCG = 0
        # loop through ranks 1 to n
        # TODO: check index
        for i in range(1, len(retrieved)):

            # get document relevance
            relevance = qrel.get_doc_rel(retrieved[i])
            if i == 1:
                iDCG += i

            # discount the gain if rank is not 1
            if i != 1:
                relevance /= math.log(i, 2)
                iDCG += math.log(i,2)

            # increment sum
            DCG += relevance
        nDCG = DCG / iDCG
        return nDCG


def N_t_c(term, t, c, docs, c_no):
    # t: boolean value
    # c: boolean value
    # there are four cases, N_0_0, N_0_1, N_1_0, N_1_1
    N_t_c = 0

    if c == 0:
        for doc_no, doc in enumerate(docs):
            if doc_no != c_no:
                if t == 0:
                    if term not in doc:
                        N_t_c += 1
                if t == 1:
                    if term in doc:
                        N_t_c += 1
    if c == 1:
        for doc_no, doc in enumerate(docs):
            if doc_no == c_no:
                if t == 0:
                    if term not in doc:
                        N_t_c += 1
                if t == 1:
                    if term in doc:
                        N_t_c += 1
    return N_t_c


def N_t(term, t, docs):
    # documents containing term
    # t: boolean value
    N_t = 0
    if t == 0:
        for doc in docs:
            if term not in doc:
                N_t += 1
    if t == 1:
        for doc in docs:
            if term in doc:
                N_t += 1
    return N_t


def N_c(c, N):
    # documents not in class c
    # c: boolean value
    if c == 0:
        return N - 1
    if c == 1:
        return 1


class PreProcessor:
    # this is an altered version of my code.py module from CW1

    def __init__(self):
        pass
    def read_in(self, filename):

        # if file to be read is stop words, return string
        if filename == "englishST.txt":
            return open(filename, 'r').read()
        # otherwise process different documents

        # holds different documents
        OT = []
        NT = []
        QR = []

        # iterate over input and split into documents
        lines = open(filename, 'r').readlines()
        for line in lines:
            # check the first letter to determine OT, NT, or QR
            if line[0] == 'O':
                OT.append(line)
            elif line[0] == 'N':
                NT.append(line)
            else:
                QR.append(line)
        # space = " "
        # print(len(OT), space, len(NT), space, len(QR))
        return OT, NT, QR

    def tokeniser(self, doc):
        tokens = []
        # iterate over lines in the document
        for line in doc:
            # split the line using regex, and iterate over the list
            # note the first item is discarded as we do not want "OT", "NT", or "Quran"
            for word in re.split(r'[^\w+\']+', line)[1:]:
                if word != "":
                    # append case folded word to token list
                    tokens.append(word.lower())
        # TODO: it looks like there are words like "'" in the token list
        # print(tokens)
        return tokens

    def sentence_tokeniser_and_stemmer(self, docs):
        # tokensises, stems, and stops
        sentences = []
        # read stop words from file
        stop_words = self.read_in("englishST.txt").split()
        for doc in docs:
            # iterate over lines in the document
            # get list of lists
            for line in doc:
                # split the line using regex, and iterate over the list
                # note the first item is discarded as we do not want "OT", "NT", or "Quran"
                sentence = []
                # for word in re.split(r'[^\w+\']+', line)[1:]:
                for word in re.split(r'[^\w+\']+', line):
                    if word != "":
                        # append case folded word to sentence
                        sentence.append(word.lower())
                # use porter stemmer to stem tokens if not present in stop words list
                sentence = [stem(t) for t in sentence if t not in stop_words]
                if sentence: sentences.append(sentence)
        return sentences

    def stopping_and_stemming(self, tokens):
        # performs stopping
        # makes a list of stop words and removes any such words in token stream

        # read stop words from file
        stop_words = self.read_in("englishST.txt").split()
        # use porter stemmer to stem tokens if not present in stop words list
        tokens = [stem(t) for t in tokens if t not in stop_words]
        return tokens

    def pre_process(self):
        print("preprocessing...")
        # get documents
        OT, NT, QR = self.read_in("train_and_dev.tsv")
        # pre-processed documents
        docs = []

        # for each document
        for doc in [OT, NT, QR]:
            # tokenise
            tokens = self.tokeniser(doc)
            # stop and stem
            docs.append(self.stopping_and_stemming(tokens))
        # print(self.mutual_information(docs))

        # get sentences from read in for LDA

        # corpus = self.sentence_tokeniser_and_stemmer([OT, NT, QR])
        self.LDA(docs)
        print("Complete.")

    def chi_squared(self):
        # x^2(D,t,c) = (A * B) / (C * D * E * F)

        chi_sqr = []
        print("Beginning chi-squared calculations...")
        for c_no, doc in enumerate(docs):
            used_terms = []
            for term in doc:
                if term not in used_terms:
                    used_terms.append(term)

                    N00 = self.N_t_c(term, 0, 0, docs, c_no)
                    N01 = self.N_t_c(term, 0, 1, docs, c_no)
                    N10 = self.N_t_c(term, 1, 0, docs, c_no)
                    N11 = self.N_t_c(term, 1, 1, docs, c_no)

                    A = N00 + N01 + N10 + N11
                    B = ((N11 * N00) - (N10 * N01)) ** 2

                    C = N11 + N01
                    D = N11 + N10
                    E = N10 + N00
                    F = N01 + N00

                    chi_sqr.append((A * B) / (C * D * E * F))
                else:
                    pass

        return chi_sqr

    def mutual_information(self, docs):
        # N - total number of documents
        # N_t_c - documents that do/don't contain e_t, e_c depending on the t, c values
        # N_t - documents that contain/don't contain e_t
        # N_c - documents that contain/don't contain e_c

        # list to hold the mutual information
        m_i = []
        N = len(docs)
        print("Beginning mutual information calculations...")
        for c_no, doc in enumerate(docs):
            used_terms = []
            for term in doc:

                formula = ""
                I = 0
                # print(term)
                if term not in used_terms:
                    used_terms.append(term)
                    for t in range(0, 2):
                        for c in range(0, 2):
                            # print("==============")
                            # print("term: ", term)
                            # print("N_", t,"_", c, ": ", self.N_t_c(term, t, c, docs, c_no))
                            # print("N_", t, ": ", self.N_t(term, t, docs))
                            # print("N_", c, ": ", self.N_c(c, N))
                            # print("==============")
                            formula += "N" + str(t) + str(c) + "/N log " + "(N N" + str(t) + str(c) + \
                                       ") /(N" + str(t) + "N" + str(c) + ")        +        "

                            Ntc = self.N_t_c(term, t, c, docs, c_no)
                            Nt = self.N_t(term, t, docs)
                            Nc = self.N_c(c, N)
                            # print(Ntc, Nt, Nc)
                            # need to check if the log will give an error
                            # if an exception is thrown, skip this
                            try:
                                s1 = Ntc / N
                                # print(s1)
                                s2 = (N*Ntc) / (Nt * Nc)
                                # print(s2)
                                I += s1 * math.log(s2, 2)
                                print(I)
                            except:
                                pass

            exit()
            m_i.append(I)
        return m_i

    def LDA(self, docs):
        # do I need OT, etc at the start of the line

        # Create a corpus from a list of texts
        common_dictionary = Dictionary(docs)

        common_corpus = [common_dictionary.doc2bow(text) for text in docs]

        # Train the model on the corpus.

        lda = LdaModel(common_corpus, num_topics=20, id2word=common_dictionary)
        # for i in range(lda.num_topics):
        #     for tt in lda.get_document_topics(common_corpus[i]):
        #         print(tt)
        #     lda.show_topic(i)
        #     print(lda.print_topic(i, 10))
        for a in lda.get_document_topics(common_corpus):
            print(a)

        print("hello")

        pass

class TextClassifier:
    def __init__(self, pp):
        self.pre_processor = pp
    def extract_BOW(self):
        # extracts bag of words from corpus

        # read in the data
        OT, NT, QR = self.pre_processor.read_in("train_and_dev.tsv")

        # get unique tokens
        unique_tokens = []

        docs = []
        for doc in [OT, NT, QR]:
            tokens = self.pre_processor.tokeniser(doc)
            docs.append(tokens)

            for t in tokens:
                if t not in unique_tokens:
                    unique_tokens.append(t)

        n_terms = len(unique_tokens)

        # add ID codes to tokens
        token_ids = self.get_token_ids(unique_tokens)

        # getting input verses
        sentences = self.pre_processor.sentence_tokeniser_and_stemmer([OT, NT, QR])
        print(len(sentences))


        # shuffle and split data into train and test
        train, test =self.shuffle_and_split(sentences, "9:1")

        # get X and y for each set
        pre_Xtrn, ytrn = self.form_data_and_labels(train, token_ids)
        pre_Xtst, ytst = self.form_data_and_labels(test, token_ids)

        # get lens for matrices
        pre_Xtrn_len = len(pre_Xtrn)
        pre_Xtst_len = len(pre_Xtst)

        # instansiate our sparse matrices
        Xtrn = dok_matrix((pre_Xtrn_len, n_terms))
        Xtst = dok_matrix((pre_Xtst_len, n_terms))

        # fill matrix
        # TODO: can make this a loop
        # (i, j) is number of times word j appears in document i
        for i in range(pre_Xtrn_len):
            for j in range(n_terms):
                cnt = self.word_count(pre_Xtrn[i], unique_tokens[j])
                if cnt != 0:
                    Xtrn[i, j] = cnt

        for i in range(pre_Xtst_len):
            for j in range(n_terms):
                cnt = self.word_count(pre_Xtst[i], unique_tokens[j])
                if cnt != 0:
                    Xtst[i, j] = cnt
        print(Xtrn)

        # now training the baseline SVC on train data
        baseline = SVC(C=1000)
        baseline.fit(Xtrn, ytrn)

        # make prediction on test data ("unseen data")
        ypred = baseline.predict(Xtst)


        ## test models
        bigger = SVC(C=1000)
        bigger.fit(Xtrn, ytrn)
        big_pred = bigger.predict(Xtst)


        ##############

        print("SCORE FOR PREDICTION!")
        # get baseline score as a fraction 0-1, 1 being the best, as normalise is set to True
        # multiply by 100 for percentage score
        print("baseline score")
        score = accuracy_score(y_true=ytst, y_pred=ypred, normalize=True) * 100
        print(score)

        print("bigger score")
        score = accuracy_score(y_true=ytst, y_pred=big_pred, normalize=True) * 100
        print(score)

        # print("the fit was successful!")
        print("Goodbye Cruel World")

    def shuffle_and_split(self, lst, ratio):
        # shuffles list of docs and splits into test and train
        # ratio of form n:1-n, meaning train=data[:n] test=data[n+1:] where n + (1-n) = 10

        # get multipliers from ratio
        trn_mult = int(ratio[0])
        tst_mult = int(ratio[2])

        # shuffle data
        random.shuffle(lst)

        # split
        # TODO: check that the split indices are correct
        return lst[:trn_mult], lst[tst_mult+1:]



    def form_data_and_labels(self, sentences, ids):
        X = []
        y = []
        for sentence in sentences:
            # the first word in the sentence defines the corpus
            # get the relevant id
            corpus = sentence[0]
            y.append(self.get_corpus_id(corpus))
            # the remainder of the sentence is to be added to X
            X.append(sentence[1:])
        # ignore warning for making a ragged nested sequence TODO for now
        np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
        return np.array(X), np.array(y)

    def get_token_ids(self, unique_tokens):
        # enumerate tokens and use that as ID
        token_ids = {}
        for id, tok in enumerate(unique_tokens):
            token_ids[tok] = id
        return token_ids


    def get_corpus_id(self, corpus):
        if corpus == "ot":
            return 0
        if corpus == "nt":
            return 1
        if corpus == "quran":
            return 2

    def word_count(self, doc, word):
        # returns occurnces of word in doc
        # count = 0
        # to speed up search, first ensure that word is present
        # trying to optimise this
        if word in doc:
            return doc.count(word)
        else:
            return 0

        # if word in doc:
        #     for w in doc:
        #         if w == word:
        #             count += 1
        #     return count
        # else:
        #     return count



if __name__ == "__main__":
    print("Hello World")


    # IR EVALUATION =-=-=-=-=-=-=-=-=-=-=-=-=-#
    ev = EVAL()                               #
    ev.read_in_sys_res("system_results.csv")  #
    ev.read_in_qrels("qrels.csv")             #
    ev.evaluate()                             #
    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-#

    # pp = PreProcessor()
    # pp.pre_process()

    # tc = TextClassifier(pp)
    # tc.extract_BOW()

    # TODO: fix AP so that it's p@k not just p
    # TODO: check r-precision
