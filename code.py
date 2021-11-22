import math
import re
from stemming.porter2 import stem

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
    # list of Qrels; there qill be 6 as there are 6 queries
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


    def get_top_n(self, metric, n):
        # gets top n results

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
                if sys_num == 1 and qry_num == 10:
                    for a in sys_res.get_system(1).get_query(10):
                        a.pretty_print()
                    print("Yes")
                # print("~~~~~~~~~~~~~ system:1 , query:", qry_num, " ~~~~~~~~~~~~~")
                a = self.precision(sys_res.get_system(sys_num).get_query(qry_num)[:n], qrels)
                b = self.recall(sys_res.get_system(sys_num).get_query(qry_num)[:n], qrels)
                c = self.ap(sys_res.get_system(sys_num).get_query(qry_num)[:n], qrels)
                d = self.r_precison(sys_res.get_system(sys_num).get_query(qry_num)[:n], qrels)
                e = self.nDCG(sys_res.get_system(sys_num).get_query(qry_num)[:n], qrels)
                f = self.nDCG(sys_res.get_system(sys_num).get_query(qry_num)[:n], qrels)

                for i in [sys_num, qry_num, a, b, c, d, e, f]:
                    output += str(i)[:4]
                    output += " "
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
        FP = len([r for r in retrieved if r not in relevant])


        return TP/(TP + FP)

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
        FP = len([r for r in retrieved if r not in relevant])

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
        for k in range(0, n):
            # TODO: think I need to change this to precision at k rather than just precision
            sum += self.precision(sys, qrels) * self.relevant(retrieved[k], relevant)
        return sum / r

    def r_precison(self, sys, qrels):
        # TODO: check this
        # get the query number
        qry_num = sys[0].qry_num
        relevant = self.get_relevant_for_query(qrels, qry_num)
        r = len(relevant)

        retrieved = self.get_retrieved_for_query(sys)

        relevant_and_retrieved = len([r for r in retrieved if r in relevant])

        return relevant_and_retrieved/r

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

        # for each document
        for doc in [OT, NT, QR]:
            # tokenise
            tokens = self.tokeniser(doc)
            # stop and stem
            doc = self.stopping_and_stemming(tokens)
        print(self.mutual_information([OT,NT,QR]))
        print("Complete.")

    def chi_squared(self):
        pass

    def mutual_information(self, docs):
        # N - total number of documents
        # N_t_c - documents that do/don't contain e_t, e_c depending on the t, c values
        # N_t - documents that contain/don't contain e_t
        # N_c - documents that contain/don't contain e_c

        # list to hold the mutual information
        m_i = []

        N = len(docs)
        c_no = 0 # class number
        for t in range(0,2):
            for c in range(0,2):
                s1 = self.N_t_c(t, c, docs, c_no) / N
                s2 = s1 / self.N_t(t, docs) * self.N_c(c, N)
                m_i.append(s1 * math.log(s2, 2))
        return m_i

    def N_t_c(self, t, c, docs, c_no):
        # t: boolean value
        # c: boolean value
        # there are four cases, N_0_0, N_0_1, N_1_0, N_1_1
        N_t_c = 0

        if c == 0:
            for doc_no, doc in enumerate(docs):
                if doc_no == c_no:
                    if t == 0:
                        if t not in doc:
                            N_t_c += 1
                    if t == 1:
                        if t in doc:
                            N_t_c += 1
        if c == 1:
            for doc_no, doc in enumerate(docs):
                if doc_no == c_no:
                    if t == 0:
                        if t not in doc:
                            N_t_c += 1
                    if t == 1:
                        if t in doc:
                            N_t_c += 1
        return N_t_c


    def N_t(self, t, docs):
        # documents containing t
        # t: boolean value
        N_t = 0
        if t == 0:
            for doc in docs:
                if t not in doc:
                    N_t += 1
        if t == 1:
            for doc in docs:
                if t in doc:
                    N_t += 1
        return N_t


    def N_c(self, c, N):
        # documents not in class c
        # c: boolean value
        if c == 0:
            return N - 1
        if c == 1:
            return 1


if __name__ == "__main__":
    print("Hello World")
    ev = EVAL()
    # ev.read_in_sys_res("system_results.csv")
    # ev.read_in_qrels("qrels.csv")

    # to calculate p@10:
    # ev.get_top_n('p', 10)

    pp = PreProcessor()
    pp.pre_process()



    # TODO: fix AP so that it's p@k not just p
    # TODO: check r-precision
