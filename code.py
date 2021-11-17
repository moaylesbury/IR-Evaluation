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
                curr_sys = qry.sys_num
                # create new system
                sys = System(curr_sys)
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
                curr_query = res.qry_num
                # create new qrel
                qrel = Qrel(curr_query)
        # catch the last qrel
        qrels.add_qrel(qrel)
        # for res in qrels.get_qrel(4).res_list:
        #     res.pretty_print()
        return qrels




    def precision(self):
        pass


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

    def p_10(self):
        # precision at cutoff 10
        p_n = 10
        # fetch the system results and qrels
        # only want the top 10 results as cutoff at 10 and ranked linearly

        # TODO: change this, they don't need to pass the filename
        qrels = self.read_in_qrels("qrels.csv")
        sys_res = self.read_in_sys_res("system_results.csv")

        # TODO: first beginning with system1, need top 10 results for each query

        # IR systems are numbered 1-6
        for sys_num in range(1, 7):
            print("=-=-=-=-=-=-=-=-=-=-=-=-=- IR SYSTEM ", sys_num, " =-=-=-=-=-=-=-=-=-=-=-=-=-")
            # queries are numbered 1-10
            for qry_num in range(1, 11):
                print("~~~~~~~~~~~~~ system:1 , query:", qry_num, " ~~~~~~~~~~~~~")
                for query in sys_res.get_system(sys_num).get_query(qry_num)[:p_n]:
                    # top 10 results of query qry_num from system sys_num
                    query.pretty_print()





if __name__ == "__main__":
    print("Hello World")
    ev = EVAL()
    # ev.read_in_sys_res("system_results.csv")
    # ev.read_in_qrels("qrels.csv")
    ev.p_10()