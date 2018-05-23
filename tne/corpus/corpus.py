import random
import graph as deepwalk
import node2vec
import networkx as nx


class Corpus:

    def __init__(self, nxg):
        self.graph = nxg
        self.corpus = ""

    def graph2walks(self, method="", params=dict()):

        if method == "deepwalk":
            number_of_walks = params['number_of_walks']
            walk_length = params['walk_length']
            alpha = params['alpha']

            # Temporarily generate the edge list
            with open("./temp/graph.edgelist", 'w') as f:
                for line in nx.generate_edgelist(self.graph, data=False):
                    f.write("{}\n".format(line))

            dwg = deepwalk.load_edgelist("./temp/graph.edgelist", undirected=True)
            corpus = deepwalk.build_deepwalk_corpus(G=dwg, num_paths=number_of_walks,
                                                    path_length=walk_length,
                                                    alpha=alpha,
                                                    rand=random.Random(0))

        elif method == "node2vec":

            number_of_walks = params['number_of_walks']
            walk_length = params['walk_length']
            p = params['p']
            q = params['q']

            for edge in self.graph.edges():
                self.graph[edge[0]][edge[1]]['weight'] = 1
            G = node2vec.Graph(nx_G=self.graph, p=p, q=q, is_directed=False)
            G.preprocess_transition_probs()
            corpus = G.simulate_walks(num_walks=number_of_walks, walk_length=walk_length)

        else:
            raise ValueError("Invalid method name!")

        self.corpus = corpus

        return corpus

    def save(self, filename, with_title=False):

        with open(filename, 'w') as f:
            if with_title is True:
                f.write("{}\n".format(self.graph.number_of_nodes()))

            for walk in self.corpus:
                f.write(u"{}\n".format(u" ".join(v for v in walk)))


