import sys
sys.path.append("./ext/deepwalk/deepwalk")
sys.path.append("./ext/node2vec/src")
import time
from corpus.corpus import *
from ext.updatedgensim.models.word2vec import *


dataset = "citeseer"
networkx_graph_path = "./datasets/{}.gml".format(dataset)

method_name = "deepwalk"
embedding_size = 128
number_of_walks = 40
walk_length = 10
workers = 3


window_size = 10
number_of_topics = 65


g = nx.read_gml(networkx_graph_path)
number_of_nodes = g.number_of_nodes()

params = dict()
params['number_of_walks'] = number_of_walks
params['walk_length'] = walk_length
params['alpha'] = 0
params['p'] = 1.0
params['q'] = 1.0


corpus_file = "./temp/{}_n{}_l{}_k{}_{}.corpus".format(dataset, number_of_walks, walk_length, number_of_topics, method_name)
node_embedding_file = "./output/{}_n{}_l{}_w{}_k{}_{}_node.embedding".format(dataset, number_of_walks, walk_length, window_size, number_of_topics, method_name)
topic_embedding_file = "./output/{}_n{}_l{}_w{}_k{}_{}_topic.embedding".format(dataset, number_of_walks, walk_length, window_size, number_of_topics, method_name)
concatenated_embedding_file_max = "./output/{}_n{}_l{}_w{}_k{}_{}_final_max.embedding".format(dataset, number_of_walks, walk_length, window_size, number_of_topics, method_name)
concatenated_embedding_file_avg = "./output/{}_n{}_l{}_w{}_k{}_{}_final_avg.embedding".format(dataset, number_of_walks, walk_length, window_size, number_of_topics, method_name)
concatenated_embedding_file_min = "./output/{}_n{}_l{}_w{}_k{}_{}_final_min.embedding".format(dataset, number_of_walks, walk_length, window_size, number_of_topics, method_name)
lda_exe_path = "./lib/gibbslda/lda"

initial_time = time.time()
# Generate a corpus
corpus = Corpus(g)
walks = corpus.graph2walks(method=method_name, params=params)
# Save the corpus
corpus.save(filename=corpus_file, with_title=True)
print("-> The corpus was generated and saved in {:.2f} secs | {}".format((time.time()-initial_time), corpus_file))

initial_time = time.time()
# Extract the node embeddings
model = TNEWord2Vec(sentences=walks, size=embedding_size, window=window_size, sg=1, hs=1, workers=workers,
                    sample=0.001, min_count=0)
# Save the node embeddings
model.wv.save_word2vec_format(fname=node_embedding_file)
print("-> The node embeddings were generated and saved in {:.2f} secs | {}".format((time.time()-initial_time), node_embedding_file))

initial_time = time.time()
# Run GibbsLDA++
lda_corpus_path = corpus_file
lda_alpha = 50.0/float(number_of_topics)
cmd = "{} -est -alpha {} -beta {} -ntopics {} -niters {} -dfile {}".format(lda_exe_path, lda_alpha, 0.1, number_of_topics, 200, lda_corpus_path)
os.system(cmd)
print("-> The LDA algorithm run in {:.2f} secs".format(time.time()-initial_time))


# Define the paths for the files generated by GibbsLDA++
wordmapfile = "./temp/wordmap.txt"
tassignfile = "./temp/model-final.tassign"
lda_node_corpus = "./temp/lda_node.file"
lda_topic_corpus = "./temp/lda_topic.file"
phi_file = "./temp/model-final.phi"

initial_time = time.time()
# Convert node corpus to the corresponding topic corpus
utils.convert2topic_corpus(tassignfile, lda_topic_corpus)
# Generate sentences in the following form: (node, topic)
corpus.save(filename=lda_node_corpus, with_title=False)
# Construct the tuples (word, topic) with each word in the corpus and its corresponding topic assignment
combined_sentences = CombineSentences(lda_node_corpus, lda_topic_corpus)
# Extract the topic embeddings
model.train_topic(number_of_topics, combined_sentences)

# Save the topic embeddings
model.wv.save_word2vec_topic_format(fname=topic_embedding_file)
print("-> The topic embeddings were generated and saved in {:.2f} secs | {}".format((time.time()-initial_time), node_embedding_file))


# Generate the id2node dictionary
id2node = utils.generate_id2node(wordmapfile)
# Compute the corresponding topics for each node
initial_time = time.time()
node2topic_max = utils.find_max_topic_for_nodes(phi_file, id2node, number_of_nodes, number_of_topics)
# Concatenate the embeddings
utils.concatenate_embeddings(node_embedding_file=node_embedding_file,
                       topic_embedding_file=topic_embedding_file,
                       node2topic=node2topic_max,
                       output_filename=concatenated_embedding_file_max)
print("-> The final_max embeddings were generated and saved in {:.2f} secs | {}".format((time.time()-initial_time), concatenated_embedding_file_max))

# Concatenate the embeddings
initial_time = time.time()
utils.concatenate_embeddings_avg(node_embedding_file=node_embedding_file,
                           topic_embedding_file=topic_embedding_file,
                           phi_file=phi_file,
                           id2node=id2node,
                           output_filename=concatenated_embedding_file_avg)
print("-> The final_avg embeddings were generated and saved in {:.2f} secs | {}".format((time.time()-initial_time), concatenated_embedding_file_avg))

initial_time = time.time()
node2topic_min = utils.find_min_topic_for_nodes(phi_file, id2node, number_of_nodes, number_of_topics)
# Concatenate the embeddings

utils.concatenate_embeddings_min(node_embedding_file=node_embedding_file,
                           topic_embedding_file=topic_embedding_file,
                           node2topic=node2topic_min,
                           output_filename=concatenated_embedding_file_min)
print("-> The final_min embeddings were generated and saved in {:.2f} secs | {}".format((time.time()-initial_time), concatenated_embedding_file_min))
