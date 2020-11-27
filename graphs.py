
import os 

import networkx as nx
import matplotlib.pyplot as plt


def drawGraph(graph, gname: str):
	'''Draws and saves a given graph in .png format'''

	graph_dir = os.path.join(os.getcwd(), 'graphs')
	if not os.path.isdir(graph_dir):
		os.makedirs(graph_dir)
	graph_name = os.path.join(graph_dir, '{}.png'.format(gname))
	nx.draw(graph, with_labels = True)
	plt.savefig(graph_name)
	plt.clf()
	
	
def getPath(graph, path) -> list:
	'''Returns a path given a graph and an initial sink node'''

	sink = path[0]
	leaf = True
	for node in graph.nodes():
		if (node, sink) in graph.edges():
			leaf = False
			break
	if leaf:
		return path
	else:
		path.insert(0, node)
		graph.remove_node(sink)
		return getPath(graph, path)
	
	
def giveGraph(nodes, edges = None):
	'''Returns a digraph given nodes and optionally edges'''

	graph = nx.DiGraph()
	if edges:
		graph.add_nodes_from(nodes)
		graph.add_edges_from(edges)
	else:
		path = nx.path_graph(nodes)
		graph.add_edges_from(path.edges())
		
	return graph


