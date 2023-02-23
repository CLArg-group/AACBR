
import os 

import networkx as nx
import matplotlib.pyplot as plt

from warnings import warn
import graphviz

def drawGraph(graph, gname: str, output_dir = None, engine = "networkx",
              node_formatter = None):
   '''Draws and saves a given graph in .png format'''

   if output_dir is None:
      graph_dir = os.path.join(os.getcwd(), 'graphs')
   else:
      graph_dir = output_dir
   if not os.path.isdir(graph_dir):
      os.makedirs(graph_dir)
   graph_name = os.path.join(graph_dir, '{}.png'.format(gname))
   
   match node_formatter:
      case function if callable(function):
         format_node = function
      case "full":
         format_node = lambda x: str((x.id, x.factors, x.outcome))
      case "id":
         format_node = lambda x: str((x.id, x.outcome))
      case "factors":
         format_node = lambda x: str((x.factors, x.outcome))
      case None:
         format_node = str
      case _:
         raise(RuntimeError(f"Unsupported {node_formatter=}"))
      
   match engine:
      case "networkx":
         if node_formatter is not None:
            warn(f"{node_formatter=} is not None, but is ignored by networkx engine.")
         nx.draw(graph, with_labels = True)
         plt.savefig(graph_name)
         plt.clf()
      case "graphviz":
         # Currently using graph given by nx, in giveGraph
         # perhaps this can be simplified later
         dot = graphviz.Digraph(gname, filename=gname, format='png')
         dot.attr(rankdir='BT')
         nodes, edges = graph
         for arg_node in nodes:
            # breakpoint()
            dot.node(str(hash(arg_node)), format_node(arg_node))
         for att_edge in edges:
            dot.edge(str(hash(att_edge[0])), str(hash(att_edge[1])))
         dot.render(directory=output_dir)
      case _:
         raise(Exception(f"Unsupported {engine=}"))
   pass
   
   
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


