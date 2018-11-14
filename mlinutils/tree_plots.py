from IPython.display import display, Image
from sklearn import tree
import pygraphviz as pgv
import pydot
import io, re


def apply_styles(graph, styles):
    graph.graph_attr.update(
        ('graph' in styles and styles['graph']) or {}
    )
    graph.node_attr.update(
        ('nodes' in styles and styles['nodes']) or {}
    )
    graph.edge_attr.update(
        ('edges' in styles and styles['edges']) or {}
    )
    return graph


styles = {
    'graph': {
        'label': 'Decision Tree representation',
        'fontsize': '20',
        'fontcolor': '#337ab7',
    },
    'nodes': {
        'fontname': 'Helvetica',
        'shape': 'ellipse',
        'color': '#337ab7',
        'style': 'filled',
        'fillcolor': '#4C8BE2',
    },
    'edges': {
        'color': '#ABC1E2',
        'arrowhead': 'none',
        'fontsize': '20',
        'fontcolor': '#337ab7',
    }
}


def tree_viz(tree_model, feature_names, path=[]):
    dot_data = io.StringIO()
    tree.export_graphviz(tree_model, out_file=dot_data, impurity=False, 
                         feature_names=feature_names,rounded=True)
    graph_string = dot_data.getvalue()
    graph_string = re.sub(r'samples = [0-9]+','',graph_string)
    graph_string = re.sub(r'value = ','Value: ',graph_string)
    grp1 = pgv.AGraph(graph_string)
    for i in range(len(path)):
        n=grp1.get_node(i)
        if (path[i]==1):
            n.attr['color']='red'
    apply_styles(grp1, styles)
    grp = grp1.to_string()
    (graph,) = pydot.graph_from_dot_data(grp)
    display(Image(graph.create_png()))


def path_viz(tree_model, feature_names, sample, predict_fn):
    prediction = predict_fn(sample.values.reshape(1,-1))
    print(f"The prediction that the model made is {prediction:,.2f} and here's the path that led to it:")
    path = tree_model.decision_path(sample.values.reshape(1,-1)).todense().tolist()[0]
    tree_viz(tree_model, feature_names, path=path)

    
def tree_to_code(tree, feature_names):
    from sklearn.tree import _tree

    '''
    Outputs a decision tree model as a Python function.
    Source: https://www.kdnuggets.com/2017/05/simplifying-decision-tree-interpretation-decision-rules-python.html
    
    Parameters:
    -----------
    tree: decision tree model
        The decision tree to represent as a function
    feature_names: list
        The feature names of the dataset used for building the decision tree
    '''

    tree_ = tree.tree_
    feature_name = [feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!" for i in tree_.feature]

    def recurse(node, depth):
        indent = "  " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            print("{}if {} <= {}:".format(indent, name, threshold))
            recurse(tree_.children_left[node], depth + 1)
            print( "{}else:  # if {} > {}".format(indent, name, threshold))
            recurse(tree_.children_right[node], depth + 1)
        else:
            print("{}return {}".format(indent, tree_.value[node]))

    recurse(0, 1)
