# A dependency graph is a directed graph where each node represents a binary feature vector and each edge represents a dependency between two feature vectors.
# A dependency graph is used to resolve dependencies between feature vectors.
## dependencies:
# - performance limit
# - dependency paths -> pfadabhängige "Interaktion" = Konfiguration 
# - distanzabhängige Interaktion => Symmetriebrüche(Independency) bei Decoupling (gradual weakening of dependencies)
from dataclasses import dataclass
import xml.etree.ElementTree as ET
import ete3
from networkx import DiGraph
from util.draw_tree import DrawTree, DrawTreeWithNames
from util.xml_to_newick import xml_to_newick
from collections import defaultdict, deque
from numpy import hstack

@dataclass
class BinaryTree:
    value: str
    left: bool
    right: bool

class TreeNodes(DiGraph):
    def __init__(self, root, value=None, binary=False):
        
        self._value = value
        if binary:
            self._left = None
            self._right = None
        
        super().__init__(incoming_graph_data=None) # might add **attr later
        
        # construct tree
        self.root = id(root)
        self.construct_tree_from_xml(root)

    def construct_tree_from_xml(self, element, parent=None):
        node_id = id(element)
        self.add_node(node_id, value=element.tag)

        if parent is not None:
            self.add_edge(parent, node_id)

        children = list(element) # expands the element to its children

        if len(children) == 0 and element.text and element.text.strip():
            leaf_id = id(element.text.strip())
            self.add_node(leaf_id, value=element.text.strip())
            self.add_edge(node_id, leaf_id)
        
        for child in children:
            self.construct_tree_from_xml(child, parent=node_id)

        return self
    
    def __str__(self):
        return self._to_newick(self, self.root) + ';'
        
    def _to_newick(self, graph, node_id):
        node = graph.nodes[node_id]
        children = list(graph.successors(node_id))
        newick = ""

        if children:
            newick += "("
            for i, child_id in enumerate(children):
                newick += self._to_newick(graph, child_id)
                if i < len(children) - 1:
                    newick += ","
            newick += ")"

        newick += node['value']
        return newick
    
    def draw_graph_test(self):
        for i in range(10):
            try:
                DrawTreeWithNames(ete3.TreeNode(newick=self.__str__(), format=i))
                print("format", i, "works")
            except:
                print("wrong format")

    def draw_graph(self):
        DrawTreeWithNames(ete3.TreeNode(newick=self.__str__(), format=1))


class DependencyGraph:
    def __init__(self, df, xml_path=None, name=None, color=None):
        # load feature model
        self.xml = ET.parse(xml_path)
        #self.newick = xml_to_newick(ET.tostring(self.xml.getroot(), encoding='utf-8', method='xml'))
        #self.features = [feature.attrib['name'] for feature in self.feature_tree.getroot().findall('configurationOption')]
        self.features = set([config.find('name').text for config in self.xml.findall('./binaryOptions/configurationOption')]) # works for LLVM_energy Featuremodel structure only
        #
        #root = ET.tostring(self.xml.getroot(), encoding='utf-8', method='xml')
        #xml_root = ET.fromstring(root)
        self.feature_graph_from_xml = TreeNodes(root=self.xml.getroot())
        self.feature_graph_from_pd = self.create_balanced_tree(list(self.features) if self.features else df.columns.tolist())
        self.name = name
        self.color = color
        #self.feature_graph.add_features(self.features)
        #self.context_tree = self.create_permutations(self.features)
        #self.random_dependencies = self.generate_random_tree(list(self.features))
        
    def add_dependency(self, a, b, features):
        """
        Add a dependency: binary feature vector 'a' depends on 'b' via divide and conquer
        """
        self.dependency_graph = defaultdict(list)
        while features:
            self.dependency_graph[a].append(b)
            features.pop()

    def create_balanced_tree(self, columns):
    # Base case: if there are no columns, return None
        if len(columns) == 0:
            return None
        
        # Recursive case: take the middle column, make it the root, 
        # and do the same for the two halves of the remaining columns
        mid = len(columns) // 2
        root = BinaryTree(
            columns[mid],
            self.create_balanced_tree(columns[:mid]),
            self.create_balanced_tree(columns[mid+1:]))
    
        return root

    def constraint_graph(self, features):
        """
        apply dependencies to a feature graph
        """
        import random
        if not features:
            return None

        feature = random.choice(features)
        features.remove(feature)

        if not features:
            return TreeNodes(feature)

        left_child = self.constraint_graph(features[:])
        right_child = self.constraint_graph(features[:])

        return TreeNodes(feature, left_child, right_child)

    def resolve_dependencies(self, feature_vector):
        """
        Resolve all dependencies for the given binary feature vector
        """
        resolved = set()
        queue = deque([feature_vector])

        while queue:
            current = queue.popleft()

            if current not in resolved:
                resolved.add(current)

                for dependency in self.graph[current]:
                    queue.append(dependency)

        return resolved
    
    # for a corresponding feature graph (constraints) which features in a dependency graph (solution) are not satisfied
    def unsatisfied_dependencies(self, feature_vector):
        """
        Resolve all dependencies for the given binary feature vector
        """
        resolved = set()
        queue = deque([feature_vector])

        while queue:
            current = queue.popleft()

            if current not in resolved:
                resolved.add(current)

                for dependency in self.graph[current]:
                    queue.append(dependency)

        return resolved
    
    def return_hierarchy(self, tree, mode="BFS"):
        """
        create a hierarchy of the feature graph using numpy
        using a BFS or DFS approach
        """
        if mode == "BFS":
            queue = deque([tree])
            hierarchy = []
            while queue:
                current = queue.popleft()
                hierarchy.append(current.value)
                if current.left:
                    queue.append(current.left)
                if current.right:
                    queue.append(current.right)
            return hstack(hierarchy)
        elif mode == "DFS":
            stack = deque([tree])
            hierarchy = []
            while stack:
                current = stack.pop()
                hierarchy.append(current.value)
                if current.left:
                    stack.append(current.left)
                if current.right:
                    stack.append(current.right)
            return hstack(hierarchy)


    
    def create_permutations(self, features, path=None, remaining=None):
        if path is None:
            path = []
        if remaining is None:
            remaining = features[:]

        node = TreeNodes(path)

        if not remaining:
            return node

        for i in range(len(remaining)):
            new_remaining = remaining[:i] + remaining[i + 1:]
            new_path = path + [remaining[i]]
            child = self.create_permutations(features, new_path, new_remaining)
            node.add_child(child)

        return node
    
    def generate_random_tree(self, features):
        from random import shuffle

        if not features:
            return TreeNodes()
        
        middle = len(features) // 2
        root = TreeNodes(features[middle])

        shuffle(features[:middle])
        shuffle(features[middle + 1:])

        root.left = self.generate_random_tree(features[:middle])
        root.right = self.generate_random_tree(features[middle + 1:])

        return root
    
    def tree_to_newick(self, node: TreeNodes, branch_length=None):
        import random

        if not node:
            return ""
        elif not node.children:  # Leaf node
            return f"{node.value}:{branch_length}"
        else:
            children_newick = ",".join(self.tree_to_newick(child, random.random()) for child in node.children)
            return f"({children_newick}){node.value}:{branch_length}" if branch_length else f"({children_newick}){node.value}"

    
class DepencyMultiGraph(DependencyGraph):
    def __init__(self):
        self.graphs = []

    def register_graph(self, graph: DependencyGraph, name: str, color: hash, metrics: dict):
        graph.name = name
        graph.color = color
        graph.metrics = metrics
        self.graphs.append(graph)
