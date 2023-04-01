"""
a python implementation of https://github.com/IoannisPapageorgiou/Bayesian-Context-Trees
"""

from dataclasses import dataclass, field
import ete3
import numpy as np
from typing import List, Any, Mapping
from math import log2

m : int = 0
D : int = 0

@dataclass
class Node:
    id : int
    name : str
    context : np.ndarray
    occurence : List = field(default_factory=list)
    le : np.double = 0
    lw : np.double = 0
    lm : np.ndarray((1,m), np.double) = 0
    matrix : np.ndarray((2,D), np.short) = 0

    def __hash__(self):
        return hash(self.context)

    def __dir__(self):
        return ['id', 'name', 'context', 'occurence', 'le', 'lw', 'lm', 'matrix']


class Tree(ete3.TreeNode):
    def __init__(self, m=0, D=0):
        self.m = m
        self.D = D
        super().__init__(newick=None, format=0, dist=None, support=None,
                 name=None, quoted_node_names=False)
        # construct tree
        self.construct(self.get_tree_root())
        
        #node = Node(id=n, name='', context=str(j)*i, occurence=[0]*m, le=0, lw=0, lm=[0], matrix=np.matrix([[],[]]))
        # add features
        for node in self.traverse():
            node.add_features(id=node.up.children.index(node) if node.up else '',
                occurence=[0]*m, 
                le=0, 
                lw=0, 
                lm=np.zeros((1,m), np.double), 
                matrix=np.zeros((2,D), np.short))
        self.insert_context()
    
    def insert_context(self):
        for node in self.traverse():
            if node.is_root():
                node.name += ''
            else:
                print(node.up.id, '', node.id)
                node.name = node.up.name + str(node.id)
                node.context = node.name

    def construct(self, node, rec_D=0):
        """
        recursively construct tree
        m..size of the Alphabet ~ children per subtree
        D..depth ~ #level
        """    
        if rec_D < self.D:
            for i in range(self.m):
                node.add_child()
            for child in node.children:
                self.construct(child, rec_D+1)
        else:
            return

    def update(self, s: np.short, ct: np.ndarray):
        node = self.get_tree_root() # start at root
        self.occur(node, s)

        for j in range(self.D):
            candidate = 1
            if node.children:
                node = node.children[node.children.name[j]]
                self.occur(node, s)
            elif False:
                pass
            else:
                ct2 = ct.copy()
                ch = 0
                for k in range(D-j):
                    self.insert(ct2, ch)
                    self.occur(candidate, s)


    def occur(self, node, s: np.short):
        node.occurence[s] += 1
        M = 0
        for i in range(self.m):
            M += node.occurence[i]

        node.le = 1.0 * node.le + log2(1.0*node.occurence[s] - 0.5) - log2(0.5*self.m + 1.0*M - 1.0)

    def insert(self, ct2, ch):
        # transpose
        pass

    def getLvl(self, node):
        lvl = 0
        while not node.is_root():
            lvl += 1
            node = node.up
        return lvl

    def getNodesInLvlOrder(self):
        return [node for node in self.traverse()]

    def getNodesInLvl(self, lvl):
        nodes = []
        for node in self.traverse():
            if self.getLvl(node) == lvl:
                nodes.append(node)
        print(nodes)
        nodes = []
        nodes = [node for node in self.traverse() if self.getLvl(node) == lvl]
        return nodes

    def getAllNodesByLvl(self):
        nodes = self.getNodesInLvlOrder()
        nodes_by_lvl = []
        while nodes:
            lvl = self.getLvl(nodes[0])
            nodes_in_lvl = self.getNodesInLvl(lvl)
            nodes_by_lvl.append(nodes_in_lvl)
            for node in nodes_in_lvl:
                nodes.remove(node)
            
        return nodes_by_lvl

    def getNodesByLvl(self, lvl):
        all_nodes_by_lvl = self.getAllNodesByLvl()
        return all_nodes_by_lvl[lvl]
    
    def getNodeContext(self, node):
        lvl = self.getLvl(node)
        return self.getAllNodesByLvl()[lvl:]

    def getLvlContext(self, lvl):
        return self.getAllNodesByLvl()[lvl:]

    def isLeaf(self, node):
        return node.is_leaf()
    
    def countLeaves(self, *node):
        if node.is_root():
            return len(node)
        else:
            leaves = 0
            for node in self.traverse():
                if node.is_leaf():
                    leaves += 1
            return leaves


    def isChild(self, candidate, node):
        if candidate in node.children: 
            return True
        else:
            return False 

    def addNode(self):
        pass