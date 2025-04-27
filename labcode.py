# labcodes.py

def dfs1():
    return """

# Implement Recursive Depth First Search Algorithm. Read the undirected 
# unweighted graph from a .csv file. 
import pandas as pd
import networkx as nx

df = pd.read_csv('dfs.csv')
G = nx.Graph()
for _, row in df.iterrows():
    G.add_edge(row['source'], row['target'])

def recursive_dfs(graph, node, visited):
    visited.append(node)
    for neighbor in graph.neighbors(node):
        if neighbor not in visited:
            recursive_dfs(graph, neighbor, visited)
    return visited

start_node = input("Enter starting node: ")
visited_nodes = recursive_dfs(G, start_node, [])

print("DFS Traversal:", visited_nodes)

    """

def dfs2():
    return """

    
# Implement Non-Recursive Depth First Search Algorithm. Read the 
# undirected unweighted graph from user.  
import networkx as nx
import matplotlib.pyplot as plt

def non_recursive_dfs(G, start):
    visited = []
    stack = [start]
    while stack:
        node = stack.pop()
        if node not in visited:
            visited.append(node)
            stack.extend(neighbor for neighbor in G.neighbors(node) if neighbor not in visited)
    return visited

G = nx.Graph()
num_edges = int(input("Number of edges: "))

for _ in range(num_edges):
    node1, node2 = input("Enter edge (node1 node2): ").split()
    G.add_edge(node1, node2)

# DFS and plot
start_node = input("Start node: ")
visited_nodes = non_recursive_dfs(G, start_node)

print("Visited:", visited_nodes)

nx.draw(G, with_labels=True, node_color='lightgreen', node_size=1000)
plt.show()
    
    
    """


def bfs1():
    return """

# Implement Breadth First Search Algorithm. Read the undirected 
# unweighted graph from user.  
import networkx as nx
import matplotlib.pyplot as plt

G = nx.Graph()
num_edges = int(input("Number of edges: "))
for _ in range(num_edges):
    u, v = input("Enter edge (node1 node2): ").split()
    G.add_edge(u, v)

def bfs(G, start):
    visited = []
    queue = [start]
    while queue:
        node = queue.pop(0)
        if node not in visited:
            visited.append(node)
            queue.extend(n for n in G.neighbors(node) if n not in visited)
    print("Visited:", visited)

start_node = input("Start node: ")
bfs(G, start_node)

nx.draw(G, with_labels=True, node_color='skyblue', node_size=1000)
plt.show()

    """

def befs1():
    return """

# Implement Best First Search Algorithm. Read the directed unweighted 
# graph and the heuristic values from user. 
import heapq
import networkx as nx

def best_first_search(graph, heuristic, start, goal):
    heap = []
    heapq.heappush(heap, (heuristic[start], [start]))  # Push start node

    while heap:
        cost, path = heapq.heappop(heap)
        node = path[0]
        if node == goal:
            return path[::-1]
        for neighbor in graph.neighbors(node):  
            if neighbor not in path:
                heapq.heappush(heap, (heuristic[neighbor], [neighbor] + path))
     
    return None

G = nx.DiGraph()
n = int(input("Enter number of edges: "))
for _ in range(n):
    u, v = input("Enter edge (from to): ").split()
    G.add_edge(u, v)

heuristic = {}
for node in G.nodes():
    heuristic[node] = int(input(f"Enter heuristic for {node}: "))


start = input("Enter start node: ")
goal = input("Enter goal node: ")
path = best_first_search(G, heuristic, start, goal)
print("Path:", path)
    """

def befs2():
    return """

# Implement Best First Search Algorithm. Read the undirected weighted 
# graph and the heuristic values from user. 
import heapq
import networkx as nx

def best_first_search(graph, heuristic, start, goal):
    heap = []
    heapq.heappush(heap, (heuristic[start], [start]))

    while heap:
        cost, path = heapq.heappop(heap)
        node = path[0]
        
        if node == goal:
            return path[::-1]
        
        for neighbor in graph.neighbors(node):
            if neighbor not in path:
                heapq.heappush(heap, (heuristic[neighbor], [neighbor] + path))
    
    return None

G = nx.Graph()
n = int(input("Enter number of edges: "))
for _ in range(n):
    u, v, w = input("Enter edge (from to weight): ").split()
    G.add_edge(u, v, weight=int(w))

heuristic = {}
for node in G.nodes():
    heuristic[node] = int(input(f"Enter heuristic for {node}: "))

start = input("Enter start node: ")
goal = input("Enter goal node: ")

path = best_first_search(G, heuristic, start, goal)

if path:
    print("Path:", path)
else:
    print("No path found")


    """


def befs3():
    return """
# Implement Best First Search Algorithm. Read the undirected unweighted 
# graph and the heuristic values from user.

import heapq
import networkx as nx

def best_first_search(graph, heuristic, start, goal):
    heap = []
    heapq.heappush(heap, (heuristic[start], [start]))

    while heap:
        _, path = heapq.heappop(heap)
        node = path[0]
        
        if node == goal:
            return path[::-1]
        
        for neighbor in graph.neighbors(node):
            if neighbor not in path:
                heapq.heappush(heap, (heuristic[neighbor], [neighbor] + path))
    
    return None

G = nx.Graph()
n = int(input("Enter number of edges: "))
for _ in range(n):
    u, v = input("Enter edge (from to): ").split()
    G.add_edge(u, v)

heuristic = {}
for node in G.nodes():
    heuristic[node] = int(input(f"Enter heuristic for {node}: "))

start = input("Enter start node: ")
goal = input("Enter goal node: ")

path = best_first_search(G, heuristic, start, goal)

if path:
    print("Path:", path)
else:
    print("No path found")


    """

def befs4():
    return """

# Implement Best First Search Algorithm. Read the directed weighted graph 
# and the heuristic values from user. 
import heapq
import networkx as nx

def best_first_search(graph, heuristic, start, goal):
    heap = []
    heapq.heappush(heap, (heuristic[start], [start]))

    while heap:
        cost, path = heapq.heappop(heap)
        node = path[0]
        
        if node == goal:
            return path[::-1]
        
        for neighbor in graph.neighbors(node):
            if neighbor not in path:
                heapq.heappush(heap, (heuristic[neighbor], [neighbor] + path))
    
    return None

G = nx.DiGraph()

n = int(input("Enter number of edges: "))
for _ in range(n):
    u, v, w = input("Enter edge (from to weight): ").split()
    G.add_edge(u, v, weight=int(w))

heuristic = {}
for node in G.nodes():
    heuristic[node] = int(input(f"Enter heuristic for {node}: "))

start = input("Enter start node: ")
goal = input("Enter goal node: ")

path = best_first_search(G, heuristic, start, goal)

if path:
    print("Path:", path)
else:
    print("No path found")

    """

def astar1():
    return """

    # Implement A* algorithm. Read directed weighted graph and heuristic 
# values from a .csv file. 

import pandas as pd
import networkx as nx
import heapq

def astar(graph, heuristic, start, goal):
    heap = []
    heapq.heappush(heap, (0 + heuristic[start], 0, [start]))

    while heap:
        f_cost, g_cost, path = heapq.heappop(heap)
        node = path[-1]
        if node == goal:
            return path
        for neighbor in graph.neighbors(node):
            if neighbor not in path:
                edge_weight = graph[node][neighbor]['weight']
                heapq.heappush(heap, (g_cost + edge_weight + heuristic[neighbor], g_cost + edge_weight, path + [neighbor]))
    return None

# Read graph from CSV
df = pd.read_csv('graphAstar.csv')  
heuristic = pd.read_csv('heuristicAstar.csv', index_col=0).to_dict()['heuristic']  

G = nx.DiGraph()
for _, r in df.iterrows():
    G.add_edge(r['source'], r['target'], weight=r['weight'])

start = input("Enter start node: ")
goal = input("Enter goal node: ")

path = astar(G, heuristic, start, goal)
print("Path:", path)


"""

def astar2():
    return """
# Implement A* algorithm. Read directed weighted graph and heuristic 
# values from user. 

import networkx as nx
import heapq

def astar(graph, heuristic, start, goal):
    heap = []
    heapq.heappush(heap, (0 + heuristic[start], 0, [start]))

    while heap:
        f_cost, g_cost, path = heapq.heappop(heap)
        node = path[-1]
        if node == goal:
            return path
        for neighbor in graph.neighbors(node):
            if neighbor not in path:
                edge_weight = graph[node][neighbor]['weight']
                heapq.heappush(heap, (g_cost + edge_weight + heuristic[neighbor], g_cost + edge_weight, path + [neighbor]))
    return None

G = nx.DiGraph()
n = int(input("Enter number of edges: "))
for _ in range(n):
    u, v, w = input("Enter edge (source target weight): ").split()
    G.add_edge(u, v, weight=int(w))

heuristic = {}
for node in G.nodes():
    heuristic[node] = int(input(f"Enter heuristic for {node}: "))

start = input("Enter start node: ")
goal = input("Enter goal node: ")

path = astar(G, heuristic, start, goal)
print("Path:", path)



    """


def astar3():
    return """
# Implement A* algorithm. Read undirected weighted graph and heuristic 
# values from a .csv file.
import pandas as pd
import networkx as nx
import heapq

def astar(graph, heuristic, start, goal):
    heap = []
    heapq.heappush(heap, (0 + heuristic[start], 0, [start]))

    while heap:
        f_cost, g_cost, path = heapq.heappop(heap)
        node = path[-1]
        if node == goal:
            return path
        for neighbor in graph.neighbors(node):
            if neighbor not in path:
                edge_weight = graph[node][neighbor]['weight']
                heapq.heappush(heap, (g_cost + edge_weight + heuristic[neighbor], g_cost + edge_weight, path + [neighbor]))
    return None

# Read graph from CSV
df = pd.read_csv('graphAstar.csv')  # Columns: source, target, weight
heuristic = pd.read_csv('heuristicAstar.csv', index_col=0).to_dict()['heuristic']  # Columns: node, heuristic

G = nx.Graph()
for _, r in df.iterrows():
    G.add_edge(r['source'], r['target'], weight=r['weight'])

start = input("Enter start node: ")
goal = input("Enter goal node: ")

path = astar(G, heuristic, start, goal)
print("Path:", path)

    """



def astar4():
    return """
# Implement A* algorithm. Read undirected weighted graph and heuristic 
# values from user.
import networkx as nx
import heapq

def astar(graph, heuristic, start, goal):
    heap = []
    heapq.heappush(heap, (0 + heuristic[start], 0, [start]))

    while heap:
        f_cost, g_cost, path = heapq.heappop(heap)
        node = path[-1]
        if node == goal:
            return path
        for neighbor in graph.neighbors(node):
            if neighbor not in path:
                edge_weight = graph[node][neighbor]['weight']
                heapq.heappush(heap, (g_cost + edge_weight + heuristic[neighbor], g_cost + edge_weight, path + [neighbor]))
    return None

G = nx.Graph()
n = int(input("Enter number of edges: "))
for _ in range(n):
    u, v, w = input("Enter edge (node1 node2 weight): ").split()
    G.add_edge(u, v, weight=int(w))

heuristic = {}
for node in G.nodes():
    heuristic[node] = int(input(f"Enter heuristic for {node}: "))

start = input("Enter start node: ")
goal = input("Enter goal node: ")

path = astar(G, heuristic, start, goal)
print("Path:", path)
# Implement A* algorithm. Read undirected weighted graph and heuristic 
# values from user.
import networkx as nx
import heapq

def astar(graph, heuristic, start, goal):
    heap = []
    heapq.heappush(heap, (0 + heuristic[start], 0, [start]))

    while heap:
        f_cost, g_cost, path = heapq.heappop(heap)
        node = path[-1]
        if node == goal:
            return path
        for neighbor in graph.neighbors(node):
            if neighbor not in path:
                edge_weight = graph[node][neighbor]['weight']
                heapq.heappush(heap, (g_cost + edge_weight + heuristic[neighbor], g_cost + edge_weight, path + [neighbor]))
    return None

G = nx.Graph()
n = int(input("Enter number of edges: "))
for _ in range(n):
    u, v, w = input("Enter edge (node1 node2 weight): ").split()
    G.add_edge(u, v, weight=int(w))

heuristic = {}
for node in G.nodes():
    heuristic[node] = int(input(f"Enter heuristic for {node}: "))

start = input("Enter start node: ")
goal = input("Enter goal node: ")

path = astar(G, heuristic, start, goal)
print("Path:", path)



    """

def fs1():
    return """
# Implement Fuzzy set operations – union, intersection and complement. 
# Demonstrate these operations with 3 fuzzy sets. 

# Fuzzy sets
A = {'x': 0.7, 'y': 0.3, 'z': 0.9}
B = {'x': 0.5, 'y': 0.8, 'z': 0.4}
C = {'x': 0.2, 'y': 0.6, 'z': 0.7}

# Union
union_abc = {key: max(A[key], B[key], C[key]) for key in A}
print("A ∪ B ∪ C:", union_abc)

# Intersection
intersection_abc = {key: min(A[key], B[key], C[key]) for key in A}
print("A ∩ B ∩ C:", intersection_abc)

# Complements
complement_a = {key: round(1 - A[key],2) for key in A}
complement_b = {key: round(1 - B[key],2) for key in B}
complement_c = {key: round(1 - C[key],2) for key in C}
print("¬A:", complement_a)
print("¬B:", complement_b)
print("¬C:", complement_c)


    """

def fs2():
    return """

# Fuzzy sets
A = {'x': 0.7, 'y': 0.3, 'z': 0.9}
B = {'x': 0.5, 'y': 0.8, 'z': 0.4}

# Union
union_ab = {key: max(A[key], B[key]) for key in A}
print("Union: ", union_ab)

# Intersection
intersection_ab = {key: min(A[key], B[key]) for key in A}
print("Intersection: ", intersection_ab)

# Complements
complement_a = {key: round(1 - A[key], 2) for key in A}
complement_b = {key: round(1 - B[key], 2) for key in B}
print("Complement of A: ", complement_a)
print("Complement of B: ", complement_b)

# Complement of Union
complement_union = {key: round(1 - union_ab[key], 2) for key in A}

# Intersection of Complements
intersection_complements = {key: min(complement_a[key], complement_b[key]) for key in A}

# De Morgan's verification
print("¬(A ∪ B):", complement_union)
print("(¬A) ∩ (¬B):", intersection_complements)

  """


def fs3():
    return """

# Fuzzy sets
A = {'x': 0.7, 'y': 0.3, 'z': 0.9}
B = {'x': 0.5, 'y': 0.8, 'z': 0.4}

# Union
union_ab = {key: max(A[key], B[key]) for key in A}
print("Union: ", union_ab)

# Intersection
intersection_ab = {key: min(A[key], B[key]) for key in A}
print("Intersection: ", intersection_ab)

# Complements
complement_a = {key: round(1 - A[key], 2) for key in A}
complement_b = {key: round(1 - B[key], 2) for key in B}
print("Complement of A: ", complement_a)
print("Complement of B: ", complement_b)

# Complement of Intersection
complement_intersection = {key: 1 - intersection_ab[key] for key in A}

# Union of Complements
union_complements = {key: max(complement_a[key], complement_b[key]) for key in A}

print("¬(A ∩ B):", complement_intersection)
print("(¬A) ∪ (¬B):", union_complements)



    """


def tp1():
    return """
import re
import nltk  
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from spellchecker import SpellChecker

# Download NLTK resources
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

# Function to process text
def process_text(file_path):
    # Read file
    with open(file_path, 'r') as file:
        text = file.read()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()

    # Convert to lowercase
    text = text.lower()

    # Tokenize text
    tokens = word_tokenize(text)

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    # Correct misspelled words
    spell = SpellChecker()
    tokens = [spell.correction(word) for word in tokens]

    return tokens

# File path
file_path = 'textDoc.txt'

# Process and print the cleaned text
processed_text = process_text(file_path)
print("Processed Text:", " ".join(processed_text))


    """


def tp2():
    return """  
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.util import ngrams

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def process_text(path):
    with open(path, 'r') as file:
        text = file.read()

    # Clean text: remove non-letters, extra spaces, lowercase
    text = re.sub(r'[^a-zA-Z\s]', '', text).lower()
    text = re.sub(r'\s+', ' ', text)

    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if w not in stop_words]

    stemmer = PorterStemmer()
    stems = [stemmer.stem(w) for w in tokens]

    lemmatizer = WordNetLemmatizer()
    lemmas = [lemmatizer.lemmatize(w) for w in tokens]

    trigrams = list(ngrams(lemmas, 3))

    return text, stems, lemmas, trigrams

# Run it
text, stems, lemmas, trigrams = process_text('textDoc.txt')

print("Text:", text[:200])
print("Stemmed:", stems[:20])
print("Lemmatized:", lemmas[:20])
print("Trigrams:", trigrams[:10])

    """

def ohe():
    return """
import re
import nltk
import numpy as np
from nltk.tokenize import word_tokenize

nltk.download('punkt')

# Read all text files
files = ['file1.txt', 'file2.txt', 'file3.txt']
text = ''
for file in files:
    with open(file, 'r', encoding='utf-8') as f:
        text += f.read() + ' '

 # Clean text: remove non-letters, extra spaces, lowercase
text = re.sub(r'[^a-zA-Z\s]', '', text)
text = re.sub(r'\s+', ' ', text).strip()

text=text.lower()

# Tokenize
tokens = word_tokenize(text)

# Create vocabulary
vocab = sorted(set(tokens))

# One-hot encoding
one_hot = np.zeros((len(tokens), len(vocab)), dtype=int)
for i, word in enumerate(tokens):
    one_hot[i, vocab.index(word)] = 1

# Output
print("Vocab size:", len(vocab))
print("First 5 vocab words:", vocab[:5])
print("One-hot matrix shape:", one_hot.shape)
print("\nFirst 5 words and their one-hot:")
for i in range(5):
    print(f"{tokens[i]}: {one_hot[i, :10]}...")
    """

def bow():
    return """
import re
from nltk.tokenize import word_tokenize
import nltk
from collections import Counter

# Download necessary NLTK data
nltk.download('punkt')

# Read and combine text from all files
files = ['file1.txt', 'file2.txt', 'file3.txt']
all_text = ''
for file in files:
    with open(file, 'r', encoding='utf-8') as f:
        all_text += f.read() + ' '

# Clean text: remove punctuation, numbers, and extra spaces, and convert to lowercase
clean_text = re.sub(r'[^\w\s]', '', all_text)  # Remove punctuation
clean_text = re.sub(r'\d+', '', clean_text)  # Remove numbers
clean_text = clean_text.lower().strip()  # Lowercase and remove leading/trailing spaces

# Tokenize the text into words
tokens = word_tokenize(clean_text)

# Create a Bag of Words (BoW) by counting word frequencies
bow = Counter(tokens)

# Print the results
print("Bag of Words (word frequencies):")
for word, count in bow.items():
    print(f"{word}: {count}")

    """




def tfidf():
    return """
import re
from sklearn.feature_extraction.text import TfidfVectorizer

# Read text from three files
files = ['file1.txt', 'file2.txt', 'file3.txt']
texts = []
for file in files:
    with open(file, 'r', encoding='utf-8') as f:
        texts.append(f.read())

# Clean the text (remove punctuation, numbers, and extra spaces)
clean_texts = []
for text in texts:
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = text.lower().strip()  # Lowercase and remove extra spaces
    clean_texts.append(text)

# Initialize TfidfVectorizer and compute the TF-IDF matrix
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(clean_texts)

# Print TF-IDF values for each term in each document
terms = vectorizer.get_feature_names_out()
for i, doc in enumerate(clean_texts):
    print(f"\nTF-IDF for Document {i+1}:")
    for idx, term in enumerate(terms):
        tfidf_value = tfidf_matrix[i, idx]
        if tfidf_value > 0:  # Print only non-zero TF-IDF values
            print(f"{term}: {tfidf_value:.4f}")

    """