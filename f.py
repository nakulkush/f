# f.py

def fname():
    return """
    lib,csv,dfs1,dfs2,bfs1,befs1,befs2,befs3,befs4,astar1,astar2,astar3,astar4,fs1,fs2,fs3,tp1,tp2,ohe1,bow1,tfidf1,mlp1,mlp2,mlp3,mlp4,mlp5,nim1,nim2
    """

def lib():
    return """
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import heapq



tp1
import re
import nltk  
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from spellchecker import SpellChecker

# Download NLTK resources
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')


tp2
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.util import ngrams

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')


ohe
import re
import nltk
import numpy as np
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('punkt_tab')


bow
import re
from nltk.tokenize import word_tokenize
import nltk
from collections import Counter

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('punkt_tab')


tfidf
import re
from sklearn.feature_extraction.text import TfidfVectorizer


mlp
import numpy as np
import itertools



    """


def csv():
    return """
source,target,weight
a b 1
a c 4
b c 2
b d 3
c e 5
d g 4
d f 2
e g 3
f g 1


heuristic,heuristic
a 5
b 6
c 4
d 3
e 3
g 0
f 1

start a
goal g

    """

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
nltk.download('punkt_tab')
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

def ohe1():
    return """
import re
import nltk
import numpy as np
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('punkt_tab')

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

def bow1():
    return """
import re
from nltk.tokenize import word_tokenize
import nltk
from collections import Counter

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('punkt_tab')

# Read and combine text from all files
files = ['file1.txt', 'file2.txt', 'file3.txt']
text = ''
for file in files:
    with open(file, 'r', encoding='utf-8') as f:
         text += f.read() + ' '

text = re.sub(r'[^a-zA-Z\s]', '', text)
text = re.sub(r'\s+', ' ', text).strip()

text=text.lower()
# Tokenize the text into words
tokens = word_tokenize(text)

# Create a Bag of Words (BoW) by counting word frequencies
bow = Counter(tokens)

# Print the results
print("Bag of Words (word frequencies):")
for word, count in bow.items():
    print(f"{word}: {count}")

    """




def tfidf1():
    return """
import re
from sklearn.feature_extraction.text import TfidfVectorizer

# List of file names
files = ['file1.txt', 'file2.txt', 'file3.txt']

# Read and clean text from files
texts = []
for file in files:
    with open(file, 'r', encoding='utf-8') as f:
        text = f.read()
        text = re.sub(r'[^a-zA-Z\s]', '', text)   # Keep only letters and spaces
        text = re.sub(r'\s+', ' ', text).strip()  # Replace multiple spaces with single space
        text = text.lower()                       # Convert to lowercase
        texts.append(text)

# Initialize TfidfVectorizer and fit texts
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(texts)

# Get all words (features)
words = vectorizer.get_feature_names_out()

# Print TF-IDF values for each document
for i in range(len(texts)):
    print(f"\nTF-IDF for Document {i+1}:")
    for j in range(len(words)):
        value = tfidf_matrix[i, j]
        if value > 0:
            print(f"{words[j]}: {value:.4f}")

    """



def mlp1():
    return """


import numpy as np
import itertools

# Step activation
def step(x):
    return (x >= 0).astype(int)

# Forward pass
def forward(X, W1, b1, W2, b2, W3, b3):
    h1 = step(np.dot(X, W1) + b1)
    h2 = step(np.dot(h1, W2) + b2)
    out = step(np.dot(h2, W3) + b3)
    return out

# Inputs & expected outputs
N = int(input("Enter number of binary inputs: "))
X = np.array(list(itertools.product([0, 1], repeat=N)))
y = np.array([int(input(f"Expected output for {list(x)}: ")) for x in X]).reshape(-1, 1)

found = False
steps = 0

while not found:
    steps += 1

    # Random weights and biases
    W1 = np.random.uniform(-2, 2, (N, 2))
    W2 = np.random.uniform(-2, 2, (2, 2))
    W3 = np.random.uniform(-2, 2, (2, 1))
    b1 = np.random.uniform(-3, 3)
    b2 = np.random.uniform(-3, 3)
    b3 = np.random.uniform(-3, 3)

    # Prediction
    y_pred = forward(X, W1, b1, W2, b2, W3, b3)

    if np.array_equal(y_pred, y):
        found = True

# Results
print(f"\n✅ Solution found in {steps} steps.")
print(f"\nW1:\n{W1}\nW2:\n{W2}\nW3:\n{W3}")
print(f"Biases: b1 = {b1}, b2 = {b2}, b3 = {b3}")


    """



def mlp2():
    return """

import numpy as np
import itertools

# Step activation function
def step(x):
    return (x >= 0).astype(int)

# Forward pass function
def forward(X, W1, b1, W2, b2):
    hidden_layer = step(np.dot(X, W1) + b1)  # Hidden layer
    output_layer = step(np.dot(hidden_layer, W2) + b2)  # Output layer
    return output_layer

# Inputs & expected outputs
X = np.array(list(itertools.product([0, 1], repeat=4)))  # 4 binary inputs
print("Enter the expected outputs for each input combination (two binary outputs):")
y = np.array([list(map(int, input(f"Expected output for {list(x)}: ").split())) for x in X])  # Expected outputs for 2 outputs

found = False
steps = 0

while not found:
    steps += 1

    # Random weight and bias initialization
    W1 = np.random.uniform(-2, 2, (4, 2))  # Weights from input layer to hidden layer
    W2 = np.random.uniform(-2, 2, (2, 2))  # Weights from hidden layer to output layer
    b1 = np.random.uniform(-3, 3, 2)  # Biases for hidden layer
    b2 = np.random.uniform(-3, 3, 2)  # Biases for output layer

    # Perform forward pass
    y_pred = forward(X, W1, b1, W2, b2)

    # Check if predictions match the expected outputs
    if np.array_equal(y_pred, y):
        found = True

# Display results
print(f"\n✅ Solution found in {steps} steps.")
print(f"\nW1 (Input to Hidden Layer):\n{W1}")
print(f"W2 (Hidden to Output Layer):\n{W2}")
print(f"Biases: b1 = {b1}, b2 = {b2}")


    """







def mlp3():
    return """

import numpy as np
import itertools

# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of sigmoid
def sigmoid_derivative(x):
    return x * (1 - x)

# Forward pass
def forward_pass(X, W1, b1, W2, b2, W3, b3):
    z1 = np.dot(X, W1) + b1
    a1 = sigmoid(z1)

    z2 = np.dot(a1, W2) + b2
    a2 = sigmoid(z2)

    z3 = np.dot(a2, W3) + b3
    output = sigmoid(z3)

    return a1, a2, output

# Input: number of binary inputs
num_inputs = int(input("Enter number of binary inputs (N): "))

# Generate all possible binary input combinations (2^N combinations)
X = np.array(list(itertools.product([0, 1], repeat=num_inputs)))

# Input: expected outputs for each input combination
print("\nEnter expected output (0 or 1) for each input combination:")
y_true = np.array([int(input(f"Input {list(x)}: ")) for x in X]).reshape(-1, 1)

# Initialize weights and biases randomly
np.random.seed(42)  # For reproducibility
W1 = np.random.randn(num_inputs, 4)
b1 = np.random.randn(4)
W2 = np.random.randn(4, 3)
b2 = np.random.randn(3)
W3 = np.random.randn(3, 1)
b3 = np.random.randn(1)

# Training parameters
epochs = 10000
learning_rate = 0.5

# Training loop
for epoch in range(epochs):
    # Forward pass
    a1, a2, output = forward_pass(X, W1, b1, W2, b2, W3, b3)

    # Compute error
    error = y_true - output

    # Backward pass
    d_output = error * sigmoid_derivative(output)

    error_hidden2 = d_output.dot(W3.T)
    d_hidden2 = error_hidden2 * sigmoid_derivative(a2)

    error_hidden1 = d_hidden2.dot(W2.T)
    d_hidden1 = error_hidden1 * sigmoid_derivative(a1)

    # Update weights and biases
    W3 += a2.T.dot(d_output) * learning_rate
    b3 += np.sum(d_output, axis=0) * learning_rate

    W2 += a1.T.dot(d_hidden2) * learning_rate
    b2 += np.sum(d_hidden2, axis=0) * learning_rate

    W1 += X.T.dot(d_hidden1) * learning_rate
    b1 += np.sum(d_hidden1, axis=0) * learning_rate

    # (Optional) Print loss every 1000 epochs
    if (epoch + 1) % 1000 == 0:
        loss = np.mean(np.square(error))
        print(f"Epoch {epoch+1}: Loss = {loss:.6f}")

# Final predictions
_, _, final_output = forward_pass(X, W1, b1, W2, b2, W3, b3)

# Display input-output mapping
print("\nInput -> Expected Output -> Predicted Output")
for xi, yi, oi in zip(X, y_true, final_output):
    print(f"{list(xi)} -> {yi[0]} -> {oi[0]:.4f}")


    """


def mlp4():
    return """


import numpy as np
import itertools

# ReLU activation function
def relu(x):
    return np.maximum(0, x)

# Derivative of ReLU
def relu_derivative(x):
    return np.where(x > 0, 1, 0)

# Forward pass
def forward_pass(X, weights1, bias1, weights2, bias2, weights3, bias3):
    z1 = np.dot(X, weights1) + bias1
    activation1 = relu(z1)

    z2 = np.dot(activation1, weights2) + bias2
    activation2 = relu(z2)

    z3 = np.dot(activation2, weights3) + bias3
    output = relu(z3)

    return z1, activation1, z2, activation2, z3, output

# Input: number of binary inputs
num_inputs = int(input("Enter number of binary inputs (N): "))

# Generate all possible binary input combinations (2^N combinations)
X = np.array(list(itertools.product([0, 1], repeat=num_inputs)))

# Input: expected outputs for each input combination
print("\nEnter expected output (0 or 1) for each input combination:")
y_true = np.array([int(input(f"Input {list(x)}: ")) for x in X]).reshape(-1, 1)

# Initialize weights and biases randomly
np.random.seed(42)  # For reproducibility
weights1 = np.random.randn(num_inputs, 4)
bias1 = np.random.randn(4)
weights2 = np.random.randn(4, 3)
bias2 = np.random.randn(3)
weights3 = np.random.randn(3, 1)
bias3 = np.random.randn(1)

# Training parameters
epochs = 10000
learning_rate = 0.01  # Note: Lower learning rate for ReLU to avoid instability

# Training loop
for epoch in range(epochs):
    # Forward pass
    z1, activation1, z2, activation2, z3, output = forward_pass(X, weights1, bias1, weights2, bias2, weights3, bias3)

    # Compute error
    output_error = y_true - output

    # Backward pass
    delta_output = output_error * relu_derivative(z3)

    error_hidden2 = delta_output.dot(weights3.T)
    delta_hidden2 = error_hidden2 * relu_derivative(z2)

    error_hidden1 = delta_hidden2.dot(weights2.T)
    delta_hidden1 = error_hidden1 * relu_derivative(z1)

    # Update weights and biases
    weights3 += activation2.T.dot(delta_output) * learning_rate
    bias3 += np.sum(delta_output, axis=0) * learning_rate

    weights2 += activation1.T.dot(delta_hidden2) * learning_rate
    bias2 += np.sum(delta_hidden2, axis=0) * learning_rate

    weights1 += X.T.dot(delta_hidden1) * learning_rate
    bias1 += np.sum(delta_hidden1, axis=0) * learning_rate

    # (Optional) Print loss every 1000 epochs
    if (epoch + 1) % 1000 == 0:
        loss = np.mean(np.square(output_error))
        print(f"Epoch {epoch+1}: Loss = {loss:.6f}")

# Final predictions after training
_, _, _, _, _, final_output = forward_pass(X, weights1, bias1, weights2, bias2, weights3, bias3)

# Display input-output mapping
print("\nInput -> Expected Output -> Predicted Output")
for input_vec, true_label, prediction in zip(X, y_true, final_output):
    print(f"{list(input_vec)} -> {true_label[0]} -> {prediction[0]:.4f}")


    """




def mlp5():
    return """


import numpy as np
import itertools

# Tanh activation function
def tanh(x):
    return np.tanh(x)

# Derivative of Tanh
def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2

# Forward pass
def forward_pass(X, weights1, bias1, weights2, bias2, weights3, bias3):
    z1 = np.dot(X, weights1) + bias1
    activation1 = tanh(z1)

    z2 = np.dot(activation1, weights2) + bias2
    activation2 = tanh(z2)

    z3 = np.dot(activation2, weights3) + bias3
    output = tanh(z3)

    return z1, activation1, z2, activation2, z3, output

# Input: number of binary inputs
num_inputs = int(input("Enter number of binary inputs (N): "))

# Generate all possible binary input combinations (2^N combinations)
X = np.array(list(itertools.product([0, 1], repeat=num_inputs)))

# Input: expected outputs for each input combination
print("\nEnter expected output (0 or 1) for each input combination:")
y_true = np.array([int(input(f"Input {list(x)}: ")) for x in X]).reshape(-1, 1)

# Initialize weights and biases randomly
np.random.seed(42)  # For reproducibility
weights1 = np.random.randn(num_inputs, 4)
bias1 = np.random.randn(4)

weights2 = np.random.randn(4, 3)
bias2 = np.random.randn(3)

weights3 = np.random.randn(3, 1)
bias3 = np.random.randn(1)

# Training parameters
epochs = 10000
learning_rate = 0.01  # Tanh works well with smaller learning rates

# Training loop
for epoch in range(epochs):
    # Forward pass
    z1, activation1, z2, activation2, z3, output = forward_pass(X, weights1, bias1, weights2, bias2, weights3, bias3)

    # Compute error
    output_error = y_true - output

    # Backward pass
    delta_output = output_error * tanh_derivative(z3)

    error_hidden2 = delta_output.dot(weights3.T)
    delta_hidden2 = error_hidden2 * tanh_derivative(z2)

    error_hidden1 = delta_hidden2.dot(weights2.T)
    delta_hidden1 = error_hidden1 * tanh_derivative(z1)

    # Update weights and biases
    weights3 += activation2.T.dot(delta_output) * learning_rate
    bias3 += np.sum(delta_output, axis=0) * learning_rate

    weights2 += activation1.T.dot(delta_hidden2) * learning_rate
    bias2 += np.sum(delta_hidden2, axis=0) * learning_rate

    weights1 += X.T.dot(delta_hidden1) * learning_rate
    bias1 += np.sum(delta_hidden1, axis=0) * learning_rate

    # (Optional) Print loss every 1000 epochs
    if (epoch + 1) % 1000 == 0:
        loss = np.mean(np.square(output_error))
        print(f"Epoch {epoch+1}: Loss = {loss:.6f}")

# Final predictions after training
_, _, _, _, _, final_output = forward_pass(X, weights1, bias1, weights2, bias2, weights3, bias3)

# Display input-output mapping
print("\nInput -> Expected Output -> Predicted Output")
for input_vec, true_label, prediction in zip(X, y_true, final_output):
    print(f"{list(input_vec)} -> {true_label[0]} -> {prediction[0]:.4f}")


    """



def nim1():
    return """
#computer wins or draws
def minimax(sticks, is_maximizing, memo={}):
    # Create unique key for memoization
    key = (sticks, is_maximizing)
    if key in memo:
        return memo[key]
    
    # Base case - exactly 1 stick left (next player must take it and lose)
    if sticks == 1:
        return -1 if is_maximizing else 1
    
    # Base case - no sticks left (previous player took the last one and lost)
    if sticks <= 0:
        return 1 if is_maximizing else -1
        
    if is_maximizing:
        # Computer's turn - try to maximize score
        best_val = -float('inf')
        for move in range(1, min(4, sticks + 1)):
            val = minimax(sticks - move, False, memo)
            best_val = max(best_val, val)
        memo[key] = best_val
        return best_val
    else:
        # Player's turn - they try to minimize score
        best_val = float('inf')
        for move in range(1, min(4, sticks + 1)):
            val = minimax(sticks - move, True, memo)
            best_val = min(best_val, val)
        memo[key] = best_val
        return best_val

def find_best_move(sticks):
    best_score = -float('inf')
    best_move = 1
    
    for move in range(1, min(4, sticks + 1)):
        score = minimax(sticks - move, False, {})
        if score > best_score:
            best_score = score
            best_move = move
    
    return best_move

def nim_game_computer_wins():
    sticks = int(input("Enter number of sticks to start with: "))
    current_player = input("Who goes first? (p for player/c for computer): ").lower()
    
    while sticks > 0:
        print(f"\nSticks remaining: {sticks}")
        
        if current_player == 'p':
            # Player's turn
            while True:
                try:
                    move = int(input("How many sticks do you take? (1-3): "))
                    if 1 <= move <= 3 and move <= sticks:
                        break
                    print("Invalid move! Take 1-3 sticks.")
                except ValueError:
                    print("Please enter a number.")
            
            sticks -= move
            if sticks == 0:
                print("You took the last stick. You lose!")
            current_player = 'c'
            
        else:
            # Computer's turn
            move = find_best_move(sticks)
            print(f"Computer takes {move} stick(s).")
            sticks -= move
            if sticks == 0:
                print("Computer took the last stick. Computer loses!")
            current_player = 'p'

if __name__ == "__main__":
    print("Nim Game - Computer plays to win!")
    nim_game_computer_wins()

    """



def nim2():
    return """
#computer loses or draws
def minimax(sticks, is_maximizing, memo={}):
    # Create unique key for memoization
    key = (sticks, is_maximizing)
    if key in memo:
        return memo[key]
    
    # Base case - exactly 1 stick left (next player must take it and lose)
    if sticks == 1:
        return -1 if is_maximizing else 1
    
    # Base case - no sticks left (previous player took the last one and lost)
    if sticks <= 0:
        return 1 if is_maximizing else -1
        
    if is_maximizing:
        # Computer's turn - try to minimize score (to lose)
        best_val = float('inf')
        for move in range(1, min(4, sticks + 1)):
            val = minimax(sticks - move, False, memo)
            best_val = min(best_val, val)
        memo[key] = best_val
        return best_val
    else:
        # Player's turn - they try to maximize score
        best_val = -float('inf')
        for move in range(1, min(4, sticks + 1)):
            val = minimax(sticks - move, True, memo)
            best_val = max(best_val, val)
        memo[key] = best_val
        return best_val

def find_worst_move(sticks):
    worst_score = float('inf')
    worst_move = 1
    
    for move in range(1, min(4, sticks + 1)):
        score = minimax(sticks - move, False, {})
        if score < worst_score:
            worst_score = score
            worst_move = move
    
    return worst_move

def nim_game_computer_loses():
    sticks = int(input("Enter number of sticks to start with: "))
    current_player = input("Who goes first? (p for player/c for computer): ").lower()
    
    while sticks > 0:
        print(f"\nSticks remaining: {sticks}")
        
        if current_player == 'p':
            # Player's turn
            while True:
                try:
                    move = int(input("How many sticks do you take? (1-3): "))
                    if 1 <= move <= 3 and move <= sticks:
                        break
                    print("Invalid move! Take 1-3 sticks.")
                except ValueError:
                    print("Please enter a number.")
            
            sticks -= move
            if sticks == 0:
                print("You took the last stick. You lose!")
            current_player = 'c'
            
        else:
            # Computer's turn - choose worst move
            move = find_worst_move(sticks)
            print(f"Computer takes {move} stick(s).")
            sticks -= move
            if sticks == 0:
                print("Computer took the last stick. Computer loses!")
            current_player = 'p'

if __name__ == "__main__":
    print("Nim Game - Computer plays to lose!")
    nim_game_computer_loses()


    """


