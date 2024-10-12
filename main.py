import random
import networkx as nx
import time
import heapq
import math


def astar(graph, start, target, queue_type):
    if queue_type == "binary_heap":
        priority_queue = [(0, start)]  # create a priority queue using a linear array and add the start node
        heapq.heapify(priority_queue)  # turn priority queue into binary heap
    elif queue_type == "ordered_array" or queue_type == "unordered_array":
        priority_queue = [(0, start)]
    g_score = {node: float('inf') for node in
               graph.nodes}  # create a dictionary g_score which holds the shortest known distance to all nodes, and initialise all values to infinity
    g_score[start] = 0  # set g_score of start node to 0
    f_score = {node: float('inf') for node in
               graph.nodes}  # create a dictionary f_score which holds the heuristic to all nodes, initialising all values to infinity
    f_score[start] = heuristic(start, target, graph)  # calculate heuristic of start node
    start_time = time.time()  # start a timer to record the execution time of algorithm
    while priority_queue:  # while the priority queue still contains elements
        if queue_type == "binary_heap":
            curr = heapq.heappop(priority_queue)[1]  # remove the first element of the priority queue
        elif queue_type == "ordered_array":
            priority_queue.sort(key=lambda x: x[0])
            curr = priority_queue.pop(0)[1]  # sort array and then remove the first element
        elif queue_type == "unordered_array":
            min = 0
            for i in range(1, len(priority_queue)):
                if priority_queue[i][0] < priority_queue[min][0]:
                    min = i
            curr = priority_queue.pop(min)[1]
            # iterate over the array, storing the index of the smallest known distance in the priority queue currently

        if curr == target:  # if target node is found
            end_time = time.time()
            execution_time = end_time - start_time
            # find the change in time of the algorithm
            return execution_time

        # otherwise, iterate over all neighbours and recalculate the g values and f_values
        for neighbour in graph.neighbours(curr):
            tentative_g = g_score[curr] + graph[curr][neighbour]["weight"]
            if tentative_g < g_score[neighbour]:
                g_score[neighbour] = tentative_g
                f_score[neighbour] = tentative_g + heuristic(neighbour, target, graph)
                if queue_type == "binary_heap":
                    heapq.heappush(priority_queue, (f_score[neighbour],
                                                    neighbour))  # if the g_score or f_score is different than the currently known ones, then push the new node onto the priority queue
                elif queue_type == "ordered_array":
                    i = 0
                    while index < len(priority_queue) and priority_queue[index][0] <= f_score[neighbour]:
                        i += 1
                    priority_queue.insert(i, (f_score[neighbour], neighbour))
                    # while the end of the array hasn’t been reached and the f_score is larger than the f_score of the node in the priority queue, move to the next element

                elif queue_type == "unordered_array":
                    priority_queue.append((f_score[neighbour], neighbour))
                # add the new node to the end of the unordered array

    return 0


def heuristic(curr, target, graph):
    # Euclidean distance calculation for heuristic
    x1, y1 = curr['x'], curr['y']
    x2, y2 = target['x'], target['y']
    distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    return distance


def generate_graph(num_nodes, density):
    graph = nx.Graph()
    # create graph using networkx package
    for node in range(num_nodes):
        x = random.randint(0, 100)
        y = random.randint(0, 100)
        # generate a random x,y coordinate for node

        graph.add_node(node, x=x, y=y)
        # add node to graph

    for node in graph.nodes:
        for neighbour in range(node + 1, num_nodes):
            if random.random() < density:  # generate a random floating point between 0 and 1; if it is greater than the density, than no edge is generated, other wise an edge is generated
                x_neighbour = graph.nodes[neighbour]['x']
                y_neighbour = graph.nodes[neighbour]['y']
                weight = calculate_weight(graph.nodes[node], {'x': x_neighbour, 'y': y_neighbour})
                graph.add_edge(node, neighbour, weight=weight)

    return graph


def calculate_weight(node1, node2):
    x1, y1, x2, y2 = node1['x'], node1['y'], node2['x'], node2['y']
    distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) + random.uniform(0, 10)
    # generate a weighted edge using Euclidean distance and then adding a random variation floating point between 0 and 10
    return distance
