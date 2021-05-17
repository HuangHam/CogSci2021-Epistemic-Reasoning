import numpy as np
import networkx as nx 
import itertools as it
import src.utils as f


class modal_model(object):
    '''
    The epistemic model of the task and how to update it
    '''
    def __init__(self):
        self.all_states = ['AAAA88', 'AAA8A8', 'AAA888',
                  'AA88AA', 'AA88A8', 'AA8888',
                  'A8AAA8', 'A8AA88', 'A8A8AA',
                  'A8A8A8', 'A8A888', 'A888AA',
                  'A888A8', '88AAAA', '88AAA8',
                  '88AA88', '88A8AA', '88A8A8',
                  '8888AA'] # by default A before 8 each pair
        self.num_all_states = len(self.all_states)
        self.player_number = {'You':1, 'Amy':2, 'Ben':3} # 'you' means the participant
        self.player_one_uncertainty = [(1, 8), (1, 16), (8, 16),
                           (7, 15), (2, 10), (10, 18),
                           (2, 18), (5, 13), (4, 12),
                           (12, 19), (4, 19), (9, 17),
                           (3, 11)]
        self.player_two_uncertainty = [(1, 3), (1, 6), (3, 6),
                           (7, 10), (10, 13), (7, 13),
                           (14, 19), (14, 17), (17, 19),
                           (9, 12), (18, 15), (2, 5),
                           (8, 11)]
        self.player_three_uncertainty = [(4, 5), (5, 6), (4, 6),
                             (9, 10), (9, 11), (10, 11),
                             (14, 15), (15, 16), (14, 16),
                             (2, 3), (12, 13), (7, 8),
                             (17, 18)]
        self.inference_level = {'A8A8AA:You,Amy,Ben':4, 'A8A8AA:Amy,Ben,You':4, 'AAA888:You,Amy,Ben':4, 
                                'A8AAA8:Amy,Ben,You':4, 'AA88A8:Amy,Ben,You':4, 'AAA8A8:You,Amy,Ben':3, 
                                'AAA8A8:Amy,Ben,You':3, 'AAA8A8:Ben,You,Amy':3, 'A8A8A8:Amy,Ben,You':3, 
                                'A8A8A8:Ben,You,Amy':3, 'A8A8A8:You,Amy,Ben':3, 'A8A8AA:Ben,You,Amy':2, 
                                'A8AAA8:You,Amy,Ben':2, 'A8AAA8:Ben,You,Amy':2, 'AA88A8:Ben,You,Amy':2, 
                                'AA88A8:You,Amy,Ben':2, 'AAA888:Amy,Ben,You':4, 'AAA888:Ben,You,Amy':2,
                                'A8AA88:Amy,Ben,You':1, 'A8AA88:Ben,You,Amy':1, 'A8AA88:You,Amy,Ben':1, 
                                'AAAA88:Ben,You,Amy':1, 'AAAA88:You,Amy,Ben':1, 'AAAA88:Amy,Ben,You':1, 
                                'AA88AA:Amy,Ben,You':1, 'AA88AA:Ben,You,Amy':1, 'AA88AA:You,Amy,Ben':1,
                                'AA8888:Ben,You,Amy':0, 'AA8888:Amy,Ben,You':0, 'AA8888:You,Amy,Ben':0}
        self.iso_map = {'8888AA': 'AAAA88', '88A8AA': 'AAA888', '88AAAA': 'AA8888',
                        '88AAA8': 'AA88A8', '88A8A8': 'AAA8A8', '88AA88': 'AA88AA',
                        'A888AA': 'A8AA88', 'A8A888': 'A8A8AA', 'A888A8': 'A8AAA8',
                        'A8A8A8':'A8A8A8',
                        'AAAA88': 'AAAA88', 'AAA888': 'AAA888', 'AA8888': 'AA8888',
                        'AA88A8': 'AA88A8', 'AAA8A8': 'AAA8A8', 'AA88AA': 'AA88AA',
                        'A8AA88': 'A8AA88', 'A8A8AA': 'A8A8AA', 'A8AAA8': 'A8AAA8'} # maps to equivalent states
    # helper functions
    def map_str_to_int(self,s):
        if s == 'A':
            return 1
        elif s == '8':
            return 8
    def a_and_e(self,n, players_dict, play_order):
        cards = players_dict[play_order[n]]
        if cards == (1, 8) or cards == (8, 1):
            return True
        else:
            return False
    def convert_default_to_game_order(self, tuple_state_order):
        '''
        The experiment code used a different convention for representing state
        Instead of fixed as PAB, it follows the order
        '''
        state = tuple_state_order[0]
        order = tuple_state_order[1]

        def map_name_to_num(name):
            if name == 'You':
                return [0, 2]
            elif name == 'Amy':
                return [2, 4]
            elif name == 'Ben':
                return [4, 6]

        s = ''
        for i in range(len(order)):
            indices = map_name_to_num(order[i])
            s = s + state[indices[0]:indices[1]]
        return list(s)
    def convert_game_order_to_default(self, tuple_state_order):
        '''
        The experiment code used a different convention for representing state
        Instead of fixed as PAB, it follows the order
        '''
        state = tuple_state_order[0]
        order = tuple_state_order[1]

        def map_name_to_num(name):
            if name == 'You':
                return [0, 2]
            elif name == 'Amy':
                return [2, 4]
            elif name == 'Ben':
                return [4, 6]

        l = [None]*len(state)
        for i in range(len(order)):
            indices = map_name_to_num(order[i])
            l[indices[0]:indices[1]] = state[2*i:2*i+2]
        return l
    # graph update functions
    def generate_full_model(self):
        '''
        Generate the full initial epistemic structure
        return (nx object): the initial full epistemic structure
        '''
        G = nx.Graph()
        index = np.arange(1, self.num_all_states + 1, 1)
        for i in range(self.num_all_states):
            G.add_node(index[i], state=self.all_states[i])
        G.add_edges_from(self.player_one_uncertainty, player=1)
        G.add_edges_from(self.player_two_uncertainty, player=2)
        G.add_edges_from(self.player_three_uncertainty, player=3)
        return G
    def compute_my_response(self, graph, Amy_cards, Ben_cards):
        '''
        Compute participant's response 
        return (-1 or bool): 
            -1: eliminated all possible states that are consistent with Amy's and Ben's cards seen by the participant (doesn't happen for DEL)
            True: only one possible state left so respond 'I know my cards'
            False: more than one possible states left so respond 'I don't know my cards'
        '''
        G = graph
        possibilities = [] # possible states for the participant
        for i in list(G.nodes):
            if G.nodes[i]['state'][2:4] == Amy_cards and G.nodes[i]['state'][4:6] == Ben_cards:
                possibilities.append(i)
        n = len(possibilities)
        if n == 0:
            return -1
        elif n == 1:
            return True
        else:
            return False
    def compute_correct_response(self, graph, state, player):
        '''
        Given a model, state of cards ('AA88A8'), and player number (int), 
        return a bool whether the player should know her card
        '''
        G = graph
        state_index = [i for i, d in G.nodes(data=True) if d['state'] == state][0]
        players_with_uncertainties = set([G.edges[state_index, v]['player'] for (u, v) in G.edges([state_index])])
        if player not in players_with_uncertainties:
            return True
        else:
            return False
    def compute_possible_states(self, graph, state, player):
        '''
        Give the list of possible states compatible with a model for a player
        player: 1,2 or 3
        '''
        G = graph
        state_index = [i for i, d in G.nodes(data=True) if d['state'] == state][0]
        other_states_possible = [G.nodes(data=True)[v]['state'] for (u, v) in G.edges([state_index])
                                      if G.edges[state_index, v]['player'] == player]
        return [state]+other_states_possible
    def update_model_neg(self, graph, player):
        '''
        Update the model if a player announces she doesn't know
        '''
        G = graph.copy()
        nodes_to_del = []
        for n in G.nodes():
            players_with_uncertainties = set([G.edges[n, v]['player'] for (u, v) in G.edges([n])])
            if player not in players_with_uncertainties:
                nodes_to_del.append(n)
        for n in nodes_to_del:
            G.remove_node(n)
        return G
    def update_model_pos(self, graph, player):
        '''
        Update the model if a player announces knows
        '''
        G = graph.copy()
        nodes_to_del = []
        for n in G.nodes():
            players_with_uncertainties = set([G.edges[n, v]['player'] for (u, v) in G.edges([n])])
            if player in players_with_uncertainties:
                nodes_to_del.append(n)
        for n in nodes_to_del:
            G.remove_node(n)
        return G
    def update_model(self, graph, announcement, player_number):
        '''
        Update the model according to a player's announcement
        '''
        if announcement:
            G = self.update_model_pos(graph, player_number)
        else:
            G = self.update_model_neg(graph, player_number)
        return G
    def compute_subj_response(self,cards, order):
        '''
        cards: an list of 8 characters representing 8 cards
        e.g., ['A', '8', 'A', 'A', '8', '8', 'A', '8']
        order: ['Ben', 'You', 'Amy']
        
        return (list of lists): whether player 1 (the human subject) should know their cards or not
        Each innerlist represents a round
        
        Important: first two are USER, second two are Amy, third two are Ben
        Amy has pn (player number) 2 and Ben is 3
        '''
        G = modal_model.generate_full_model(self)
        state = ''.join(cards)[:6]

        curr_asmt = [self.map_str_to_int(s) for s in cards]
        p1_cards = tuple(curr_asmt[:2])
        p2_cards = tuple(curr_asmt[2:4])
        p3_cards = tuple(curr_asmt[4:6])

        players_cards = [p1_cards, p2_cards, p3_cards]
        players_name = ['You', 'Amy', 'Ben']
        players_dict = dict(zip(players_name, players_cards))

        play_order = order
        user_order = order.index('You') + 1
        curr_player_num = -1
        round_num = -1
        game_over = False

        early_stop_cond_1 = play_order[1] != 'You' and self.a_and_e(1, players_dict, play_order)
        early_stop_cond_2 = play_order[1] == 'You' and self.a_and_e(0,
                                                               players_dict, play_order) and self.a_and_e(2, players_dict,
                                                                                                     play_order)
        early_stop = early_stop_cond_1 or early_stop_cond_2
        responses = []
        while game_over == False:
            curr_player_num = (curr_player_num + 1) % 3
            curr_player = play_order[curr_player_num]
            if curr_player_num == 0:
                responses.append(list())
                round_num += 1
            if curr_player == 'You':
                pn = 1
                correct_response = self.compute_correct_response(G, state, 1)
                if correct_response:
                    game_over = True
                else:
                    if early_stop and round_num == 2:  # How many rounds?
                        game_over = True
                    else:
                        G = modal_model.update_model_neg(self, G, pn)
                responses[round_num].append(correct_response)
            else:
                if curr_player == 'Amy':
                    pn = 2
                else:
                    pn = 3
                correct_response = self.compute_correct_response(G, state, 1)
                agent_response = self.compute_correct_response(G, state, pn)
                if correct_response:
                    game_over = True
                if agent_response:
                    G = modal_model.update_model_pos(self, G, pn)
                else:
                    G = modal_model.update_model_neg(self, G, pn)
                responses[round_num].append(correct_response)
        return responses
    def compute_game_response(self, cards, order):
        '''
        cards: an list of 8 characters representing 8 cards
        e.g., ['A', '8', 'A', 'A', '8', '8', 'A', '8']
        order: ['Ben', 'You', 'Amy']
        player: 1, 2, or 3
        
        return (list of lists): whether each agent knows their cards or not
        Each innerlist represents a round
        
        Important: first two are USER, second two are Amy, third two are Ben
        Amy has pn (player number) 2 and Ben is 3
        '''
        G = self.generate_full_model()
        state = ''.join(cards)[:6]

        curr_asmt = [self.map_str_to_int(s) for s in cards]
        p1_cards = tuple(curr_asmt[:2])
        p2_cards = tuple(curr_asmt[2:4])
        p3_cards = tuple(curr_asmt[4:6])
        players_cards = [p1_cards, p2_cards, p3_cards]
        players_name = ['You', 'Amy', 'Ben']
        players_dict = dict(zip(players_name, players_cards))

        play_order = order
        user_order = order.index('You') + 1
        curr_player_num = -1
        round_num = -1

        game_over = False

        early_stop_cond_1 = play_order[1] != 'You' and self.a_and_e(1, players_dict, play_order)
        early_stop_cond_2 = play_order[1] == 'You' and self.a_and_e(0,
                                                               players_dict, play_order) and self.a_and_e(2, players_dict,
                                                                                                     play_order)

        early_stop = early_stop_cond_1 or early_stop_cond_2

        responses = []

        while game_over == False:

            curr_player_num = (curr_player_num + 1) % 3
            curr_player = play_order[curr_player_num]

            if curr_player_num == 0:
                responses.append(list())
                round_num += 1

            if curr_player == 'You':
                pn = 1
                correct_response = self.compute_correct_response(G, state, pn)
                if correct_response:
                    game_over = True
                else:
                    if early_stop and round_num == 2: 
                        game_over = True
                    else:
                        G = self.update_model_neg(G, pn)
                responses[round_num].append(correct_response)

            else:
                if curr_player == 'Amy':
                    pn = 2
                else:
                    pn = 3
                correct_response = self.compute_correct_response(G, state, pn)
                if correct_response:
                    G = self.update_model_pos(G, pn)
                else:
                    G = self.update_model_neg(G, pn)
                responses[round_num].append(correct_response)

        return responses

class bounded_modal_model(modal_model):
    def __init__(self):
        super().__init__()
    def generate_full_model(self):
        '''
        Generate the full epistemic structure (as directed graph)
        '''
        def swap(lst):
            l = []
            for x in lst:
                l.append((x[1], x[0]))
            return l

        G = nx.DiGraph()
        index = np.arange(1, self.num_all_states + 1, 1)
        for i in range(self.num_all_states):
            G.add_node(index[i], state=self.all_states[i])
        G.add_edges_from(self.player_one_uncertainty, player=1)
        G.add_edges_from(self.player_two_uncertainty, player=2)
        G.add_edges_from(self.player_three_uncertainty, player=3)
        G.add_edges_from(swap(self.player_one_uncertainty), player=1)
        G.add_edges_from(swap(self.player_two_uncertainty), player=2)
        G.add_edges_from(swap(self.player_three_uncertainty), player=3)
        return G
    def update_model_neg(self, graph, player):
        '''
        Update the model if a player announces she doesn't know
        '''
        G = graph.copy()
        nodes_to_del = []
        elimable = [i for i in G.nodes if G.nodes[i]['elimable']]
        for n in elimable:
            players_with_uncertainties = set([G.edges[n, v]['player'] for (u, v) in G.edges([n])])
            if player not in players_with_uncertainties:
                nodes_to_del.append(n)
        for n in nodes_to_del:
            G.remove_node(n)
        return G
    def update_model_pos(self, graph, player):
        '''
        Update the model if a player announces knows
        '''
        G = graph.copy()
        nodes_to_del = []
        elimable = [i for i in G.nodes if G.nodes[i]['elimable']]
        for n in elimable:
            players_with_uncertainties = set([G.edges[n, v]['player'] for (u, v) in G.edges([n])])
            if player in players_with_uncertainties:
                nodes_to_del.append(n)
        for n in nodes_to_del:
            G.remove_node(n)
        return G
    def update_model(self, graph, announcement, player_number):
        '''
        Update the model according to a player's announcement
        '''
        if announcement:
            G = self.update_model_pos(graph, player_number)
        else:
            G = self.update_model_neg(graph, player_number)
        return G
    def generate_partial_model(self, Amy_cards, Ben_cards, level=4):
        G, full_graph = nx.DiGraph(), self.generate_full_model()
        index = np.arange(1, self.num_all_states + 1, 1)
        initial = []
        for i in range(self.num_all_states):
            if self.all_states[i][2:4] == Amy_cards and self.all_states[i][4:6] == Ben_cards:
                initial.append(i)
        for i in initial:
            G.add_node(index[i], state=self.all_states[i], elimable=True)
        if len(list(G.nodes)) > 1:
            perm = it.permutations(list(G.nodes), 2)
            for edge in list(perm):
                G.add_edge(edge[0], edge[1], player=1)
        for l in range(int(level)):
            nodes = list(G.nodes)
            to_remove = []
            for u in nodes:
                for v in full_graph.neighbors(u):
                    if v not in list(G.nodes):
                        if l < int(level) - 1:
                            G.add_node(v, state=full_graph.nodes[v]['state'], elimable=True)
                        else:
                            G.add_node(v, state=full_graph.nodes[v]['state'], elimable=False)
                    G.add_edge(u, v, player=full_graph[u][v]['player'])

        return G
    
class imperfect_update_model(bounded_modal_model):
    def __init__(self):
        super().__init__()
    def update_model_neg(self, graph, elim_prob, player):
        '''
        Update the model if a player announces she doesn't know
        '''
        G = graph.copy()
        nodes_to_del = []
        elimable = [i for i in G.nodes if G.nodes[i]['elimable']]
        for n in elimable:
            players_with_uncertainties = set([G.edges[n, v]['player'] for (u, v) in G.edges([n])])
            if player not in players_with_uncertainties:
                nodes_to_del.append(n)
        for n in nodes_to_del:
            if np.random.random_sample() < elim_prob:
                G.remove_node(n)
        return G
    def update_model_pos(self, graph, elim_prob, player):
        '''
        Update the model if a player announces knows
        '''
        G = graph.copy()
        nodes_to_del = []
        elimable = [i for i in G.nodes if G.nodes[i]['elimable']]
        for n in elimable:
            players_with_uncertainties = set([G.edges[n, v]['player'] for (u, v) in G.edges([n])])
            if player in players_with_uncertainties:
                nodes_to_del.append(n)
        for n in nodes_to_del:
            if np.random.random_sample() < elim_prob:
                G.remove_node(n)
        return G
    def update_model(self, graph, elim_prob, announcement, player_number):
        '''
        Update the model according to a player's announcement
        '''
        if announcement:
            G = self.update_model_pos(graph, elim_prob, player_number)
        else:
            G = self.update_model_neg(graph, elim_prob, player_number)
        return G
