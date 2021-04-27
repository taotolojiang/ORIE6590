Airline Revenue Management

This project aims to optimize the revenue of a hub which serves L destinations using reinforcement learning. We analyze both single and two legged itineraries. Every possible itinerary has a high-fare and low-fare class. We use m to denote the total number of single-legged itineraries and n to denote the total number of fare and itinerary combinations that a customer could belong to.

The state space is the set of all possible available seats for every flight into and out of each location up to the full capacities.

The action space is all possible binary vectors of length n which tells you whether a customer (with a specific fare and itinerary) is accepted or declined by the airline company.

The one-step reward is the revenue gained from applying the predetermined action (of this time-step) to a customer who appears during this time-step (at most one will do so).

The transitions are of the form: (state = [...c_i...c_j...], action[k] = 1, customerClass = k, customerDemand = [0,...,d_i,...,d_j,...,0], c - d >= 0, i != j, d_i, d_j not both 0) --> newState = c - d (noticeably, this will only differ at indices i and j). All other transitions lead to newState = state.
