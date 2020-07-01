import networkx as nx

def transform_pagerank(pi):
	n_nodes = len(pi)
	PI = {}
	for i in range(n_nodes):
		PI[i] = pi[i]
	return PI

def normalize(pi):
	pii = pi
	sum_elems = sum(x for x in pi)
	for i in range(len(pi)):
		pii[i] = pi[i]/sum_elems
	return pii

def pagerank(g, max_iter, alpha, tau):
	sg = nx.stochastic_graph(g)		#stochastic graph
	n_nodes = nx.number_of_nodes(g)
	nodes = g.nodes()

	PI = [1.0/n_nodes] * n_nodes	#initialization of pagerank
	
	a = []							#dangling nodes vector
	for n in nodes:
		if g.out_degree(n):
			a.append(1)
		else:
			a.append(0)

	H = nx.adjacency_matrix(sg)		
	for i in range(max_iter):
		pi_previous = PI

		#v1 = alpha(pi_previous^T*H)
		v1 = [0] * n_nodes
		for r in range(n_nodes):
			row = H[r,:].toarray()		
			for c in range(n_nodes):
				v1[c] += pi_previous[c]*row[0][c]
		v1 = [alpha*v for v in v1]

		#v2 = alpha(pi_previous^T*a)1/n*e^T
		dang_pi = 0
		for e in range(n_nodes):
			dang_pi += pi_previous[e]*a[e]
		constant = alpha*dang_pi+1-alpha
		v2 = [float(constant)/n_nodes] * n_nodes

		#pi = v2 + v3
		for e in range(n_nodes): 
			PI[e] = v1[e] + v2[e]
		
		PI = normalize(PI)

		#check convergence
		delta = 0 
		for e in range(n_nodes):
			delta += abs(PI[e] - pi_previous[e])
		if delta < tau*n_nodes:
			return transform_pagerank(PI)
	return transform_pagerank(PI)