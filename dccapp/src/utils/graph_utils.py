import numpy as np
import pandas as pd

import networkx as nx
import matplotlib.pyplot as plt
from operator import itemgetter


class mgcgraph(object):
    """plot graph given a networkx graph type"""

    def __init__(self, graph_input, **kwargs):
        """
        Arguement:
            graph_input: can be directed or undirected graph create with networkx
        """
        # super(mgcgraph, self).__init__()
        self.graph_input = graph_input
        
    def mgcgraph_plt(self, **kwargs):
        """
        Arguement:
            graph_in: input graph, if not provided, the same graph used to initiate the method is used
            axs_k: matplotlib axes
            centrality_algo: string with centraility algorithm, different algorithm available for both directed and undirected graph
                directed: ['degree', 'betweenness', 'eigenvector', 'harmonic', 'load', 'closeness']
                undirected: ['degree', 'betweenness', 'eigenvector', 'flow_closeness', 'harmonic', 'information']
            var_title: string to describe on plot tile
            node_size_range: a list of 2 numeric values
            node_att: node attribute available in the graph, only applicable if centrality_algo is set to None
            layout_type: string represent graph layout algorithm to be used, default is 'kamada_kawai_layout', other option
                ['circular_layout', 'random_layout', 'shell_layout', 'spring_layout', 'spectral_layout', 'kamada_kawai_layout', 'spiral_layout']
            
        """
        import matplotlib
        # user matplotlib default
        matplotlib.rcParams.update(matplotlib.rcParamsDefault)
        plt.style.use('default')
        import warnings
        warnings.filterwarnings('ignore')
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        
        # default values used when argument is not provided
        def_vals = {'graph_in' : self.graph_input,
                    'axs_k': None, 
                    'centrality_algo': 'degree', 
                    'var_title': None, 
                    'node_size_range': [100, 2000], 
                    'node_att': None, 
                    'node_att_thresh': 0.5, 
                    'node_att_categorical': 'normal',
                    'edge_att': None, # not used
                    'layout_type': 'kamada_kawai_layout', 
                    'local_legend': True, 
                    'local_colorbar': True, 'target_node': None}

        for k, v in def_vals.items():
            kwargs.setdefault(k, v)

        graph_in = kwargs['graph_in']
        axs_k = kwargs['axs_k']
        centrality_algo = kwargs['centrality_algo']
        var_title = kwargs['var_title']
        node_size_range = kwargs['node_size_range']
        node_att = kwargs['node_att']
        node_att_thresh = kwargs['node_att_thresh']
        node_att_categorical = kwargs['node_att_categorical']
        edge_att = kwargs['edge_att'] # for future
        layout_type = kwargs['layout_type']
        local_legend = kwargs['local_legend']
        local_colorbar = kwargs['local_colorbar']
        target_node = kwargs['target_node']
        
        if node_att_thresh is None:
            node_att_thresh = 0.3
        if node_att_categorical is None:
            node_att_categorical = 'normal'
            
        # function to scale a list to min-max range
        def norm_list2(ls, rangex):
            return [((rangex[1] - rangex[0]) * ((float(i) - min(ls))/(np.finfo(np.double).eps + max(ls) - min(ls)))) + rangex[0] for i in ls]
        
#         print("edge {}, node {}".format(graph_in.number_of_edges(), graph_in.number_of_nodes()))
        if graph_in.number_of_edges() > 0:
            if edge_att is None:
                # hard coded to edge_betweenness for graph edge
                dict_edge_centrality = {'edge_flow': nx.edge_current_flow_betweenness_centrality,
                                        'edge_betweenness':nx.edge_betweenness_centrality}

                edge_centr_attr = 'edge_betweenness'
                # Set edge attribute
                edges_x, edge_color_x = zip(*dict_edge_centrality[edge_centr_attr](graph_in).items())
                edge_width_x = edge_color_x

                # scale data
    #             edge_color_x = norm_list2(edge_color_x, [1, 10])
                edge_color_x = [round(i, 3) for i in edge_color_x]
                edge_width_x = norm_list2(edge_width_x, [1, 2])
                dict_labels = dict(zip(edges_x, edge_color_x))
                
            elif edge_att is not None:
                dict_edge_x = nx.get_edge_attributes(graph_in, edge_att)
                edges_x, edge_color_x = zip(*dict_edge_x.items())
                edge_width_x = edge_color_x
                edge_color_x = [round(i, 3) for i in edge_color_x]
                edge_width_x = norm_list2(edge_width_x, [1, 2])
                dict_labels = dict(zip(edges_x, edge_color_x))
        else:
            edge_color_x = 1
            edge_width_x = 1            
        dict_class_inv = {'no_issue': 0,
                         'access_cell_edge': 1,
                         'resource_limit': 2,
                         'high_load': 3,
                         'access_resource': 4,
                         'ho_SR': 5,
                         'prach_interference': 6,
                         'uplink_interf': 7,
                         'normal': 8,
                         'signaling_load': 9,
                         'signaling_SR': 10,
                         'others': 11,
                         'very_high_load': 12}
        
        # Select node cetrality option
        dict_ctr = {'directed': {'degree': nx.degree_centrality, 
                       'betweenness': nx.betweenness_centrality, 
                       'eigenvector': nx.eigenvector_centrality_numpy, 
                       'harmonic': nx.harmonic_centrality, 
                       'load': nx.load_centrality, 
                       'closeness': nx.closeness_centrality},
                    'undirected': {'degree': nx.degree_centrality, 
                       'betweenness': nx.betweenness_centrality, 
                       'eigenvector': nx.eigenvector_centrality_numpy, 
                       'flow_closeness': nx.current_flow_closeness_centrality, 
                       'harmonic': nx.harmonic_centrality,
                       'information': nx.information_centrality}}
        
        # check if the graph is directed or undirected
        if nx.is_directed(graph_in):
            dict_centrality = dict_ctr.get('directed')
        else:
            dict_centrality = dict_ctr.get('undirected')
        
        if centrality_algo is None:
            dict_tmp = nx.get_node_attributes(graph_in, node_att)
            ls_tmp = list(dict_tmp.values())
        
            if type(ls_tmp[0]) == float: #if node attribute is string type
                sorted_node_attri = sorted(dict_tmp.items(), key=itemgetter(1), reverse=True)
                ls_node1 = [i[0] for i in sorted_node_attri if i[1] > node_att_thresh]
                ls_node2 = list(set(graph_in.nodes()) - set(ls_node1))
                # Set plot node size and color    
                node_size_x1 = [dict_tmp[v] for v in ls_node1]
                node_color_x1 = [dict_tmp[v] for v in ls_node1]

                node_size_x2 = [dict_tmp[v] for v in ls_node2]
                node_color_x2 = [dict_tmp[v] for v in ls_node2]

                # Scale a list with min and max
                node_size_x1 = [(size_i + 5000) for size_i in node_size_x1]
                node_size_x1 = norm_list2(node_size_x1, node_size_range)
                node_size_x2 = [(size_i + 50) for size_i in node_size_x2]
                node_size_x2 = norm_list2(node_size_x2, [i/2 for i in node_size_range])
                # replace nan with a value
                node_size_x1 = [20 if np.isnan(i) else i for i in node_size_x1]
                node_size_x2 = [20 if np.isnan(i) else i for i in node_size_x2]
                
                node_color_x1 = [min(node_color_x1) if np.isnan(i) else i for i in node_color_x1]
                node_color_x2 = [min(node_color_x2) if np.isnan(i) else i for i in node_color_x2] 
            else:
                dict_class_color = dict(zip(set(ls_tmp), range(1, len(ls_tmp) + 1))) # map categorical to integer
                ls_node1 = [k for k, v in dict_tmp.items() if v == node_att_categorical]
                ls_node2 = list(set(graph_in.nodes()) - set(ls_node1))
                node_size_x1 = [node_size_range[1]] * len(ls_node1)
                node_color_x1 = [dict_class_color[dict_tmp[n]] for n in ls_node1]

                node_size_x2 = [node_size_range[0]] * len(ls_node1)
                node_color_x2 = [dict_class_color[dict_tmp[n]] for n in ls_node2]

            
            
#             print(node_size_x1, node_size_x2)
        else:
            # Set node attribute with node centrality
            centrality_type_dict = dict_centrality[centrality_algo](graph_in)
            
            # Assign each to an attribute in your network
            nx.set_node_attributes(graph_in, centrality_type_dict, centrality_algo)
            # Set plot node size and color
            node_size_x = [size_i for size_i in centrality_type_dict.values()]
            node_color_x = [color_i for color_i in centrality_type_dict.values()]
#             print(node_color_x)
            # Scale a list with min and max
            node_size_x = norm_list2(node_size_x, node_size_range)
#             node_color_x = norm_list2(node_color_x, [0, 100]) 

        # Set layout
        def rescl_layout(graph_in):
            # use latitude and longitude as layout position
            ls_node_pos = list(nx.get_node_attributes(graph_in, 'pos').values())
            if len(ls_node_pos) > 0:
                posx = nx.rescale_layout(np.array(ls_node_pos))# using gps position
                layout_d = dict(zip(graph_in.nodes(), posx))
            else: 
                layout_d = nx.random_layout(graph_in)
            return layout_d
        def pydot_neato(graph_in):
            return nx.nx_pydot.pydot_layout(graph_in, 'neato')

        def pydot_dot(graph_in):
            return nx.nx_pydot.pydot_layout(graph_in, 'dot')

        def pydot_twopi(graph_in):
            return nx.nx_pydot.pydot_layout(graph_in, 'twopi')

        def pydot_fdp(graph_in):
            return nx.nx_pydot.pydot_layout(graph_in, 'fdp')

        def pydot_circo(graph_in):
            return nx.nx_pydot.pydot_layout(graph_in, 'circo')


        dict_layout = {'circular_layout': nx.circular_layout, 
                        'random_layout': nx.random_layout, 
                        'shell_layout': nx.shell_layout, 
                        'spring_layout': nx.spring_layout, 
                        'spectral_layout': nx.spectral_layout, 
                        'kamada_kawai_layout': nx.kamada_kawai_layout, 
                        'spiral_layout': nx.spiral_layout, 
                       'rescale_layout':rescl_layout, 
                       'pydot_neato': pydot_neato, 
                       'pydot_dot': pydot_dot, 
                       'pydot_twopi': pydot_twopi, 
                       'pydot_fdp': pydot_fdp, 
                       'pydot_circo': pydot_circo}

        layout_x = dict_layout[layout_type](graph_in)
        
        node_cmapx = plt.cm.autumn
        edge_cmapx = plt.cm.cool

        # draw node label
        if target_node is None:
            labelsx = dict(zip(graph_in.nodes(), graph_in.nodes()))
        else:
            labelsx = {target_node: dict(zip(graph_in.nodes(), graph_in.nodes())).get(target_node)}
            
        if axs_k is None:
            fig, axs_k = plt.subplots(figsize=(18,16))
        
        if centrality_algo is None:
            if len(node_size_x2) != 0:
                plt_node = nx.draw_networkx_nodes(graph_in, 
                                                  pos = layout_x, 
                                                  node_size = node_size_x2,
                                                  node_color = node_color_x2,
                                                  ax = axs_k,
                                                  alpha = 0.2, 
                                                  with_labels = False, 
                                                  cmap = node_cmapx, 
                                                  nodelist = ls_node2, node_shape='o'
                                                 )

                plt_node = nx.draw_networkx_nodes(graph_in, 
                                                  pos = layout_x, 
                                                  node_size = node_size_x1,
                                                  node_color = node_color_x1,
                                                  ax = axs_k,
                                                  alpha = 0.7, 
                                                  with_labels = False, 
                                                  cmap = node_cmapx, 
                                                  nodelist = ls_node1, node_shape='s'
                                                 )
#                 nx.draw_networkx_labels(graph_in, pos = layout_x, labels = labelsx, font_color='r')
            else:
                plt_node = nx.draw_networkx_nodes(graph_in, 
                                                  pos = layout_x, 
                                                  node_size = node_size_x1,
                                                  node_color = node_color_x1,
                                                  ax = axs_k,
                                                  alpha = 0.7, 
                                                  with_labels = False, 
                                                  cmap = node_cmapx, 
                                                  nodelist = ls_node1, node_shape='s'
                                                 ) 

        else:
            plt_node = nx.draw_networkx_nodes(graph_in, 
                                              pos = layout_x, 
                                              node_size = node_size_x,
                                              node_color = node_color_x,
                                              ax = axs_k,
                                              alpha = 0.7, 
                                              with_labels = False, 
                                              cmap = node_cmapx
                                             )
        
        if len(graph_in) > 1:
            plt_edge = nx.draw_networkx_edges(graph_in, 
                                              pos = layout_x, 
                                              ax = axs_k,
                                              alpha = 1, 
                                              connectionstyle = 'arc3,rad=0.3', 
                                              edge_color = edge_color_x,
                                              width = edge_width_x,                                           
                                              edge_cmap=edge_cmapx)
            
        else:
            plt_edge = nx.draw_networkx_edges(graph_in, 
                                              pos = layout_x, 
                                              ax = axs_k,
                                              alpha = 0.2, 
                                              connectionstyle = 'arc3,rad=0.3', 
#                                               edge_color = edge_color_x,
#                                               width = edge_width_x,                                           
                                              edge_cmap=edge_cmapx)


        # show label
        nx.draw_networkx_labels(graph_in, pos = layout_x, labels = labelsx, font_color='b', ax = axs_k,)
#         nx.draw_networkx_edge_labels(graph_in, layout_x, ax = axs_k, edge_labels=dict_labels)  
        axs_k.set_title('nodeid: {0}, node count: {1} edge count: {2}'.format(var_title, graph_in.number_of_nodes(), graph_in.number_of_edges()), fontsize = 14, loc = 'right')
        if local_colorbar:
            
            # create new axes on the right and on the top of the current axes
            if centrality_algo is None:
                divider = make_axes_locatable(axs_k)
                # below height and pad are in inches
                cax2 = divider.append_axes("top", 0.2, pad=0.3) # colorbar for node
                plt_node.set_clim(min(node_color_x1 + node_color_x2), max(node_color_x1 + node_color_x2))
                cb_node = plt.colorbar(plt_node, cax2,  orientation='horizontal').set_label(label="node " + str(node_att if centrality_algo is None else centrality_algo), size=12)
                cax2.xaxis.set_ticks_position("top")
                cax2.xaxis.set_label_position("top")
            else:
                divider = make_axes_locatable(axs_k)
                # below height and pad are in inches
                cax2 = divider.append_axes("top", 0.2, pad=0.3) # colorbar for node value_when_true if condition else value_when_false
                cb_node = plt.colorbar(plt_node, cax2,  orientation='horizontal').set_label(label="node " + str(centrality_algo), size=12)
                cax2.xaxis.set_ticks_position("top")
                cax2.xaxis.set_label_position("top")                
            
            if nx.is_directed(graph_in):
            # colorbar for edge
            # set alpha value for each edge
                M = graph_in.number_of_edges()
                if M > 0:
                    edge_alphas = [(5 + i) / (M + 4) for i in range(M)]
                    for i in range(M):
                        plt_edge[i].set_alpha(edge_alphas[i])

                    pc = matplotlib.collections.PatchCollection(plt_edge, cmap=edge_cmapx)
                    pc.set_array(edge_color_x)
                    cax1 = divider.append_axes('right', size='2%', pad=0.05)
                    plt.colorbar(pc, cax=cax1, orientation='vertical').set_label(label=edge_centr_attr,size=12)

            else:
            # define the bins and normalize
                M = graph_in.number_of_edges()
                if M > 0:
                    cmap = matplotlib.cm.cool
                    norm = matplotlib.colors.Normalize(vmin=min(edge_color_x), vmax=max(edge_color_x))
                    cax1 = divider.append_axes('right', size='2%', pad=0.05) 
                    cb1 = matplotlib.colorbar.ColorbarBase(cax1, cmap=edge_cmapx,
                                                           norm=norm,
                                                           orientation='vertical').set_label(label=" " + str(edge_att if edge_att is not None else edge_centr_attr), size=12)
#                     plt_edge.set_clim(min(edge_color_x), max(edge_color_x))
#                     cax1 = divider.append_axes('right', size='2%', pad=0.05) 
#                     cb_edge = plt.colorbar(plt_edge, cax=cax1).set_label(label=" " + edge_centr_attr,size=12)
        return plt_node, plt_edge


class mgcgraph_adj_m(object):
    """plot graph given a networkx graph type"""

    def __init__(self, graph_input, **kwargs):
        """
        Arguement:
            graph_input: can be directed or undirected graph create with networkx
        """
        super(mgcgraph_adj_m, self).__init__()
        self.graph_input = graph_input
        
    def mgcgraph_plt(self, **kwargs):
        """
        Arguement:
            graph_in: input graph, if not provided, the same graph used to initiate the method is used
            axs_k: matplotlib axes
            centrality_algo: string with centraility algorithm, different algorithm available for both directed and undirected graph
                directed: ['degree', 'betweenness', 'eigenvector', 'harmonic', 'load', 'closeness']
                undirected: ['degree', 'betweenness', 'eigenvector', 'flow_closeness', 'harmonic', 'information']
            var_title: string to describe on plot tile
            node_size_range: a list of 2 numeric values
            node_att: node attribute available in the graph, only applicable if centrality_algo is set to None
            layout_type: string represent graph layout algorithm to be used, default is 'kamada_kawai_layout', other option
                ['circular_layout', 'random_layout', 'shell_layout', 'spring_layout', 'spectral_layout', 'kamada_kawai_layout', 'spiral_layout']
            
        """
        import matplotlib
        # user matplotlib default
        matplotlib.rcParams.update(matplotlib.rcParamsDefault)
        plt.style.use('default')
        import warnings
        warnings.filterwarnings('ignore')
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        
        # default values used when argument is not provided
        def_vals = {'graph_in' : self.graph_input,
                    'axs_k': None, 
                    'var_title': None, 
                    'local_legend': True, 
                    'fig': None, 
                    'local_colorbar': True }

        for k, v in def_vals.items():
            kwargs.setdefault(k, v)

        graph_in = kwargs['graph_in']
        axs_k = kwargs['axs_k']
        var_title = kwargs['var_title']
        local_legend = kwargs['local_legend']
        fig = kwargs['fig']
        local_colorbar = kwargs['local_colorbar']
        
        
        if axs_k is None:
            fig, axs_k = plt.subplots(figsize=(18,12))

        adj_x = nx.adjacency_matrix(graph_in).todense()
        im = axs_k.imshow(adj_x, cmap='Blues', interpolation='nearest')
#         axs_k.grid('off')
#         axs_k.axis('off')
        axs_k.set_title("{} edge: {}, node: {}".format(var_title, graph_in.number_of_edges(), graph_in.number_of_nodes()))
        # Turn spines off and create white grid.
        for edge, spine in axs_k.spines.items():
            spine.set_visible(False)

        axs_k.set_xticks(np.arange(adj_x.shape[1]+1)-.5, minor=True)
        axs_k.set_yticks(np.arange(adj_x.shape[0]+1)-.5, minor=True)
#         axs_k.grid(which="minor", color="g", linestyle='-', linewidth=1)
        axs_k.tick_params(which="minor", bottom=False, left=False)        
        if local_colorbar:
            if nx.is_directed(graph_in):
            # set alpha value for each edge
                M = graph_in.number_of_edges()
                if M > 0:
#                     edge_alphas = [(5 + i) / (M + 4) for i in range(M)]
#                     for i in range(M):
#                         ec[i].set_alpha(edge_alphas[i])

#                     pc = mpl.collections.PatchCollection(im, cmap=plt.cm.Blues)
#                     pc.set_array(edge_colorsx)
                    divider = make_axes_locatable(axs_k)
                    cax = divider.append_axes('right', size='5%', pad=0.05)
                    plt.colorbar(im, cax=cax, orientation='vertical').set_label(label='weight',size=16)
            else:
            # define the bins and normalize
#                 print(edge_colorsx)
                if graph_in.number_of_edges() > 0:
                    bounds = np.linspace(np.min(adj_x), np.max(adj_x), 11)
                    norm = matplotlib.colors.BoundaryNorm(bounds, plt.cm.Blues.N)

                    # create a second axes for the colorbar
                    divider = make_axes_locatable(axs_k)
                    cax = divider.append_axes('right', size='5%', pad=0.05)
                    cbar = matplotlib.colorbar.ColorbarBase(cax, cmap=plt.cm.Blues, ticks=bounds, spacing='proportional',
                                                     boundaries=bounds).set_label(label='weight',size=16)


class mgcgraph_adj_m2(object):
    """plot graph given a networkx graph type"""

    def __init__(self, adj_x, **kwargs):
        """
        Arguement:
            graph_input: can be directed or undirected graph create with networkx
        """
        super(mgcgraph_adj_m2, self).__init__()
        self.adj_x = adj_x
        
    def mgcgraph_plt(self, **kwargs):
        """
        Arguement:
            graph_in: input graph, if not provided, the same graph used to initiate the method is used
            axs_k: matplotlib axes
            centrality_algo: string with centraility algorithm, different algorithm available for both directed and undirected graph
                directed: ['degree', 'betweenness', 'eigenvector', 'harmonic', 'load', 'closeness']
                undirected: ['degree', 'betweenness', 'eigenvector', 'flow_closeness', 'harmonic', 'information']
            var_title: string to describe on plot tile
            node_size_range: a list of 2 numeric values
            node_att: node attribute available in the graph, only applicable if centrality_algo is set to None
            layout_type: string represent graph layout algorithm to be used, default is 'kamada_kawai_layout', other option
                ['circular_layout', 'random_layout', 'shell_layout', 'spring_layout', 'spectral_layout', 'kamada_kawai_layout', 'spiral_layout']
            
        """
        import matplotlib
        # user matplotlib default
        matplotlib.rcParams.update(matplotlib.rcParamsDefault)
        plt.style.use('default')
        import warnings
        warnings.filterwarnings('ignore')
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        
        # default values used when argument is not provided
        def_vals = {'adj_x' : self.adj_x,
                    'axs_k': None, 
                    'var_title': None, 
                    'local_legend': True, 
                    'fig': None, 
                    'local_colorbar': True }

        for k, v in def_vals.items():
            kwargs.setdefault(k, v)

        adj_x = kwargs['adj_x']
        axs_k = kwargs['axs_k']
        var_title = kwargs['var_title']
        local_legend = kwargs['local_legend']
        fig = kwargs['fig']
        local_colorbar = kwargs['local_colorbar']
        
        
        if axs_k is None:
            fig, axs_k = plt.subplots(figsize=(10,8))

#         adj_x = nx.adjacency_matrix(graph_in).todense()
        im = axs_k.imshow(adj_x, cmap='Blues', interpolation='nearest')

        axs_k.set_title("{} edge: {}, node: {}".format(var_title, adj_x.sum(), adj_x.shape[0]))
        # Turn spines off and create white grid.
        for edge, spine in axs_k.spines.items():
            spine.set_visible(False)

        axs_k.set_xticks(np.arange(adj_x.shape[1]+1)-.5, minor=True)
        axs_k.set_yticks(np.arange(adj_x.shape[0]+1)-.5, minor=True)
#         axs_k.grid(which="minor", color="g", linestyle='-', linewidth=1)
        axs_k.tick_params(which="minor", bottom=False, left=False)        
        if local_colorbar:
#             if nx.is_directed(graph_in):
#             # set alpha value for each edge
#                 M = graph_in.number_of_edges()
#                 if M > 0:
#                     divider = make_axes_locatable(axs_k)
#                     cax = divider.append_axes('right', size='5%', pad=0.05)
#                     plt.colorbar(im, cax=cax, orientation='vertical').set_label(label='weight',size=16)
#             else:
#             # define the bins and normalize

            if adj_x.sum() > 0:
                bounds = np.linspace(np.min(adj_x), np.max(adj_x), 11)
                norm = matplotlib.colors.BoundaryNorm(bounds, plt.cm.Blues.N)

                # create a second axes for the colorbar
                divider = make_axes_locatable(axs_k)
                cax = divider.append_axes('right', size='5%', pad=0.05)
                cbar = matplotlib.colorbar.ColorbarBase(cax, cmap=plt.cm.Blues, ticks=bounds, spacing='proportional',
                                                 boundaries=bounds).set_label(label='weight',size=16)