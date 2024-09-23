
#---------------------------------------------------------------------
# 导入所需的库
import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network
import torch
import io
import pandas as pd
import urllib.request
import os

from sims import get_random_graph
from NetSIR import NetSIR


# 设置全局属性
st.set_page_config(
    page_title='EpiLearn',
    page_icon=' ',
    layout='wide'
)

if 'initialized' not in st.session_state:
    st.session_state['initialized'] = False

if not st.session_state['initialized']:
    if not os.path.exists('./example.pt'):
        url_feature = "https://drive.google.com/uc?export=download&id=1512vgZWrboh3Pm7OhQtFu7hxdEaS9d9I"
        urllib.request.urlretrieve(url_feature, './example.pt')
    data = torch.load('./example.pt')
    st.session_state['initialized'] = True


#-------------------------------------------------------
with st.sidebar:
    st.title('Controls')
    st.markdown('---')
    upload_tab, simulation_tab = st.tabs(['Select From File', 'Random Simulation'])
    if 'tab1_clicked' not in st.session_state:
        st.session_state['tab1_clicked'] = True

    if 'tab2_clicked' not in st.session_state:
        st.session_state['tab2_clicked'] = False

    # select local file
    with upload_tab:
        if st.button("Change to Upload"):
            st.session_state['tab1_clicked'] = True
            st.session_state['tab2_clicked'] = False
        # import ipdb; ipdb.set_trace()
        if st.session_state['tab1_clicked']:
            data = None
            @st.cache_data
            def get_example():
                if not os.path.exists('./example.pt'):
                    url_feature = "https://drive.google.com/uc?export=download&id=1512vgZWrboh3Pm7OhQtFu7hxdEaS9d9I"
                    urllib.request.urlretrieve(url_feature, './example.pt')
                data = torch.load('./example.pt')
                return data
            if st.button("load_example"):
                data = get_example()
            # upload button
            uploaded_file = st.file_uploader("Upload .pt File", type=["pt"])
            # import ipdb; ipdb.set_trace()
            if uploaded_file is not None:
                # 读取上传的文件内容
                # import ipdb; ipdb.set_trace()
                data = torch.load(io.BytesIO(uploaded_file.read()))
                # 显示文件内容
                st.write(f"File Contents: {list(data.keys())}")
                # for k in data.keys():
                #     st.write(f"{k}: {data[k].shape}")
                # import ipdb; ipdb.set_trace()
            
            if data is None:
                data = torch.load("interface/interface_example.pt")
        
        

    # generate random simulation
    with simulation_tab:
        if st.button("Change to Simulate"):
            st.session_state['tab1_clicked'] = False
            st.session_state['tab2_clicked'] = True

        if st.session_state['tab2_clicked']:
            num_nodes = st.number_input('Number of Nodes', value=5, step=1)
            p_connect = st.number_input('Probability of Connectivity', value=0.15, min_value=0.0, max_value=1.0)
            horizon = st.number_input('Future Pridiction Steps', value=10, step=1)
            st.text("Red denotes Infected\nOrange denotes Suspected\nBlue denotes Recovered")
            infected = st.multiselect('Select initial nodes to be infected', options=list(range(num_nodes)))
            p_infect = st.number_input('Infection Probability', value=0.05, min_value=0.0, max_value=1.0)
            p_recover = st.number_input('Recovery Probability', value=0.05, min_value=0.0, max_value=1.0)

            @st.cache_data
            def get_simulation(n_nodes, connect, pred_steps, infect_rate, recover_rate, infected):
                initial_graph = get_random_graph(num_nodes=n_nodes, connect_prob=connect)
                initial_states = torch.zeros(n_nodes,3) # [S,I,R]
                initial_states[:, 0] = 1.0
                for i in infected:
                    initial_states[i, 0] = 0
                    initial_states[i, 1] = 1
                model = NetSIR(num_nodes=n_nodes, horizon=pred_steps, infection_rate=infect_rate, recovery_rate=recover_rate) # infection_rate, recover_rate, fixed_population
                preds = model(initial_states, initial_graph)
                # import ipdb; ipdb.set_trace()
                states = preds.argmax(2).detach().numpy()

                return initial_graph.long(), torch.LongTensor(states)
            
            sim_graph, sim_states = get_simulation(num_nodes, p_connect, horizon, p_infect, p_recover, infected)
            # import ipdb; ipdb.set_trace()

            data = {'targets': torch.ones(horizon, num_nodes).float(), 'graph': sim_graph, 'states': sim_states}

    # import ipdb; ipdb.set_trace()
    # timestep select bar
    if 'data' in locals().keys():
        targets = data['targets'].numpy()
        timesteps = targets.shape[0]
        graphs = torch.LongTensor(data['graph']).unsqueeze(0).repeat(timesteps, 1, 1)
        try:
            edge_weights = data['dynamic_graph'].numpy()
        except:
            edge_weights = torch.ones_like(graphs)
        locations = graphs.shape[0]
        try:
            states = data['states'].numpy()
        except:
            states = torch.zeros((timesteps, locations)).numpy()
        
        cur_time = st.slider('Current Time Step', 0, timesteps-1, 0, 1)
    
    traced = st.multiselect('Select ids of nodes to be traced...', options=list(range(graphs.shape[1])))
    plot_time_series = st.multiselect('Select ids of nodes to be plot', options=list(range(graphs.shape[1])))
        
        
#-------------------------------------------------------

def get_net():
    nt = Network( neighborhood_highlight=True, )
    return nt

# Main
st.title('Interactive Visualization of Epidemic Spreading')

main_tab = st.tabs(['Visualization'])

with main_tab[0]:

    col1, col2 = st.columns(2)
    color_map = {0: 'orange', 1: 'red', 2: 'blue', 3:'green'}
    nt = get_net()
    with col1:
        
        if 'targets' in dir() and 'graphs' in dir() and 'states' in dir():
            # filter_menu=True,select_menu=True
            
            assert len(targets.shape) == 2 and len(graphs.shape) == 3 and len(states.shape) == 2
            weights, edge_weight, graph, state = targets[cur_time], edge_weights[cur_time], graphs[cur_time], states[cur_time]

            num_locs = list(range(len(graph)))
            
            for i in num_locs:
                for j in num_locs:
                    if graph[i, j] > 0 and i!=j: # and (i not in nt.node_ids or j not in nt.node_ids)
                        i_color = 'green' if i in traced else color_map[state[i]]
                        j_color = 'green' if j in traced else color_map[state[j]]
                            
                        if i not in nt.node_ids:
                            nt.add_node(i, i, title=str(i), color=i_color, value = int(weights[i]))# 
                        if j not in nt.node_ids:
                            nt.add_node(j, j, title=str(j), color=j_color, value = int(weights[j]))# 

                        nt.add_edge(i, j, value=int(edge_weight[i,j]))# 


        # import ipdb; ipdb.set_trace()
        # show
        # nt.toggle_physics(False)
        # nt.toggle_stabilization(False)
        nt.set_edge_smooth('discrete')
        nt.repulsion(node_distance=500, central_gravity=0.2, spring_length=800, spring_strength=0.05, damping=0.09)
        nt.show('example.html', notebook=False)
        # nt.show_buttons(filter_=['physics'])
        components.html(open("example.html", 'r', encoding='utf-8').read(), height=600)
    
    with col2:
        for node_id in plot_time_series:
            st.markdown(f"## Node ID: {node_id}")
            y = targets[:, node_id]
            # fig = ff.create_distplot([y], ['Infected Count'])
            # # Plot!
            # st.plotly_chart(fig, use_container_width=True)
            st.line_chart(pd.DataFrame({"Infected Count": y}), y=["Infected Count"])



    




