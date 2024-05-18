# Deep Learning for Epidemic Data Modeling and Analysis

### Updates
* Updated the interface. Now we can visualize the graph data and also the simulations using a web app based on streamlit and pyvis. 05/18/2024

        Steps:
            python setup.py install
            pip install pytorch_geometric
            
            Use python -m streamlit run interface/app.py to activate the interface

  
* Merged code and updated transformation module. 05/17/2024
* Updated SIR simulation on graphs(See examples/data_simulation.ipynb for more details). 05/06/2024
* Use examples/forecast_task.ipynb to try epidemic models as well as other baselines (STAN is currently not available). 05/06/2024