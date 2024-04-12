
class Dataset():
    def __init__(   self,
                    x = None,
                    states = None,
                    y = None,
                    edge_index = None,
                    edge_attr = None,
                 ):
        
        self.x = x # N*D; L*D; L*N*D; 
        self.y = y # N*1; L*1; L*N*1
        self.edge_index = edge_index # None; 2*Links; L*2*Links
        self.edge_attr = edge_attr # same as edge_index
        self.states = states # same as x

    def download(self):
        """Download selected files of the dataset."""

    def save(self):
        """Save current dataset."""

    def get_slice(self, timestamp):
        """Get a slice of graph or time series"""
    

