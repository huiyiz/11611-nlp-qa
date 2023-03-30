import torch.nn as nn
class Network(nn.Module):

    def __init__(self, dropout):

        super(Network, self).__init__()

        input_size = 128
        output_size = 1

        self.model = nn.Sequential(
            #layer 1
            nn.Linear(input_size, input_size * 2),
            nn.BatchNorm1d(input_size * 2),
            nn.LeakyReLU(0.2),
            

            #layer 2
            nn.Dropout(dropout),
            nn.Linear(input_size * 2, input_size * 4),
            nn.BatchNorm1d(input_size * 4),
            nn.LeakyReLU(0.2),

            #layer 3
            nn.Dropout(dropout),
            nn.Linear(input_size * 4, input_size * 8),
            nn.BatchNorm1d(input_size * 8),
            nn.LeakyReLU(0.2),


            #layer 4
            nn.Dropout(dropout),
            nn.Linear(input_size * 8, input_size * 4),
            nn.BatchNorm1d(input_size * 4),
            nn.LeakyReLU(0.2),


            #layer 5
            nn.Dropout(dropout),
            nn.Linear(input_size * 4, input_size * 2),
            nn.BatchNorm1d(input_size * 2),
            nn.LeakyReLU(0.2),

            #layer 6
            nn.Dropout(dropout),
            nn.Linear(input_size * 2, input_size), 
            nn.BatchNorm1d(input_size),
            nn.LeakyReLU(0.2),

            #layer 7
            nn.Dropout(dropout),
            nn.Linear(input_size, input_size // 2), 
            nn.BatchNorm1d(input_size // 2),
            nn.LeakyReLU(0.21),
            
            #layer 8
            nn.Dropout(dropout),
            nn.Linear(input_size // 2, input_size // 4),
            nn.BatchNorm1d(input_size // 4),   
            nn.LeakyReLU(0.2),

            #layer 9
            nn.Dropout(dropout),
            nn.Linear(input_size // 4, input_size // 8), 
            nn.BatchNorm1d(input_size // 8),  
            nn.LeakyReLU(0.2),

            #layer 10
            nn.Linear(input_size // 8, output_size),  
            nn.Sigmoid(),
        )      

    def forward(self, x):
        out = self.model(x)

        return out