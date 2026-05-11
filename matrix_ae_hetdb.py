import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import random


def set_seed(seed: int = 42) -> None:
    """Sets the random seed for reproducibility across multiple libraries."""
    np.random.seed(seed)  # NumPy seed
    random.seed(seed)    # Python's built-in random module seed
    torch.manual_seed(seed) # PyTorch CPU seed
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # PyTorch CUDA seed for current GPU
        torch.cuda.manual_seed_all(seed) # PyTorch CUDA seed for all GPUs

    # Ensure deterministic behavior for CuDNN backend operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    

class AE6to128(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)   
        )        

    def forward(self, x):
        z = self.encoder(x)   # Encode input to latent space
        x_hat = self.decoder(z)  # Reconstruct input
        return x_hat, z
    

list_seed=[16]  
list_auto=[256] 
for k in range(len(list_auto)):
    model = AE6to128()
    for j in range(0,1):
        set_seed(list_seed[j])
        df=pd.read_csv("hetdb.csv")
        for i in range(len(df)):        
            param1=df['j_twist_angle'][i]
    

            B=np.array([param1])
            if i>=1:
                A=np.vstack([A,B.flatten()])
            else:
                A=B.flatten()
                A=A.reshape(1,1)
        unique_row=np.unique(A,axis=0)
        print(unique_row.shape,1)
        unique_row = torch.tensor(unique_row, dtype=torch.float32)
        dataset = TensorDataset(unique_row)
        dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        epochs = 500
        for epoch in range(epochs):
            total_loss = 0
            for batch in dataloader:
                x = batch[0]             
                x_hat, z = model(x)
                loss = criterion(x_hat,x)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch+1}/{epochs}, Loss={total_loss/len(dataloader):.10f}")


        df=pd.read_csv("hetdb.csv")
       
        print(len(df))
        for i in range(len(df)):
       

            param1=df['j_twist_angle'][i]
            

            B=np.array([param1])

            if i>=1:
                A=np.vstack([A,B.flatten()])
                
            else:
                A=B.flatten()
                A=A.reshape(1,1)
        A_tensor = torch.tensor(A, dtype=torch.float32)
        with torch.no_grad():
            z_all = model.encoder(A_tensor)

        print(z_all)
   
        list_emb_feat = [f"m_{k}" for k in range(128)]
 
        new_df = pd.DataFrame(z_all, columns=list_emb_feat)
        final_df = pd.concat([df, new_df], axis=1)
        final_df.to_csv(f"embeddings/matrix_ae_prop_new_hetdb_{list_auto[k]}.csv", index=False)
        print(len(final_df)) 
        print(f"Embeddings saved to matrix_ae_prop_hetdb_{list_auto[k]}.csv")