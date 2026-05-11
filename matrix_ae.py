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

    

class AE16to128(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 16)   
        )        

    def forward(self, x):
        z = self.encoder(x)   # Encode input to latent space
        x_hat = self.decoder(z)  # Reconstruct input
        return x_hat, z
    

list_seed=[16,32,48]  
list_auto=[128] 
for k in range(len(list_auto)):

    model = AE16to128()
    for j in range(0,3):
        set_seed(list_seed[j])
        df=pd.read_csv("bidb.csv")
        df3=pd.read_csv("bidb_dataset_not_in_c2db.csv")
        df = df[~df["mat_name"].isin(df3["mat_name"])]
        df=df.reset_index()
        df=df[["mat_name","stable","mono_name","param1","param2","param3", "param4","param5", "param6","param7"]]
        print(len(df))
        for i in range(len(df)):
            #param8=0
            param1=df['param1'][i]
            param2=df['param2'][i]
            param3=df['param3'][i]
            param4=df['param4'][i]
            #param8=df['delta_z'][i]
            param5=df['param5'][i]
            param6=df['param6'][i]
            param7=df['param7'][i]
            #param8=0
            if df['param5'][i]==1:
                param5=1
                param8=0.25
            else:
                param5=-1
                param8=0.9
            B=np.array([[param1,param2,0,param6],
                    [param3,param4,0,param7],
                    [0,0,param5,param8],
                    [0,0,0,0]])
            
            if i>=1:
                A=np.vstack([A,B.flatten()])
            else:
                A=B.flatten()
                A=A.reshape(1,16)
        
        unique_row=np.unique(A,axis=0)
        print(unique_row.shape,1)
        unique_row = torch.tensor(unique_row, dtype=torch.float32)
        dataset = TensorDataset(unique_row)
        dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        epochs = 100
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


        df=pd.read_csv("bidb.csv")
        df3=pd.read_csv("bidb_dataset_not_in_c2db.csv")
        df = df[~df["mat_name"].isin(df3["mat_name"])]
        df=df.reset_index()
        df=df.reset_index()
        A=0
        df=df[["mat_name","stable","mono_name","param1","param2","param3", "param4","param5", "param6","param7"]]
        for i in range(len(df)):
            #param8=0
            param1=df['param1'][i]
            param2=df['param2'][i]
            param3=df['param3'][i]
            param4=df['param4'][i]
            #param8=df['delta_z'][i]
            param5=df['param5'][i]
            param6=df['param6'][i]
            param7=df['param7'][i]
            #param8=0
            if df['param5'][i]==1:
                param5=1
                param8=0.25
            else:
                param5=-1
                param8=0.9
            B=np.array([[param1,param2,0,param6],
                    [param3,param4,0,param7],
                    [0,0,param5,param8],
                    [0,0,0,0]])
            
            if i>=1:
                A=np.vstack([A,B.flatten()])
            else:
                A=B.flatten()
                A=A.reshape(1,16)
        A_tensor = torch.tensor(A, dtype=torch.float32)
        with torch.no_grad():
            z_all = model.encoder(A_tensor)

        list_emb_feat = [f"m_{k}" for k in range(len(z_all[0]))]
        new_df = pd.DataFrame(z_all, columns=list_emb_feat)
        final_df = pd.concat([df, new_df], axis=1)
        final_df.to_csv(f"cgcnn/embeddings/matrix_ae_final_constant_dz_{list_auto[k]}_{j}.csv", index=False)
        print(f"Embeddings saved to matrix_ae_final_constant_dz_{list_auto[k]}_{j}.csv")