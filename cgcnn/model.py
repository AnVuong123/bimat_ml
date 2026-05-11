from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvLayer(nn.Module):
    """
    Convolutional operation on graphs
    """
    def __init__(self, atom_fea_len, nbr_fea_len):
        """
        Initialize ConvLayer.

        Parameters
        ----------

        atom_fea_len: int
          Number of atom hidden features.
        nbr_fea_len: int
          Number of bond features.
        """
        super(ConvLayer, self).__init__()
        self.atom_fea_len = atom_fea_len
        self.nbr_fea_len = nbr_fea_len
        self.fc_full = nn.Linear(2*self.atom_fea_len+self.nbr_fea_len,2*self.atom_fea_len)
        self.sigmoid = nn.Sigmoid()
        self.softplus1 = nn.Softplus()
        self.bn1 = nn.BatchNorm1d(2*self.atom_fea_len)
        self.bn2 = nn.BatchNorm1d(self.atom_fea_len)
        self.softplus2 = nn.Softplus()

    def forward(self, atom_in_fea, nbr_fea, nbr_fea_idx):
        """
        Forward pass

        N: Total number of atoms in the batch
        M: Max number of neighbors

        Parameters
        ----------

        atom_in_fea: Variable(torch.Tensor) shape (N, atom_fea_len)
          Atom hidden features before convolution
        nbr_fea: Variable(torch.Tensor) shape (N, M, nbr_fea_len)
          Bond features of each atom's M neighbors
        nbr_fea_idx: torch.LongTensor shape (N, M)
          Indices of M neighbors of each atom

        Returns
        -------

        atom_out_fea: nn.Variable shape (N, atom_fea_len)
          Atom hidden features after convolution

        """
        # TODO will there be problems with the index zero padding?
        N, M = nbr_fea_idx.shape
        # convolution
        atom_nbr_fea = atom_in_fea[nbr_fea_idx, :]
        total_nbr_fea = torch.cat(
            [atom_in_fea.unsqueeze(1).expand(N, M, self.atom_fea_len),
             atom_nbr_fea, nbr_fea], dim=2)
        total_gated_fea = self.fc_full(total_nbr_fea)
        total_gated_fea = self.bn1(total_gated_fea.view(-1, self.atom_fea_len*2)).view(N, M, self.atom_fea_len*2)
        nbr_filter, nbr_core = total_gated_fea.chunk(2, dim=2)
        nbr_filter = self.sigmoid(nbr_filter)
        nbr_core = self.softplus1(nbr_core)
        nbr_sumed = torch.sum(nbr_filter * nbr_core, dim=1)
        nbr_sumed = self.bn2(nbr_sumed)
        out = self.softplus2(atom_in_fea + nbr_sumed)
        return out


class CrystalGraphConvNet(nn.Module):
    """
    Create a crystal graph convolutional neural network for predicting total
    material properties.
    """
    def __init__(self, orig_atom_fea_len, nbr_fea_len,
                 atom_fea_len=64, n_conv=3, h_fea_len=128, n_h=1,
                 classification=False):
        """
        Initialize CrystalGraphConvNet.

        Parameters
        ----------

        orig_atom_fea_len: int
          Number of atom features in the input.
        nbr_fea_len: int
          Number of bond features.
        atom_fea_len: int
          Number of hidden atom features in the convolutional layers
        n_conv: int
          Number of convolutional layers
        h_fea_len: int
          Number of hidden features after pooling
        n_h: int
          Number of hidden layers after pooling
        """
        super(CrystalGraphConvNet, self).__init__()
        self.classification = classification
        self.embedding = nn.Linear(orig_atom_fea_len, atom_fea_len)

        self.convs = nn.ModuleList([ConvLayer(atom_fea_len=atom_fea_len,nbr_fea_len=nbr_fea_len) for _ in range(n_conv)])

        self.conv_to_fc = nn.Linear(atom_fea_len, h_fea_len)
        self.conv_to_fc_softplus = nn.Softplus()
        if n_h > 1:
            self.fcs = nn.ModuleList([nn.Linear(h_fea_len, h_fea_len) for _ in range(n_h-1)])
            self.softpluses = nn.ModuleList([nn.Softplus() for _ in range(n_h-1)])
        if self.classification:
            self.fc_out = nn.Linear(h_fea_len, 2)
        else:
            self.fc_out = nn.Linear(h_fea_len, 1)
        if self.classification:
            # each sample normalize to sum = 1
            self.logsoftmax = nn.LogSoftmax(dim=1)
            self.dropout = nn.Dropout()

            
    @torch.no_grad()
    def encode(self, atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx,
               normalize: bool = False):
        
        self.eval()
        atom_fea = self.embedding(atom_fea)
        for conv_func in self.convs:
            atom_fea = conv_func(atom_fea, nbr_fea, nbr_fea_idx)
        crys_fea = self.pooling(atom_fea, crystal_atom_idx)
        crys_fea = self.conv_to_fc(self.conv_to_fc_softplus(crys_fea))
        crys_fea = self.conv_to_fc_softplus(crys_fea)
        if self.classification:
            crys_fea = self.dropout(crys_fea)
  
        crys_fea = F.normalize(crys_fea, dim=1)
     
        return crys_fea

      

    def forward(self, atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx):
        """
        Forward pass

        N: Total number of atoms in the batch
        M: Max number of neighbors
        N0: Total number of crystals in the batch

        Parameters
        ----------

        atom_fea: Variable(torch.Tensor) shape (N, orig_atom_fea_len)
          Atom features from atom type
        nbr_fea: Variable(torch.Tensor) shape (N, M, nbr_fea_len)
          Bond features of each atom's M neighbors
        nbr_fea_idx: torch.LongTensor shape (N, M)
          Indices of M neighbors of each atom
        crystal_atom_idx: list of torch.LongTensor of length N0
          Mapping from the crystal idx to atom idx

        Returns
        -------

        prediction: nn.Variable shape (N, )
          Atom hidden features after convolution

        """
        atom_fea = self.embedding(atom_fea)
        for conv_func in self.convs:
            atom_fea = conv_func(atom_fea, nbr_fea, nbr_fea_idx)
        crys_fea = self.pooling(atom_fea, crystal_atom_idx)
        crys_fea = self.conv_to_fc(self.conv_to_fc_softplus(crys_fea))
        crys_fea = self.conv_to_fc_softplus(crys_fea)
        if self.classification:
            crys_fea = self.dropout(crys_fea)
        if hasattr(self, 'fcs') and hasattr(self, 'softpluses'):
            for fc, softplus in zip(self.fcs, self.softpluses):
                crys_fea = softplus(fc(crys_fea))
        out = self.fc_out(crys_fea)
        if self.classification:
            out = self.logsoftmax(out)
        return out

    def pooling(self, atom_fea, crystal_atom_idx):
        """
        Pooling the atom features to crystal features

        N: Total number of atoms in the batch
        N0: Total number of crystals in the batch

        Parameters
        ----------

        atom_fea: Variable(torch.Tensor) shape (N, atom_fea_len)
          Atom feature vectors of the batch
        crystal_atom_idx: list of torch.LongTensor of length N0
          Mapping from the crystal idx to atom idx
        """
        assert sum([len(idx_map) for idx_map in crystal_atom_idx]) ==\
            atom_fea.data.shape[0]
        summed_fea = [torch.mean(atom_fea[idx_map], dim=0, keepdim=True)
                      for idx_map in crystal_atom_idx]
        return torch.cat(summed_fea, dim=0)

class HetmonoCrystalGraphConvNet(nn.Module):
    """
    CGCNN for Bilayer:
    - Graph A has its own atom feature dim & bond feature dim
    - Graph B has its own atom feature dim & bond feature dim
    - Encode A and B separately
    - Concatenate embeddings
    - Predict single target
    """

    def __init__(self,
                 orig_atom_fea_len,
                 orig_atom_fea_len2,
                 nbr_fea_len,
                 nbr_fea_len2,
                 atom_fea_len=64,
                 n_conv=3,
                 h_fea_len=128,
                 fusion_hidden=258,
                 classification=False,mono=False):

        super(HetmonoCrystalGraphConvNet, self).__init__()

        self.classification = classification

        # ===== Encoder 1 =====
        self.embedding = nn.Linear(orig_atom_fea_len, atom_fea_len)
        self.convs = nn.ModuleList(
            [ConvLayer(atom_fea_len, nbr_fea_len) for _ in range(n_conv)]
        )
        self.fc = nn.Linear(atom_fea_len, h_fea_len)

        self.mono_prop_embed = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU()
        )
        
       

        # ===== Encoder 2 =====
        self.embedding2 = nn.Linear(orig_atom_fea_len2, atom_fea_len)
        self.convs2 = nn.ModuleList(
            [ConvLayer(atom_fea_len, nbr_fea_len2) for _ in range(n_conv)]
        )
        self.fc2 = nn.Linear(atom_fea_len, h_fea_len)

        # ===== FUSION LAYER =====
        if mono==True:
            fusion_in = h_fea_len * 2 + 256+256 +256*2
        else:
            fusion_in = h_fea_len * 2 

        fusion_hidden=fusion_in

        self.fusion_fc = nn.Linear(fusion_in, fusion_hidden)
        self.fusion_act = nn.ReLU()

        # ===== OUTPUT LAYER =====
        if classification:
            self.fc_out = nn.Linear(fusion_hidden, 2)
            self.logsoftmax = nn.LogSoftmax(dim=1)
            self.dropout = nn.Dropout()
        else:
            self.fc_out = nn.Linear(fusion_hidden, 1)

    # ============================================================
    # encode 1 graph 
    # ============================================================
    def encode_grap2(self, atom, nbr, idx, crys_idx, embedding, convs,graph):
        atom = embedding(atom)
        i=1
        for conv in convs:
            atom = conv(atom, nbr, idx)
           
        crys_fea=self.pooling(atom, crys_idx)
        

      
        if graph==1:
            crys_fea = self.fc(crys_fea)
        else:
            crys_fea = self.fc2(crys_fea)

        return crys_fea
    def encode_graph(self, atom, nbr, idx, crys_idx, embedding, convs, graph):

        atom = embedding(atom)

        for conv in convs:
            atom = conv(atom, nbr, idx)


        crys_fea=self.pooling(atom, crys_idx)     # nếu chưa pooling thì vẫn là atom-level

        if graph == 1:
           crys_fea = self.fc(crys_fea)
        else:
           crys_fea = self.fc2(crys_fea)

        return crys_fea
        

    # ============================================================
    # forward() nhận HAI GRAPH
    # ============================================================
    def forward(self, atom, nbr, idx, crys_idx, atom2, nbr2, idx2, crys_idx2,s_vector,l_vector, mono_target1,mono_target2,mono=True):
        # ---- encode A graph ----
        emb = self.encode_graph(
            atom, nbr, idx, crys_idx,
            self.embedding, self.convs, 1
        )

        # ---- encode B graph ----
        emb2 = self.encode_graph(
            atom2, nbr2, idx2, crys_idx2,
            self.embedding2, self.convs2, 2
        )

        print(emb.shape,emb2.shape,mono_target1.shape,mono_target2.shape,s_vector.shape)
        if mono==True:
            fused = torch.cat([emb, emb2,mono_target1,mono_target2,s_vector], dim=1)
        else:
            fused = torch.cat([emb, emb2], dim=1)

        fused = self.fusion_act(self.fusion_fc(fused))

        # ---- output ----
        out = self.fc_out(fused)

        if self.classification:
            out = self.logsoftmax(out)

        return out

    # ============================================================
    # pooling
    # ============================================================
    def pooling(self, atom_fea, crystal_atom_idx):
        pooled = [torch.mean(atom_fea[idx], dim=0, keepdim=True)
                  for idx in crystal_atom_idx]
        return torch.cat(pooled, dim=0)
    
class BimonolayerCrystalGraphConvNet(nn.Module):
    """
    CGCNN for Bilayer:
    - Graph A has its own atom feature dim & bond feature dim
    - Graph B has its own atom feature dim & bond feature dim
    - Encode A and B separately
    - Concatenate embeddings
    - Predict single target
    """

    def __init__(self,
                 orig_atom_fea_len,
                 nbr_fea_len,
                 atom_fea_len=64,
                 n_conv=3,
                 h_fea_len=128,
                 fusion_hidden=257,
                 classification=False,mono=False):

        super(BimonolayerCrystalGraphConvNet, self).__init__()

        self.classification = classification

        # ===== Encoder A =====
        self.embedding = nn.Linear(orig_atom_fea_len, atom_fea_len)
        self.convs = nn.ModuleList(
            [ConvLayer(atom_fea_len, nbr_fea_len) for _ in range(n_conv)]
        )
        self.fc = nn.Linear(atom_fea_len, h_fea_len)
        self.config_fc = nn.Sequential(nn.Linear(128, h_fea_len), nn.ReLU())

        self.prop_fc = nn.Sequential(nn.Linear(1, h_fea_len), nn.ReLU())

     

        # ===== FUSION LAYER =====
        if mono==True:
            fusion_in = h_fea_len + 128 + 128
        else:
            fusion_in = h_fea_len + 128

        fusion_hidden=fusion_in
        
        self.fusion_fc1 = nn.Linear(fusion_in, fusion_hidden)
        self.fusion_act1 = nn.ReLU()

        # ===== OUTPUT LAYER =====
        if classification:
            self.fc_out = nn.Linear(fusion_hidden, 2)
            self.logsoftmax = nn.LogSoftmax(dim=1)
            self.dropout = nn.Dropout()
        else:
            self.fc_out = nn.Linear(fusion_hidden, 1)

    # ============================================================
    # encode 1 graph (A or B)
    # ============================================================
    def encode_graph(self, atom, nbr, idx, crys_idx, embedding, convs, fc):
        atom = embedding(atom)
        for conv in convs:
            atom = conv(atom, nbr, idx)

        # pooling per crystal
        crys_fea = self.pooling(atom, crys_idx)

        # FC after pooling
        crys_fea = F.softplus(fc(crys_fea))
        return crys_fea

    # ============================================================
    # forward() 
    # ============================================================
    def forward(self, atom, nbr, idx, crys_idx, config_vector,mono_bg,mono=False):
        


        # ---- encode A graph ----
        emb = self.encode_graph(
            atom, nbr, idx, crys_idx,
            self.embedding, self.convs, self.fc
        )

        config_vector = self.config_fc(config_vector)
        mono_bg = self.prop_fc(mono_bg)
        # ---- fuse ----
        if mono==True:
            fused = torch.cat([emb,config_vector,mono_bg], dim=1)
        else:
            fused = torch.cat([emb,config_vector], dim=1)
        fused = self.fusion_act1(self.fusion_fc1(fused))

        # ---- output ----
        out = self.fc_out(fused)

        if self.classification:
            out = self.logsoftmax(out)

        return out

    # ============================================================
    # pooling
    # ============================================================
    def pooling(self, atom_fea, crystal_atom_idx):
        pooled = [torch.mean(atom_fea[idx], dim=0, keepdim=True)
                  for idx in crystal_atom_idx]
        return torch.cat(pooled, dim=0)
    

class BiDBlayerCrystalGraphConvNet(nn.Module):
    """
    CGCNN for Bilayer:
    - Graph A has its own atom feature dim & bond feature dim
    - Graph B has its own atom feature dim & bond feature dim
    - Encode A and B separately
    - Concatenate embeddings
    - Predict single target
    """

    def __init__(self,
                 orig_atom_fea_len,
                 nbr_fea_len,
                 atom_fea_len=64,
                 n_conv=3,
                 h_fea_len=128,
                 fusion_hidden=129,
                 classification=False,mono=False):

        super(BiDBlayerCrystalGraphConvNet, self).__init__()

        self.classification = classification

        # ===== Encoder A =====
        self.embedding = nn.Linear(orig_atom_fea_len, atom_fea_len)
        self.convs = nn.ModuleList(
            [ConvLayer(atom_fea_len, nbr_fea_len) for _ in range(n_conv)]
        )
        self.fc = nn.Linear(atom_fea_len, h_fea_len)

     

        # ===== FUSION LAYER =====
        if mono==True:
            fusion_in = h_fea_len +1
        else:
            fusion_in = h_fea_len

        fusion_hidden=fusion_in
        
        self.fusion_fc1 = nn.Linear(fusion_in, fusion_hidden)
        self.fusion_act1 = nn.ReLU()

        # ===== OUTPUT LAYER =====
        if classification:
            self.fc_out = nn.Linear(fusion_hidden, 2)
            self.logsoftmax = nn.LogSoftmax(dim=1)
            self.dropout = nn.Dropout()
        else:
            self.fc_out = nn.Linear(fusion_hidden, 1)

    # ============================================================
    # encode 1 graph (A or B)
    # ============================================================
    def encode_graph(self, atom, nbr, idx, crys_idx, embedding, convs, fc):
        atom = embedding(atom)
        for conv in convs:
            atom = conv(atom, nbr, idx)

        # pooling per crystal
        crys_fea = self.pooling(atom, crys_idx)

        # FC after pooling
        crys_fea = F.softplus(fc(crys_fea))
        return crys_fea

    # ============================================================
    # forward() 
    # ============================================================
    def forward(self, atom, nbr, idx, crys_idx, mono_bg,mono=False):
        

        #atom, nbr, idx, crys = graph

        # ---- encode A graph ----
        emb = self.encode_graph(
            atom, nbr, idx, crys_idx,
            self.embedding, self.convs, self.fc
        )

       
        # ---- fuse ----
        if mono==True:
            fused = torch.cat([emb,mono_bg], dim=1)
        else:
            fused = torch.cat([emb], dim=1)
            
        fused = self.fusion_act1(self.fusion_fc1(fused))

        # ---- output ----
        out = self.fc_out(fused)

        if self.classification:
            out = self.logsoftmax(out)

        return out

    # ============================================================
    # pooling
    # ============================================================
    def pooling(self, atom_fea, crystal_atom_idx):
        pooled = [torch.mean(atom_fea[idx], dim=0, keepdim=True)
                  for idx in crystal_atom_idx]
        return torch.cat(pooled, dim=0)
    
class HetDBlayerCrystalGraphConvNet(nn.Module):
    """
    CGCNN for Bilayer:
    - Graph A has its own atom feature dim & bond feature dim
    - Graph B has its own atom feature dim & bond feature dim
    - Encode A and B separately
    - Concatenate embeddings
    - Predict single target
    """

    def __init__(self,
                 orig_atom_fea_len,
                 nbr_fea_len,
                 atom_fea_len=64,
                 n_conv=3,
                 h_fea_len=128,
                 fusion_hidden=130,n_h=1,
                 classification=False,mono=False):

        super(HetDBlayerCrystalGraphConvNet, self).__init__()

        self.classification = classification

        
        self.embedding = nn.Linear(orig_atom_fea_len, atom_fea_len)
        self.convs = nn.ModuleList(
            [ConvLayer(atom_fea_len, nbr_fea_len) for _ in range(n_conv)]
        )
        self.fc = nn.Linear(atom_fea_len, h_fea_len)

     

        # ===== FUSION LAYER =====
        if mono==True:
            fusion_in = h_fea_len + 2 
        else:
            fusion_in = h_fea_len 

        fusion_hidden=fusion_in
        self.fusion_fc1 = nn.Linear(fusion_in, fusion_hidden)
        self.fusion_act1 = nn.ReLU()

        # ===== OUTPUT LAYER =====
        if classification:
            self.fc_out = nn.Linear(fusion_hidden, 2)
            self.logsoftmax = nn.LogSoftmax(dim=1)
            self.dropout = nn.Dropout()
        else:
            self.fc_out = nn.Linear(fusion_hidden, 1)

    # ============================================================
    # encode 1 graph (A or B)
    # ============================================================
    def encode_graph(self, atom, nbr, idx, crys_idx, embedding, convs):
        atom = embedding(atom)
        for conv in convs:
            atom = conv(atom, nbr, idx)

        # pooling per crystal
        crys_fea = self.pooling(atom, crys_idx)

        # FC after pooling
        crys_fea = self.fc(crys_fea)
       
        #crys_fea = F.softplus(fc(crys_fea))
        return crys_fea

    # ============================================================
    # forward() 
    # ============================================================
    def forward(self, atom, nbr, idx, crys, mono_target1,mono_target2, mono=False):
        
        # ---- encode A graph ----
        emb = self.encode_graph(
            atom, nbr, idx, crys,
            self.embedding, self.convs
        )

       
        # ---- fuse ----
        if mono==True:
            fused = torch.cat([emb,mono_target1,mono_target2], dim=1)
        else:
            fused = torch.cat([emb], dim=1)
        fused = self.fusion_act1(self.fusion_fc1(fused))

        # ---- output ----
        out = self.fc_out(fused)

        if self.classification:
            out = self.logsoftmax(out)

        return out

    # ============================================================
    # pooling
    # ============================================================
    def pooling(self, atom_fea, crystal_atom_idx):
        pooled = [torch.mean(atom_fea[idx], dim=0, keepdim=True)
                  for idx in crystal_atom_idx]
        return torch.cat(pooled, dim=0)


