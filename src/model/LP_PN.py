import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from classifier.base import BASE




class Auto_Lambda(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc = nn.Linear(300* 2, 600 * 2)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(600 * 2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, support, center):
        ipt = torch.cat([support, center], dim=-1)
        output = self.fc(ipt)
        output = self.dropout(output)
        output = F.relu(output)
        output = self.fc2(output)
        output = self.sigmoid(output)
        return output

#Graph sigma Generator
class Graph_weight(nn.Module):
    def __init__(self):
        super(Graph_weight, self).__init__()

        self.layer1 = nn.Sequential(
                        nn.Conv2d(75,75,kernel_size=3,padding=1),
                        nn.BatchNorm2d(75),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=2, padding=1))
        self.layer2 = nn.Sequential(
                        nn.Conv2d(75,1,kernel_size=3,padding=1),
                        nn.BatchNorm2d(1),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=2, padding=1))

        self.fc3 = nn.Linear(4, 16)
        self.fc4 = nn.Linear(16, 1)



    def forward(self, x, rn):
        

        x = x.view(-1,75,2,2)
    
        out = self.layer1(x)
        out = self.layer2(out)
        
       
        out = out.view(out.size(0),-1) 
        out = F.relu(self.fc3(out))
        out = self.fc4(out) 

        out = out.view(out.size(0),-1) 
        
        return out
    
class Attention_Scores(nn.Module):
    def __init__(self,args):
        super().__init__()

        self.args=args
        self.hidden_size = 300
        self.conv1 = nn.Conv2d(1, 32, (args.shot, 1), padding=(args.shot // 2, 0))
        self.conv2 = nn.Conv2d(32, 64, (args.shot, 1), padding=(args.shot // 2, 0))
        self.conv_final = nn.Conv2d(64, 1, (args.shot, 1), stride=(args.shot, 1))
        self.drop = nn.Dropout()
        self.relu=torch.nn.ReLU()
        

    def forward(self,XS):
        fea_att_score = XS.view(self.args.way, 1, self.args.shot, self.hidden_size) 
        fea_att_score = self.relu(self.conv1(fea_att_score)) 
        fea_att_score = self.relu(self.conv2(fea_att_score)) 
        fea_att_score = self.drop(fea_att_score)
        fea_att_score = self.conv_final(fea_att_score) 
       
        fea_att_score = self.relu(fea_att_score)
        
        fea_att_score = fea_att_score.view(self.args.way, self.hidden_size).unsqueeze(1) 
        return fea_att_score
         
        
     
#Label propagation for prototypical network
class LP_PN(nn.Module):
    
    def __init__(self, ebd_dim, args):
        super(LP_PN, self).__init__()

        self.args = args
        self.hidden_size = 300
        self.mu=1
        if   args.rn == 300:   #fixed alpha
            self.alpha = torch.tensor([args.alpha], requires_grad=False).cuda(args.cuda)
        elif args.rn== 30:    # learned alpha
            self.alpha = nn.Parameter(torch.tensor([args.alpha]).cuda(args.cuda), requires_grad=True)

        self.auto_lambda = Auto_Lambda()
        self.graph_weight = Graph_weight()
        self.attention_scores=Attention_Scores(args)
  
    def _compute_l2_attention(self, XS, XQ,score):
            '''
                Compute the pairwise l2 distance
            '''
            
            diff = XS.unsqueeze(0) - XQ.unsqueeze(1)
      
            a=torch.pow(diff, 2)
            b=score.view(self.args.way,self.hidden_size).t().unsqueeze(0)
         
            dist=torch.matmul(a,b)
            dist=torch.sqrt(dist)
            dist=torch.sum(dist,dim=2)
    
            return dist




    def _compute_prototype(self, XS, YS):
        '''
            Compute the prototype for each class by averaging over the ebd.
        '''
        # sort YS to make sure classes of the same labels are clustered together
        sorted_YS, indices = torch.sort(YS)
        sorted_XS = XS[indices]

        prototype = []
        for i in range(self.args.way):
            prototype.append(torch.mean(
                sorted_XS[i*self.args.shot:(i+1)*self.args.shot], dim=0,
                keepdim=True))

        prototype = torch.cat(prototype, dim=0)

        return prototype
    
    def reidx_y(self, YS, YQ):
        '''
            Map the labels into 0,..., way
        '''
        unique1, inv_S = torch.unique(YS, sorted=True, return_inverse=True)
        unique2, inv_Q = torch.unique(YQ, sorted=True, return_inverse=True)

        if len(unique1) != len(unique2):
            raise ValueError(
                'Support set classes are different from the query set')

        if len(unique1) != self.args.way:
            raise ValueError(
                'Support set classes are different from the number of ways')

        if int(torch.sum(unique1 - unique2).item()) != 0:
            raise ValueError(
                'Support set classes are different from the query set classes')

        Y_new = torch.arange(start=0, end=self.args.way, dtype=unique1.dtype,
                device=unique1.device)

        return Y_new[inv_S], Y_new[inv_Q]

    def forward(self,XS, YS, XQ, YQ,query_data=None):
        """
            inputs are preprocessed
        """
       
        
        eps = np.finfo(float).eps
        emb_all = torch.cat((XS,XQ), 0)
        N, d    = emb_all.shape[0], emb_all.shape[1]

        #Graph Generator
        if self.args.rn in [30,300]:
            self.sigma   = self.graph_weight(emb_all, self.args.rn)
            
            ## W
            emb_all = emb_all / (self.sigma+eps) # N*d
            emb1    = torch.unsqueeze(emb_all,1) # N*1*d
            emb2    = torch.unsqueeze(emb_all,0) # 1*N*d
            W       = ((emb1-emb2)**2).mean(2)   # N*N*d -> N*N
            W       = torch.exp(-W/2)

        
        if self.args.g>0:
            topk, indices = torch.topk(W, self.args.g)
            mask = torch.zeros_like(W)
            mask = mask.scatter(1, indices, 1)
            mask = ((mask+torch.t(mask))>0).type(torch.float32)   
           
            W    = W*mask

        # normalize
        D       = W.sum(0)
        D_sqrt_inv = torch.sqrt(1.0/(D+eps))
        D1      = torch.unsqueeze(D_sqrt_inv,1).repeat(1,N)
        D2      = torch.unsqueeze(D_sqrt_inv,0).repeat(N,1)
        S       = D1*W*D2
        

        # one-hot
        unique1, inv_S = torch.unique(YS, sorted=True, return_inverse=True)
        unique2, inv_Q = torch.unique(YQ, sorted=True, return_inverse=True)
        Y_new = torch.arange(start=0, end=self.args.way, dtype=unique1.dtype,
                device=unique1.device)
        


        YS_n=Y_new[inv_S]
        YQ_n=Y_new[inv_Q]
   
        one_hot_YS= torch.zeros(len(YS_n), self.args.way).cuda(self.args.cuda)
        one_hot_YS.scatter_(1, torch.unsqueeze(YS_n, 1), 1)
        one_hot_YQ= torch.zeros(len(YQ_n), self.args.way).cuda(self.args.cuda)
        one_hot_YQ.scatter_(1, torch.unsqueeze(YQ_n, 1), 1)



        ys = one_hot_YS
        yu = torch.zeros(self.args.way*self.args.query, self.args.way).cuda(self.args.cuda)
        y  = torch.cat((ys,yu),0)
        F  = torch.matmul(torch.inverse(torch.eye(N).cuda(self.args.cuda)-self.alpha*S+eps), y)
        Fq = F[self.args.way*self.args.shot:, :] 
        

        #attention_metric
        fea_att_score=self.attention_scores(XS)

        YS, YQ = self.reidx_y(YS, YQ)
        _,topk_index=torch.topk(Fq,k=self.args.k,dim=0)
        
        au_query=XQ[torch.t(topk_index)]
        
        
        au_query_mean=torch.mean(au_query,dim=1)
       
        prototype = self._compute_prototype(XS, YS)
        
        lambda_p = self.auto_lambda(prototype, au_query_mean)
        au_prototype=(1. - lambda_p) * prototype + lambda_p *au_query_mean

        
        pred=-self._compute_l2_attention(au_prototype,XQ,fea_att_score)
        
        
        ce = nn.CrossEntropyLoss().cuda(self.args.cuda)
        
        gt = torch.argmax(torch.cat((one_hot_YS, one_hot_YQ), 0), 1)
        
        loss_lp = ce(F, gt)

        loss=ce(pred,YQ)+loss_lp*self.mu

        acc = BASE.compute_acc(pred, YQ)
    
    
        if query_data is not None:
            y_hat = torch.argmax(pred, dim=1)
            X_hat = query_data[y_hat != YQ]
            return acc, loss, X_hat

        return acc, loss
    


        



