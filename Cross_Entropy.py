import torch
import utils as u

class Cross_Entropy(torch.nn.Module):
    """docstring for Cross_Entropy"""
    def __init__(self, args, dataset):
        super().__init__()
        weights = torch.tensor(args.class_weights).to(args.device)

        self.weights = self.dyn_scale(args.task, dataset, weights)
        
    
    def dyn_scale(self,task,dataset,weights):
        # if task == 'link_pred':  commented to have a 1:1 ratio

        #     '''
        #     when doing link prediction there is an extra weighting factor on the non-existing
        #     edges
        #     '''
        #     tot_neg = dataset.num_non_existing
        #     def scale(labels):
        #         cur_neg = (labels == 0).sum(dtype = torch.float)
        #         out = weights.clone()
        #         out[0] *= tot_neg/cur_neg
        #         return out
        # else:
        #     def scale(labels):
        #         return weights
        def scale(labels):
            return weights
        return scale
    

    def logsumexp(self,logits):
        m,_ = torch.max(logits,dim=1)
        m = m.view(-1,1)
        sum_exp = torch.sum(torch.exp(logits-m),dim=1, keepdim=True)
        return m + torch.log(sum_exp)
    
    def forward(self,logits,labels):
        '''
        logits is a matrix M by C where m is the number of classifications and C are the number of classes
        labels is a integer tensor of size M where each element corresponds to the class that prediction i
        should be matching to
        '''
        labels = labels.view(-1,1)
        alpha = self.weights(labels)[labels].view(-1,1)
        loss = alpha * (- logits.gather(-1,labels) + self.logsumexp(logits))
        return loss.mean()

if __name__ == '__main__':
    dataset = u.Namespace({'num_non_existing': torch.tensor(10)})
    args = u.Namespace({'class_weights': [1.0,1.0],
                        'task': 'no_link_pred'})
    labels = torch.tensor([1,0])
    ce_ref = torch.nn.CrossEntropyLoss(reduction='sum')
    ce = Cross_Entropy(args,dataset)
    # print(ce.weights(labels))
    # print(ce.weights(labels))
    logits = torch.tensor([[1.0,-1.0],
                           [1.0,-1.0]])
    logits = torch.rand((5,2))
    labels = torch.randint(0,2,(5,))
    print(ce(logits,labels)- ce_ref(logits,labels))
    exit()
    ce.logsumexp(logits)
    # print(labels)
    # print(ce.weights(labels))
    # print(ce.weights(labels)[labels])
    x = torch.tensor([0,1])
    y = torch.tensor([1,0]).view(-1,1)
    # idx = torch.stack([x,y])
    # print(idx)  
    # print(idx)
    print(logits.gather(-1,y))
    # print(logits.index_select(0,torch.tensor([0,1])))