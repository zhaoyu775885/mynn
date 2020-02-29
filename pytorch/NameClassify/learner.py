import torch
import torch.nn as nn
import torch.optim as optim
from name_data import NameDataset
from network import RNNNaive, RNN

BATCH_SIZE = 1
INIT_LR = 1e-3
MOMENTUM = 0.9
L2_REG = 1e-5

class NameLearner():
    def __init__(self, dataset, network):
        # set device & build dataset
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        self.dataset = dataset
        self.net = network.to(self.device)

        # build dataloader
        self.trainloader = self._build_dataloader(BATCH_SIZE)

        # setup loss function
        self.criterion = self._loss_fn()

        # setup optimizatizer
        self.opt = self._setup_optimizer()
        self.lr_scheduler = self._setup_lr_scheduler()

    def _build_dataloader(self, batch_size):
        return self.dataset.build_dataloader(batch_size)
    
    def _loss_fn(self):
        return nn.CrossEntropyLoss()
    
    def _setup_optimizer(self):
        return optim.SGD(self.net.parameters(), lr=INIT_LR, momentum=MOMENTUM, weight_decay=L2_REG)
    
    def _setup_lr_scheduler(self):
        return torch.optim.lr_scheduler.MultiStepLR(self.opt, milestones=[10, 15, 20], gamma=0.1)
    
    def metrics(self, outputs, labels):
        _, predicted = torch.max(outputs, 1)
        correct = (predicted == labels).sum().item()
        loss = self.criterion(outputs, labels)
        accuracy = correct / labels.size(0)
        return accuracy, loss
    
    def train(self, n_epoch=25):
        for epoch in range(n_epoch):
            self.lr_scheduler.step()
            self.dataset.init_batch_loader()
            #init_hiddens = self.net.initHidden().to(self.device)

            pred_correct = 0
            for i, data in enumerate(self.trainloader):
                #feats, labels, lengths = data[0].to(self.device), data[1].to(self.device), data[2].to(self.device)
                feats, labels, lengths = data
                local_max_length = max(lengths)
                #hiddens = init_hiddens
                self.opt.zero_grad()
                '''
                for t in range(local_max_length):
                    inputs = feats[:, t, :].to(self.device)
                    outputs, hiddens = self.net(inputs, hiddens)
                '''
                inputs = feats.to(self.device)
                outputs = self.net(inputs[:, :local_max_length, :])
                #outputs, hiddens = self.net(inputs)
                accuracy, loss = self.metrics(outputs[:,-1,:], labels.to(self.device))
                pred_correct += accuracy
                loss.backward()
                self.opt.step()

                if (i+1) % 1000 == 0:
                    print(i+1, ' acc={0:.2f}, loss={1:.3f}, max_length={2}'.format(pred_correct*100/(i+1), loss, local_max_length))
                if i == self.dataset.n_iters()-1:
                    print('overall acc={0:.2f}'.format(pred_correct*100/(i+1)))
                    break
            print(epoch+1, 'finished')
        print('Finished Training')

#    def test(self):
#        total_accuracy_sum = 0
#        total_loss_sum = 0
#        for i, data in enumerate(self.testloader, 0):
#            images, labels = data[0].to(self.device), data[1].to(self.device)
#            outputs = self.net(images)
#            accuracy, loss = self.metrics(outputs, labels)
#            total_accuracy_sum += accuracy
#            total_loss_sum += loss.item()
#        avg_loss = total_loss_sum / len(self.testloader)
#        avg_acc = total_accuracy_sum / len(self.testloader)
