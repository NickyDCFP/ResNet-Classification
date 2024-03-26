import os
import torch
import torch.nn as nn
import torch.optim as optim

import nets


def init_weights(net, gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            nn.init.normal_(m.weight.data, 1.0, gain)
            nn.init.constant_(m.bias.data, 0.0)

    net.apply(init_func)


class ResNetModel:
    def __init__(self, opt, train=True):
        self.net = nets.ResNetCifar(opt.n)
        if train:
            self.net.train()
        else:
            self.net.eval()
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            self.net = self.net.to('cuda')
            self.net = torch.nn.DataParallel(self.net)
        else:
            self.device = torch.device('cpu')

        init_weights(self.net)
        num_params = 0
        for param in self.net.parameters():
            num_params += param.numel()
        print(f'Total number of parameters : {num_params / 1e6:.3f} M')

        if train:
            self.checkpoint_dir = opt.checkpoint_dir
            assert opt.optimizer in ['SGD', 'Adam'], 'Must use SGD or Adam optimizer'
            if opt.optimizer == 'SGD':
                self.optimizer = optim.SGD(
                    self.net.parameters(),
                    lr=opt.lr,
                    momentum=opt.momentum,
                    weight_decay=opt.weight_decay
                    )
            elif opt.optimizer == 'Adam':
                self.optimizer = optim.Adam(
                    self.net.parameters(),
                    lr=opt.lr,
                    weight_decay=opt.weight_decay
                )
            
            self.scheduler = optim.lr_scheduler.MultiStepLR(
                self.optimizer,
                milestones=[opt.decay_lr_1, opt.decay_lr_2],
                gamma=opt.lr_decay_rate
                )
            self.criterion = nn.CrossEntropyLoss()
            self.loss = 0.0

    def optimize_params(self, x, label):
        x = x.to(self.device)
        label = label.to(self.device)
        y = self._forward(x)
        _, predicted = torch.max(y.data, 1)
        total = label.size(0)
        correct = (predicted == label).sum().item()
        loss = self.criterion(y, label)
        self._update_params(y, label)
        return correct, total, loss.item()

    def _forward(self, x):
        return self.net(x)

    def _backward(self, y, label):
        self.loss = self.criterion(y, label)
        self.loss.backward()

    def _update_params(self, y, label):
        self.optimizer.zero_grad()
        self._backward(y, label)
        self.optimizer.step()
        self.scheduler.step()  # scheduler step in each iteration

    def test(self, x, label):
        with torch.no_grad():
            x = x.to(self.device)
            label = label.to(self.device)
            outputs = self._forward(x)
            _, predicted = torch.max(outputs.data, 1)
            total = label.size(0)
            correct = (predicted == label).sum().item()
            loss = self.criterion(outputs, label)
            return correct, total, loss.item()

    def val(self, x, label):
        with torch.no_grad():
            x = x.to(self.device)
            label = label.to(self.device)
            y = self._forward(x)
            return self.criterion(y, label).item()

    def save_model(self, name):
        path = os.path.join(self.checkpoint_dir, f'model_{name}.pth')
        torch.save(self.net.state_dict(), path)
        print(f'model saved to {path}')

    def load_model(self, path):
        self.net.load_state_dict(torch.load(path))
        print(f'model loaded from {path}')

    def get_current_loss(self):
        return self.loss.item()
