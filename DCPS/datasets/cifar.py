import torch
import torchvision
import torchvision.transforms as transforms
from itertools import cycle

# todo: merge Cifar10 & Cifar100

class Cifar():
    def __init__(self, dataset_fn, data_dir):
        self.dataset_fn = dataset_fn
        self.data_dir = data_dir

    def build_dataloader(self, batch_size, is_train=True, valid=False):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        train_transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                              transforms.RandomCrop(32, 4),
                                              transforms.ToTensor(),
                                              normalize])

        valid_transform = transforms.Compose([transforms.ToTensor(),
                                              normalize])

        dataset = self.dataset_fn(root=self.data_dir,
                                  train=is_train,
                                  transform=train_transform if is_train else valid_transform)

        if valid:
            dataset_train, dataset_valid = torch.utils.data.random_split(dataset, [45000, 5000])
            train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size,
                                                       shuffle=is_train, num_workers=16)
            valid_loader = torch.utils.data.DataLoader(dataset_valid, batch_size=batch_size,
                                                       shuffle=is_train, num_workers=16)
            return train_loader, valid_loader

        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=is_train, num_workers=16)

class Cifar10(Cifar):
    def __init__(self, data_dir):
        super(Cifar10, self).__init__(torchvision.datasets.CIFAR10, data_dir)
        self.n_class = 10

class Cifar100(Cifar):
    def __init__(self, data_dir):
        super(Cifar100, self).__init__(torchvision.datasets.CIFAR100, data_dir)
        self.n_class = 100


if __name__ == '__main__':
    dataset = '/home/zhaoyu/Datasets/cifar100'
    dataset = Cifar100(dataset)

    train, valid = dataset.build_dataloader(128, valid=True)

    for i, batch in enumerate(zip(train, cycle(valid))):
        print(i)

# def imshow(img):
#     img = img/2 + 0.5
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))
#     plt.show()
#
# def instance(dataloader, classes, batch_size=BATCH_SIZE):
#     dataiter = iter(dataloader)
#     images, labels = dataiter.next()
#     imshow(torchvision.utils.make_grid(images))
#     print(' '.join([classes[labels[j]] for j in range(batch_size)]))
