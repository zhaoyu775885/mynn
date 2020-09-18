import torch
import torchvision
import torchvision.transforms as transforms

# todo: merge Cifar10 & Cifar100

class Cifar10():
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.n_class = 10

    def build_dataloader(self, batch_size, is_train=True):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        # todo: standard normalization is required

        train_transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                              transforms.RandomCrop(32, 4),
                                              transforms.ToTensor(),
                                              normalize])
        # todo: data augmentation makes difference

        valid_transform = transforms.Compose([transforms.ToTensor(),
                                              normalize])

        dataset = torchvision.datasets.CIFAR10(root=self.data_dir,
                                               train=is_train,
                                               transform=train_transform if is_train else valid_transform)

        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                 shuffle=is_train, num_workers=16)
        return dataloader

class Cifar100():
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.n_class = 100

    def build_dataloader(self, batch_size, is_train=True):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        train_transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                              transforms.RandomCrop(32, 4),
                                              transforms.ToTensor(),
                                              normalize])

        valid_transform = transforms.Compose([transforms.ToTensor(),
                                              normalize])

        dataset = torchvision.datasets.CIFAR100(root=self.data_dir,
                                               train=is_train,
                                               transform=train_transform if is_train else valid_transform)

        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                 shuffle=is_train, num_workers=16)
        return dataloader

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

if __name__ == '__main__':
    dataset = '/home/zhaoyu/Datasets/cifar100'
    dataset = Cifar100(dataset)

    iterator = dataset.build_dataloader(4)
    for i, batch in enumerate(iterator):
        if i>0:
            break
        print(i, len(batch), batch[0], batch[1])
