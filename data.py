import collections

class TransformedDataset:

    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, idx):
        data, label = self.dataset[idx]
        if self.transform:
            data = self.transform(data)
        return data, label

    def __len__(self):
        return len(self.dataset)


class RAMDataset:

    def __init__(self, dataset):
        self.dataset = dataset
        
        self.store = []
        for idx in range((len(self.dataset))):
            data, label = self.dataset[idx]
            self.store.append((data, label))
    
    def __getitem__(self, idx):
        data, label = self.store[idx]
        return data, label

    def __len__(self):
        return len(self.dataset)
    

def transforms(x, steps):
    for step in steps:
        x = step(x)
    return x


class PatchSize(collections.abc.Iterable):

    def __init__(self, sizes):
        self.sizes = sizes

    def __getitem__(self, idx):
        return self.sizes[idx]

    def __pow__(self, power):
        """
        handle non-square images
        """
        product = 1
        for size in self.sizes:
            product *= size
        return product

    def __iter__(self):
        return iter(self.sizes)
