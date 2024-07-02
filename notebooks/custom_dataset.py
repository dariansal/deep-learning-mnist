'''Get data -> split into subsets -> wrap them in DataSetT class and give them transformation attribute, 
define methods necessary for DataLoader-> merge them back together'''
class DataSetT:
    def __init__(self, images, labels, transform):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self): #subset has length
        return len(self.images)
    
    def __getitem__(self, index): #works with DataLoader, lets me do dataset[index]; SubSet has this implemented but not with transform
        sample = self.images[index]
        sample_transformed = self.transform(sample) # __call__ method of Compose on Data instance: for sammple (s) in self.transforms, s = t(s)
        label = self.labels[index] 
        return sample_transformed, label