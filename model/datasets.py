import torch
from torch.utils.data import Dataset
import h5py
import json
import os


class ParagraphDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, data_folder, data_name, split, transform=None):
        """
        :param data_folder: folder where data files are stored
        :param data_name: base name of processed datasets
        :param split: split, one of 'TRAIN', 'VAL', or 'TEST'
        :param transform: image transform pipeline
        """
        self.split = split
        #assert self.split in {'TRAIN_VAL', 'TEST_TEST'}
        assert self.split in {'TRAIN', 'VAL', 'TEST'}

        # Open hdf5 file where images are stored
        self.h5_file = h5py.File(os.path.join(data_folder, self.split + '_images_' + data_name + '.hdf5'), 'r')

        # Load DenseCap caption scores
        self.densecap = self.h5_file['phrases']

        # Image features in current split
        self.imgs = self.h5_file['images']

        # List of image in current split
        self.image_ids = self.h5_file['image_ids']
        
        # Bounding boxes for attention visualisation
        self.bboxes = self.h5_file['boxes']

        # Sentence per paragraph per image
        self.cpi = self.h5_file.attrs['sentences_per_paragraph']

        # Load encoded captions (completely into memory)
        with open(os.path.join(data_folder, self.split + '_captions_' + data_name + '.json'), 'r') as j:
            self.captions = json.load(j)

        # Load caption lengths (completely into memory)
        with open(os.path.join(data_folder, self.split + '_caplens_' + data_name + '.json'), 'r') as j:
            self.caplens = json.load(j)
            
        # Load phrases to language information
        with open(os.path.join(data_folder, 'densecap_phrases_' + self.split + '.json'), 'r') as j:
            self.phrases = json.load(j)

        # PyTorch transformation pipeline for the image (normalizing, etc.)
        self.transform = transform

        # Total number of datapoints
        self.dataset_size = len(self.captions) // self.cpi

    def __getitem__(self, i):
        
        image_id = int(self.image_ids[i].split('/')[-1].split('.jpg')[0])
        
        densecap_phrases = self.phrases[str(image_id)]
        # get phrase lengths
        phrase_lengths = []
        for phrase in densecap_phrases:
            phrase_lengths.append(len(phrase.split()))
        phrase_lengths = torch.FloatTensor(phrase_lengths)
        
        image = torch.FloatTensor(self.imgs[i])
        
        phrase_scores = self.densecap[i]
        bboxes = self.bboxes[i]
        
        if self.transform is not None:
            image = self.transform(image)
        # Locate indexes of paragraph sentences for the current image
        if i != 0:
            #6
            cap_end = i * self.cpi + 6 # because of indexing starting from 0
            cap_start = cap_end - 6
        elif i == 0:
            cap_end = 6
            cap_start = 0
        captions = torch.LongTensor(self.captions[cap_start:cap_end])
        caplen = torch.LongTensor([self.caplens[cap_start:cap_end]])
        #print(this_densecap)

        #print(image, image_id, captions, caplen, phrase_scores, bboxes)
        return image, image_id, captions, caplen, phrase_scores, bboxes, phrase_lengths
        #return image, image_id, captions, caplen, this_densecap

    def __len__(self):
        return self.dataset_size
