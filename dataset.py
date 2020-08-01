import pickle
import random
import numpy as np

from torch.utils.data import Dataset


class ParallelLanguageDataset(Dataset):
    def __init__(self, data_path_1, data_path_2, num_tokens, max_seq_length):
        """
        Initializes the dataset
                Parameters:
                        data_path_1 (str): Path to the English pickle file processed in process-data.py
                        data_path_2 (str): Path to the French pickle file processed in process-data.py
                        num_tokens (int): Maximum number of tokens in each batch (restricted by GPU memory usage)
                        max_seq_length (int): Maximum number of tokens in each sentence pair
        """
        self.num_tokens = num_tokens
        self.data_1, self.data_2, self.data_lengths = load_data(data_path_1, data_path_2, max_seq_length)

        self.batches = gen_batches(num_tokens, self.data_lengths)

    def __getitem__(self, idx):
        src, src_mask = getitem(idx, self.data_1, self.batches, True)
        tgt, tgt_mask = getitem(idx, self.data_2, self.batches, False)

        return src, src_mask, tgt, tgt_mask

    def __len__(self):
        return len(self.batches)

    def shuffle_batches(self):
        self.batches = gen_batches(self.num_tokens, self.data_lengths)


def gen_batches(num_tokens, data_lengths):
    """
     Returns the batched data
             Parameters:
                     num_tokens (int): Maximum number of tokens in each batch (restricted by GPU memory usage)
                     data_lengths (dict): A dict with keys of tuples (length of English sentence, length of corresponding French sentence)
                                         and values of the indices that correspond to these parallel sentences
             Returns:
                     batches (arr): List of each batch (which consists of an array of indices)
     """

    # Shuffle all the indices
    for k, v in data_lengths.items():
        random.shuffle(v)

    batches = []
    prev_tokens_in_batch = 1e10
    for k in sorted(data_lengths):
        # v contains indices of the sentences
        v = data_lengths[k]
        total_tokens = (k[0] + k[1]) * len(v)

        # Repeat until all the sentences in this key-value pair are in a batch
        while total_tokens > 0:
            tokens_in_batch = min(total_tokens, num_tokens) - min(total_tokens, num_tokens) % (k[0] + k[1])
            sentences_in_batch = tokens_in_batch // (k[0] + k[1])

            # Combine with previous batch if it can fit
            if tokens_in_batch + prev_tokens_in_batch <= num_tokens:
                batches[-1].extend(v[:sentences_in_batch])
                prev_tokens_in_batch += tokens_in_batch
            else:
                batches.append(v[:sentences_in_batch])
                prev_tokens_in_batch = tokens_in_batch
            # Remove indices from v that have been added in a batch
            v = v[sentences_in_batch:]

            total_tokens = (k[0] + k[1]) * len(v)
    return batches


def load_data(data_path_1, data_path_2, max_seq_length):
    """
    Loads the pickle files created in preprocess-data.py
            Parameters:
                        data_path_1 (str): Path to the English pickle file processed in process-data.py
                        data_path_2 (str): Path to the French pickle file processed in process-data.py
                        max_seq_length (int): Maximum number of tokens in each sentence pair

            Returns:
                    data_1 (arr): Array of tokenized English sentences
                    data_2 (arr): Array of tokenized French sentences
                    data_lengths (dict): A dict with keys of tuples (length of English sentence, length of corresponding French sentence)
                                         and values of the indices that correspond to these parallel sentences
    """
    with open(data_path_1, 'rb') as f:
        data_1 = pickle.load(f)
    with open(data_path_2, 'rb') as f:
        data_2 = pickle.load(f)

    data_lengths = {}
    for i, (str_1, str_2) in enumerate(zip(data_1, data_2)):
        if 0 < len(str_1) <= max_seq_length and 0 < len(str_2) <= max_seq_length - 2:
            if (len(str_1), len(str_2)) in data_lengths:
                data_lengths[(len(str_1), len(str_2))].append(i)
            else:
                data_lengths[(len(str_1), len(str_2))] = [i]
    return data_1, data_2, data_lengths


def getitem(idx, data, batches, src):
    """
    Retrieves a batch given an index
            Parameters:
                        idx (int): Index of the batch
                        data (arr): Array of tokenized sentences
                        batches (arr): List of each batch (which consists of an array of indices)
                        src (bool): True if the language is the source language, False if it's the target language

            Returns:
                    batch (arr): Array of tokenized English sentences, of size (num_sentences, num_tokens_in_sentence)
                    masks (arr): key_padding_masks for the sentences, of size (num_sentences, num_tokens_in_sentence)
    """

    sentence_indices = batches[idx]
    if src:
        batch = [data[i] for i in sentence_indices]
    else:
        # If it's in the target language, add [SOS] and [EOS] tokens
        batch = [[2] + data[i] + [3] for i in sentence_indices]

    # Get the maximum sentence length
    seq_length = 0
    for sentence in batch:
        if len(sentence) > seq_length:
            seq_length = len(sentence)

    masks = []
    for i, sentence in enumerate(batch):
        # Generate the masks for each sentence, False if there's a token, True if there's padding
        masks.append([False for _ in range(len(sentence))] + [True for _ in range(seq_length - len(sentence))])
        # Add 0 padding
        batch[i] = sentence + [0 for _ in range(seq_length - len(sentence))]

    return np.array(batch), np.array(masks)
