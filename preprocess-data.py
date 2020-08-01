from tqdm import tqdm
import random
from collections import Counter
import pickle
from pathlib import Path

import spacy


def main():
    project_path = str(Path(__file__).resolve().parents[0])
    punctuation = ['(', ')', ':', '"', ' ']
    # Create the random train, validation, and test indices
    train_indices, val_indices, test_indices = generate_indices(len(load_data(project_path + '/data/raw/english.txt')))

    process_lang_data(project_path + '/data/raw/english.txt', 'en', punctuation, train_indices, val_indices, test_indices)
    process_lang_data(project_path + '/data/raw/french.txt', 'fr', punctuation, train_indices, val_indices, test_indices)


def process_lang_data(data_path, lang, punctuation, train_indices, val_indices, test_indices):
    lang_data = load_data(data_path)
    lang_model = spacy.load(lang, disable=['tagger', 'parser', 'ner'])

    # Tokenize the sentences
    processed_sentences = [process_sentences(lang_model, sentence, punctuation) for sentence in tqdm(lang_data)]

    train = [processed_sentences[i] for i in train_indices]

    # Get the 10000 most common tokens
    freq_list = Counter()
    for sentence in train:
        freq_list.update(sentence)
    freq_list = freq_list.most_common(10000)

    # Map words in the dictionary to indices but reserve 0 for padding,
    # 1 for out of vocabulary words, 2 for start-of-sentence and 3 for end-of-sentence
    freq_list = {freq[0]: i + 4 for i, freq in enumerate(freq_list)}
    freq_list['[PAD]'] = 0
    freq_list['[OOV]'] = 1
    freq_list['[SOS]'] = 2
    freq_list['[EOS]'] = 3
    processed_sentences = [map_words(sentence, freq_list) for sentence in tqdm(processed_sentences)]

    # Split the data
    train = [processed_sentences[i] for i in train_indices]
    val = [processed_sentences[i] for i in val_indices]
    test = [processed_sentences[i] for i in test_indices]

    # Save the data
    with open(f'data/processed/{lang}/train.pkl', 'wb') as f:
        pickle.dump(train, f)
    with open(f'data/processed/{lang}/val.pkl', 'wb') as f:
        pickle.dump(val, f)
    with open(f'data/processed/{lang}/test.pkl', 'wb') as f:
        pickle.dump(test, f)
    with open(f'data/processed/{lang}/freq_list.pkl', 'wb') as f:
        pickle.dump(freq_list, f)


def process_sentences(lang_model, sentence, punctuation):
    """
     Processes sentences by lowercasing text, ignoring punctuation, and using Spacy tokenization
             Parameters:
                     lang_model: Spacy language model
                     sentence (str): Sentence to be tokenized
                     punctuation (arr): Array of punctuation to be ignored
             Returns:
                     sentence (arr): Tokenized sentence
     """
    sentence = sentence.lower()
    sentence = [tok.text for tok in lang_model.tokenizer(sentence) if tok.text not in punctuation]

    return sentence


def load_data(data_path):
    data = []
    with open(data_path) as fp:
        for line in fp:
            data.append(line.strip())
    return data


def map_words(sentence, freq_list):
    return [freq_list[word] for word in sentence if word in freq_list]


def generate_indices(data_len):
    """
     Generate train, validation, and test indices
             Parameters:
                     data_len (int): Amount of sentences in the dataset
             Returns:
                     train_indices (arr): Array of indices for train dataset
                     val_indices (arr): Array of indices for validation dataset
                     test_indices (arr): Array of indices for test dataset
     """
    indices = [i for i in range(data_len)]
    random.shuffle(indices)

    # 80:20:0 train validation test split
    train_idx = int(data_len * 0.8)
    val_idx = train_idx + int(data_len * 0.2)
    return indices[:train_idx], indices[train_idx:val_idx], indices[val_idx:]


if __name__ == "__main__":
    main()
