import pickle
from einops import rearrange

import spacy
import torch


def main():
    # Load the trained model, Spacy tokenizer, and frequency lists
    model = torch.load('output/transformer.pth')
    lang_model = spacy.load('en')
    with open('data/processed/en/freq_list.pkl', 'rb') as f:
        en_freq_list = pickle.load(f)
    with open('data/processed/fr/freq_list.pkl', 'rb') as f:
        fr_freq_list = pickle.load(f)

    # Tokenize input
    sentence = input('Please enter your english sentence: ')
    sentence = tokenize(sentence, en_freq_list, lang_model)

    # Generate the translated sentence, feeding the model's output into its input
    translated_sentence = [fr_freq_list['[SOS]']]
    i = 0
    while int(translated_sentence[-1]) != fr_freq_list['[EOS]'] and i < 15:
        output = forward_model(model, sentence, translated_sentence).to('cuda')
        values, indices = torch.topk(output, 5)
        translated_sentence.append(int(indices[-1][0]))

    # Print out the translated sentence
    print(detokenize(translated_sentence, fr_freq_list))


def forward_model(model, src, tgt):
    src = torch.tensor(src).unsqueeze(0).long().to('cuda')
    tgt = torch.tensor(tgt).unsqueeze(0).to('cuda')
    tgt_mask = gen_nopeek_mask(tgt.shape[1]).to('cuda')
    output = model.forward(src, tgt, src_key_padding_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None, tgt_mask=tgt_mask)

    return output.squeeze(0).to('cpu')


def tokenize(sentence, freq_list, lang_model):
    punctuation = ['(', ')', ':', '"', ' ']

    sentence = sentence.lower()
    sentence = [tok.text for tok in lang_model.tokenizer(sentence) if tok.text not in punctuation]
    return [freq_list[word] if word in freq_list else freq_list['[OOV]'] for word in sentence]


def detokenize(sentence, freq_list):
    freq_list = {v: k for k, v in freq_list.items()}
    return [freq_list[token] for token in sentence]


def gen_nopeek_mask(length):
    mask = rearrange(torch.triu(torch.ones(length, length)) == 1, 'h w -> w h')
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))

    return mask


if __name__ == "__main__":
    main()
