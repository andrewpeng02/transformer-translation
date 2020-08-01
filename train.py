import click
from pathlib import Path
from einops import rearrange
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from torch.optim import Adam
from Optim import ScheduledOptim
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import ParallelLanguageDataset
from model import LanguageTransformer


@click.command()
@click.argument('num_epochs', type=int,  default=20)
@click.argument('max_seq_length', type=int,  default=96)
@click.argument('num_tokens', type=int,  default=2000)
@click.argument('vocab_size', type=int,  default=10000 + 4)
@click.argument('d_model', type=int,  default=512)
@click.argument('num_encoder_layers', type=int,  default=6)
@click.argument('num_decoder_layers', type=int,  default=6)
@click.argument('dim_feedforward', type=int,  default=2048)
@click.argument('nhead', type=int,  default=8)
@click.argument('pos_dropout', type=float,  default=0.1)
@click.argument('trans_dropout', type=float,  default=0.1)
@click.argument('n_warmup_steps', type=int,  default=4000)
def main(**kwargs):
    project_path = str(Path(__file__).resolve().parents[0])

    train_dataset = ParallelLanguageDataset(project_path + '/data/processed/en/train.pkl',
                                            project_path + '/data/processed/fr/train.pkl',
                                            kwargs['num_tokens'], kwargs['max_seq_length'])
    # Set batch_size=1 because all the batching is handled in the ParallelLanguageDataset class
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4, pin_memory=True)
    valid_dataset = ParallelLanguageDataset(project_path + '/data/processed/en/val.pkl',
                                            project_path + '/data/processed/fr/val.pkl',
                                            kwargs['num_tokens'], kwargs['max_seq_length'])
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=True, num_workers=4, pin_memory=True)

    model = LanguageTransformer(kwargs['vocab_size'], kwargs['d_model'], kwargs['nhead'], kwargs['num_encoder_layers'],
                                kwargs['num_decoder_layers'], kwargs['dim_feedforward'], kwargs['max_seq_length'],
                                kwargs['pos_dropout'], kwargs['trans_dropout']).to('cuda')

    # Use Xavier normal initialization in the transformer
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_normal_(p)

    optim = ScheduledOptim(
        Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-09),
        kwargs['d_model'], kwargs['n_warmup_steps'])

    # Use cross entropy loss, ignoring any padding
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    train_losses, val_losses = train(train_loader, valid_loader, model, optim, criterion, kwargs['num_epochs'])


def train(train_loader, valid_loader, model, optim, criterion, num_epochs):
    print_every = 500
    model.train()

    lowest_val = 1e9
    train_losses = []
    val_losses = []
    total_step = 0
    for epoch in range(num_epochs):
        pbar = tqdm(total=print_every, leave=False)
        total_loss = 0

        # Shuffle batches every epoch
        train_loader.dataset.shuffle_batches()
        for step, (src, src_key_padding_mask, tgt, tgt_key_padding_mask) in enumerate(iter(train_loader)):
            total_step += 1

            # Send the batches and key_padding_masks to gpu
            src, src_key_padding_mask = src[0].to('cuda'), src_key_padding_mask[0].to('cuda')
            tgt, tgt_key_padding_mask = tgt[0].to('cuda'), tgt_key_padding_mask[0].to('cuda')
            memory_key_padding_mask = src_key_padding_mask.clone()

            # Create tgt_inp and tgt_out (which is tgt_inp but shifted by 1)
            tgt_inp, tgt_out = tgt[:, :-1], tgt[:, 1:]
            tgt_mask = gen_nopeek_mask(tgt_inp.shape[1]).to('cuda')

            # Forward
            optim.zero_grad()
            outputs = model(src, tgt_inp, src_key_padding_mask, tgt_key_padding_mask[:, :-1], memory_key_padding_mask, tgt_mask)
            loss = criterion(rearrange(outputs, 'b t v -> (b t) v'), rearrange(tgt_out, 'b o -> (b o)'))

            # Backpropagate and update optim
            loss.backward()
            optim.step_and_update_lr()

            total_loss += loss.item()
            train_losses.append((step, loss.item()))
            pbar.update(1)
            if step % print_every == print_every - 1:
                pbar.close()
                print(f'Epoch [{epoch + 1} / {num_epochs}] \t Step [{step + 1} / {len(train_loader)}] \t '
                      f'Train Loss: {total_loss / print_every}')
                total_loss = 0

                pbar = tqdm(total=print_every, leave=False)

        # Validate every epoch
        pbar.close()
        val_loss = validate(valid_loader, model, criterion)
        val_losses.append((total_step, val_loss))
        if val_loss < lowest_val:
            lowest_val = val_loss
            torch.save(model, 'output/transformer.pth')
        print(f'Val Loss: {val_loss}')
    return train_losses, val_losses


def validate(valid_loader, model, criterion):
    pbar = tqdm(total=len(iter(valid_loader)), leave=False)
    model.eval()

    total_loss = 0
    for src, src_key_padding_mask, tgt, tgt_key_padding_mask in iter(valid_loader):
        with torch.no_grad():
            src, src_key_padding_mask = src[0].to('cuda'), src_key_padding_mask[0].to('cuda')
            tgt, tgt_key_padding_mask = tgt[0].to('cuda'), tgt_key_padding_mask[0].to('cuda')
            memory_key_padding_mask = src_key_padding_mask.clone()
            tgt_inp = tgt[:, :-1]
            tgt_out = tgt[:, 1:].contiguous()
            tgt_mask = gen_nopeek_mask(tgt_inp.shape[1]).to('cuda')

            outputs = model(src, tgt_inp, src_key_padding_mask, tgt_key_padding_mask[:, :-1], memory_key_padding_mask, tgt_mask)
            loss = criterion(rearrange(outputs, 'b t v -> (b t) v'), rearrange(tgt_out, 'b o -> (b o)'))

            total_loss += loss.item()
            pbar.update(1)

    pbar.close()
    model.train()
    return total_loss / len(valid_loader)


def gen_nopeek_mask(length):
    """
     Returns the nopeek mask
             Parameters:
                     length (int): Number of tokens in each sentence in the target batch
             Returns:
                     mask (arr): tgt_mask, looks like [[0., -inf, -inf],
                                                      [0., 0., -inf],
                                                      [0., 0., 0.]]
     """
    mask = rearrange(torch.triu(torch.ones(length, length)) == 1, 'h w -> w h')
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))

    return mask


if __name__ == "__main__":
    main()
