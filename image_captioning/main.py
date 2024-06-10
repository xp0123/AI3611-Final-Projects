#!/usr/bin/env python3
import os
from pathlib import Path
import pickle
import yaml
import random
import numpy as np
import fire
import pandas as pd
import torch
from tqdm import tqdm
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader
from torchvision import transforms

from datasets.flickr8k import Flickr8kDataset
from utils.metrics import bleu_score_fn
from utils.utils_torch import words_from_tensors_fn
from utils.util import get_logger
from models import Captioner


class Runner(object): 
    """Main class to run experiments"""
    def __init__(self, seed=1):
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        self.device = torch.device(device)
    
    def get_dataloaders(self, dataset_base_path, batch_size):
        train_set = Flickr8kDataset(
            dataset_base_path=dataset_base_path, dist='train',
            return_type='tensor', load_img_to_memory=False)
        vocab_set = train_set.get_vocab()
        val_set = Flickr8kDataset(
            dataset_base_path=dataset_base_path, dist='val',
            vocab_set=vocab_set, return_type='corpus',
            load_img_to_memory=False)
        test_set = Flickr8kDataset(
            dataset_base_path=dataset_base_path, dist='test',
            vocab_set=vocab_set, return_type='corpus',
            load_img_to_memory=False)
        train_eval_set = Flickr8kDataset(
            dataset_base_path=dataset_base_path, dist='train',
            vocab_set=vocab_set, return_type='corpus',
            load_img_to_memory=False)
        train_transformations = transforms.Compose([
            transforms.Resize(256),  # smaller edge of image resized to 256
            transforms.RandomCrop(256),  # get 256x256 crop from random location
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),  # convert the PIL Image to a tensor
            transforms.Normalize((0.485, 0.456, 0.406),  # normalize image for pre-trained model
                                 (0.229, 0.224, 0.225))
        ])
        eval_transformations = transforms.Compose([
            transforms.Resize(256),  # smaller edge of image resized to 256
            transforms.CenterCrop(256),  # get 256x256 crop from random location
            transforms.ToTensor(),  # convert the PIL Image to a tensor
            transforms.Normalize((0.485, 0.456, 0.406),  # normalize image for pre-trained model
                                 (0.229, 0.224, 0.225))
        ])

        train_set.transformations = train_transformations
        val_set.transformations = eval_transformations
        test_set.transformations = eval_transformations
        train_eval_set.transformations = eval_transformations

        eval_collate_fn = lambda batch: (
            torch.stack([x[0] for x in batch]),
            [x[1] for x in batch],
            [x[2] for x in batch],
            [x[3] for x in batch]
        )
        train_loader = DataLoader(train_set, batch_size=batch_size,
                                  shuffle=True, num_workers=4)
        val_loader = DataLoader(val_set, batch_size=1, shuffle=False,
                                num_workers=4, collate_fn=eval_collate_fn)
        test_loader = DataLoader(test_set, batch_size=1, shuffle=False,
                                 num_workers=4, collate_fn=eval_collate_fn)
        train_eval_loader = DataLoader(train_eval_set, batch_size=batch_size,
            shuffle=False, num_workers=4, collate_fn=eval_collate_fn)
        return {
            "train": train_loader,
            "train_eval": train_eval_loader,
            "val": val_loader,
            "test": test_loader
        }

    def train_model(self, train_loader, model, loss_fn, optimizer, desc=''):
        running_acc = 0.0
        running_loss = 0.0
        model.train()
        t = tqdm(iter(train_loader), desc=f'{desc}', leave=False)
        for batch_idx, batch in enumerate(t):
            images, captions, lengths = batch
            images = images.to(self.device)
            captions = captions.to(self.device)
            optimizer.zero_grad()

            scores, caps_sorted, decode_lengths, alphas, sort_ind = model(
                images, captions, lengths)

            # Since decoding starts with <start>, the targets are all words after <start>, up to <end>
            targets = caps_sorted[:, 1:]

            # Remove timesteps that we didn't decode at, or are pads
            # pack_padded_sequence is an easy trick to do this
            scores = pack_padded_sequence(scores, decode_lengths,
                batch_first=True)[0]
            targets = pack_padded_sequence(targets, decode_lengths,
                batch_first=True)[0]

            loss = loss_fn(scores, targets)
            loss.backward()
            optimizer.step()

            correct = (torch.argmax(scores, dim=1) == targets).sum().float().item()
            running_acc += correct / targets.size(0)
            running_loss += loss.item()
            t.set_postfix({
                'loss': running_loss / (batch_idx + 1),
                'acc': running_acc / (batch_idx + 1)}, refresh=True)
        t.close()

        return running_loss / len(train_loader)

    def evaluate_model(self, data_loader, model, bleu_score_fn,
        tensor_to_word_fn, word2idx, sample_method, desc='', return_output=False):
        bleu = [0.0] * 5
        model.eval()
        references = []
        predictions = []
        imgids = []
        t = tqdm(iter(data_loader), desc=f'{desc}', leave=False)
        for batch_idx, batch in enumerate(t):
            images, captions, lengths, imgid_batch = batch
            images = images.to(self.device)
            outputs = tensor_to_word_fn(model.sample(
                images, startseq_idx=word2idx['<start>'],
                method=sample_method).cpu().numpy())
            references.extend(captions)
            predictions.extend(outputs)
            imgids.extend(imgid_batch)
            t.set_postfix({
                'batch': batch_idx,
            }, refresh=True)
        t.close()
        for i in (1, 2, 3, 4):
            bleu[i] = bleu_score_fn(reference_corpus=references,
                                    candidate_corpus=predictions, n=i)
        references = [
           [" ".join(cap) for cap in caption] for caption in references
        ]
        predictions = [
            " ".join(caption) for caption in predictions
        ]
        return (bleu, references, predictions, imgids) if return_output else bleu

    def train(self, config_file, **kwargs):
        with open(config_file) as reader:
            config = yaml.load(reader, Loader=yaml.FullLoader)
        args = dict(config, **kwargs)
        
        dataloaders = self.get_dataloaders(args["dataset_base_path"],
                                           args["train_args"]["batch_size"])
        vocab_set = dataloaders["train"].dataset.get_vocab()
        vocab, word2idx, idx2word, max_len = vocab_set
        with open(args['vocab_path'], 'wb') as f:
            pickle.dump(vocab_set, f)
        vocab_size = len(vocab)

        Path(args["outputpath"]).mkdir(parents=True, exist_ok=True)
        logger = get_logger(Path(args["outputpath"]) / "train.log")

        model = Captioner(encoded_image_size=14,
                          encoder_dim=2048,
                          attention_dim=args['attention_dim'],
                          embed_dim=args['embedding_dim'],
                          decoder_dim=args['decoder_size'],
                          decay_method=args['decay_method'],
                          vocab_size=vocab_size).to(self.device)
        logger.info(model)
        model_path = os.path.join(args["outputpath"],
            f"{args['model']}_b{args['train_args']['batch_size']}_"
            f"emd{args['embedding_dim']}")
        
        pad_value = dataloaders["train"].dataset.pad_value
        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=pad_value).to(self.device)
        corpus_bleu_score_fn = bleu_score_fn(4, 'corpus')
        tensor_to_word_fn = words_from_tensors_fn(idx2word=idx2word)
        params = model.parameters()
        optimizer = torch.optim.RMSprop(params=params,
            lr=float(args['train_args']['learning_rate']))

        train_loss_min = 100
        val_bleu4_max = 0.0
        num_epochs = args["train_args"]["num_epochs"]
        for epoch in range(num_epochs):
            train_loss = self.train_model(desc=f'Epoch {epoch + 1}/{num_epochs}',
                                          model=model,
                                          optimizer=optimizer,
                                          loss_fn=loss_fn,
                                          train_loader=dataloaders["train"])
            with torch.no_grad():
                # train_bleu = self.evaluate_model(
                    # desc=f'Train eval: ', model=model,
                    # bleu_score_fn=corpus_bleu_score_fn,
                    # tensor_to_word_fn=tensor_to_word_fn,
                    # word2idx=word2idx, sample_method=args['sample_method'],
                    # data_loader=dataloaders["train_eval"])
                val_bleu = self.evaluate_model(
                    desc=f'Val eval: ', model=model,
                    bleu_score_fn=corpus_bleu_score_fn,
                    tensor_to_word_fn=tensor_to_word_fn,
                    sample_method=args['sample_method'],
                    word2idx=word2idx, data_loader=dataloaders["val"])
                msg = f"Epoch {epoch + 1}/{num_epochs}, train_loss: " \
                    f"{train_loss:.3f}, val_bleu1: {val_bleu[1]:.3f}, " \
                    f"val_bleu4: {val_bleu[4]:.3f}"
                logger.info(msg)
                state = {
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'train_loss_latest': train_loss,
                    'val_bleu4_latest': val_bleu[4],
                    'train_loss_min': min(train_loss, train_loss_min),
                    'val_bleu4_max': max(val_bleu[4], val_bleu4_max),
                    # 'train_bleus': train_bleu,
                    'val_bleus': val_bleu,
                }
                torch.save(state, '{}_latest.pt'.format(model_path))
                if train_loss < train_loss_min:
                    train_loss_min = train_loss
                    torch.save(state, '{}_best_train.pt'.format(model_path))
                if val_bleu[4] > val_bleu4_max:
                    val_bleu4_max = val_bleu[4]
                    torch.save(state, '{}_best_val.pt'.format(model_path))
        torch.save(state, f'{model_path}_ep{num_epochs:02d}_weights.pt')

        state = torch.load(f'{model_path}_best_val.pt', map_location="cpu")
        model.load_state_dict(state["state_dict"])

        with torch.no_grad():
            model.eval()
            train_bleu = self.evaluate_model(desc=f'Train: ', model=model,
                bleu_score_fn=corpus_bleu_score_fn,
                tensor_to_word_fn=tensor_to_word_fn,
                sample_method=args['sample_method'],
                word2idx=word2idx, data_loader=dataloaders["train_eval"])
            val_bleu = self.evaluate_model(desc=f'Val: ', model=model,
                bleu_score_fn=corpus_bleu_score_fn,
                tensor_to_word_fn=tensor_to_word_fn,
                word2idx=word2idx, sample_method=args['sample_method'],
                data_loader=dataloaders["val"])
            test_bleu = self.evaluate_model(desc=f'Test: ', model=model,
                bleu_score_fn=corpus_bleu_score_fn,
                tensor_to_word_fn=tensor_to_word_fn,
                word2idx=word2idx,
                sample_method=args['sample_method'],
                data_loader=dataloaders["test"])
            logger.info("evaluation of the best validation performance model: ")
            for setname, result in zip(('train', 'val', 'test'),
                                       (train_bleu, val_bleu, test_bleu)):
                logger.info(setname, end=' ')
                for ngram in (1, 2, 3, 4):
                    logger.info(f'Bleu-{ngram}: {result[ngram]:.3f}', end=' ')
                logger.info("")

    def evaluate(self, config_file, **kwargs):
        import json
        from pycocoevalcap.bleu.bleu import Bleu
        from pycocoevalcap.rouge.rouge import Rouge
        from pycocoevalcap.meteor.meteor import Meteor
        from pycocoevalcap.cider.cider import Cider
        from pycocoevalcap.spice.spice import Spice

        from utils.util import ptb_tokenize

        with open(config_file) as reader:
            config = yaml.load(reader, Loader=yaml.FullLoader)
        args = dict(config, **kwargs)

        vocab_set = pickle.load(open(args['vocab_path'], "rb"))
        test_set = Flickr8kDataset(dataset_base_path=args['dataset_base_path'],
                                   dist='test', vocab_set=vocab_set,
                                   return_type='corpus',
                                   load_img_to_memory=False)
        vocab, word2idx, idx2word, max_len = vocab_set
        vocab_size = len(vocab)

        eval_transformations = transforms.Compose([
            transforms.Resize(256),  # smaller edge of image resized to 256
            transforms.CenterCrop(256),  # get 256x256 crop from random location
            transforms.ToTensor(),  # convert the PIL Image to a tensor
            transforms.Normalize((0.485, 0.456, 0.406),  # normalize image for pre-trained model
                                 (0.229, 0.224, 0.225))
        ])

        test_set.transformations = eval_transformations
        eval_collate_fn = lambda batch: (
            torch.stack([x[0] for x in batch]),
            [x[1] for x in batch], [x[2] for x in batch], [x[3] for x in batch])
        test_loader = torch.utils.data.DataLoader(test_set,
            batch_size=1, shuffle=False, collate_fn=eval_collate_fn)

        model = Captioner(encoded_image_size=14, encoder_dim=2048,
                          attention_dim=args["attention_dim"],
                          embed_dim=args["embedding_dim"],
                          decoder_dim=args["decoder_size"],
                          decay_method=args['decay_method'],
                          vocab_size=vocab_size, train_embd=False)
        model_path = os.path.join(args["outputpath"],
            f"{args['model']}_b{args['train_args']['batch_size']}_"
            f"emd{args['embedding_dim']}")
        state = torch.load(f'{model_path}_best_val.pt', map_location="cpu")
        model.load_state_dict(state["state_dict"])
        model = model.to(self.device)

        corpus_bleu_score_fn = bleu_score_fn(4, 'corpus')
        tensor_to_word_fn = words_from_tensors_fn(idx2word=idx2word)

        with torch.no_grad():
            model.eval()
            test_bleu, references, predictions, imgids = self.evaluate_model(
                desc=f'Test: ', model=model, bleu_score_fn=corpus_bleu_score_fn,
                tensor_to_word_fn=tensor_to_word_fn, data_loader=test_loader,
                sample_method=args['sample_method'], word2idx=word2idx,
                return_output=True)
            key_to_pred = {}
            key_to_refs = {}
            output_pred = []
            for imgid, pred, refs in zip(imgids, predictions, references):
                key_to_pred[imgid] = [pred,]
                key_to_refs[imgid] = refs
                output_pred.append({
                    "img_id": imgid,
                    "prediction": [pred,]
                })

            key_to_refs = ptb_tokenize(key_to_refs)
            key_to_pred = ptb_tokenize(key_to_pred)
            scorers = [Bleu(n=4), Rouge(), Meteor(), Cider(), Spice()]
            output = {"SPIDEr": 0}
            with open(f"{model_path}_coco_scores.txt", "w") as writer:
                for scorer in scorers:
                    score, scores = scorer.compute_score(key_to_refs, key_to_pred)
                    method = scorer.method()
                    output[method] = score
                    if method == "Bleu":
                        for n in range(4):
                            print("Bleu-{}: {:.3f}".format(n + 1, score[n]), file=writer)
                    else:
                        print(f"{method}: {score:.3f}", file=writer)
                    if method in ["CIDEr", "SPICE"]:
                        output["SPIDEr"] += score
                output["SPIDEr"] /= 2
                print(f"SPIDEr: {output['SPIDEr']:.3f}", file=writer)

            json.dump(output_pred, open(f"{model_path}_predictions.json", "w"), indent=4)


    def train_evaluate(self, config_file, **kwargs):
        self.train(config_file, **kwargs)
        self.evaluate(config_file, **kwargs)


if __name__ == "__main__":
    fire.Fire(Runner)
