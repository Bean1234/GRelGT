# -*- coding: utf-8 -*-
import os
import logging
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from sklearn import metrics
empty=0
edges=0
total=0


class Trainer():
    def __init__(self, params, graph_classifier, train, line_graph, valid_evaluator=None):
        self.graph_classifier = graph_classifier
        self.valid_evaluator = valid_evaluator
        self.params = params
        self.critic = ['auc', 'auc_pr', 'mrr']
        self.train_data = train

        self.lg = line_graph

        self.updates_counter = 0

        model_params = list(self.graph_classifier.parameters())
        logging.info('Total number of parameters: %d' % sum(map(lambda x: x.numel(), model_params)))

        if params.optimizer == "SGD":
            self.optimizer = optim.SGD(model_params, lr=params.lr, momentum=0.9, weight_decay=self.params.l2)
        if params.optimizer == "Adam":
            self.optimizer = optim.Adam(model_params, lr=params.lr, weight_decay=self.params.l2)

        self.criterion = nn.MarginRankingLoss(self.params.margin, reduction='sum')
        # self.criterion = nn.BCELoss()
        if self.params.loss:
            logging.info('using abs loss!')

        self.reset_training_state()

    def reset_training_state(self):
        self.best_metric = 0
        self.last_metric = 0
        self.not_improved_count = 0

    def train_epoch(self, epoch):
        total_loss = 0

        all_labels = []
        all_scores = []

        dataloader = DataLoader(self.train_data, batch_size=self.params.batch_size, shuffle=True,
                                num_workers=self.params.num_workers, collate_fn=self.params.collate_fn)
        model_params = list(self.graph_classifier.parameters())

        for b_idx, batch in tqdm(enumerate(dataloader),total = len(dataloader)) :
            data_pos, targets_pos, data_neg, targets_neg = self.params.move_batch_to_device(batch, self.params.device)
            self.graph_classifier.train()
            self.optimizer.zero_grad()
            
            score_pos = self.graph_classifier(data_pos, self.lg, b_idx)  
            score_neg = self.graph_classifier(data_neg, self.lg, b_idx)  
            
            if self.params.loss == 1:
                loss = torch.abs(torch.sum(torch.sum(score_neg, dim=1) + torch.clamp(self.params.margin - score_pos, min=0)))
            else:
                loss = self.criterion(score_pos.squeeze(), score_neg.mean(dim=1), torch.Tensor([1]*score_pos.shape[0]).to(device=self.params.device))

            loss.backward(retain_graph=False)
            self.optimizer.step()
            self.updates_counter += 1

            with torch.no_grad():
                all_scores += score_pos.squeeze(1).detach().cpu().tolist() + score_neg.squeeze(1).detach().cpu().tolist()
                all_labels += targets_pos.tolist() + targets_neg.tolist()
                total_loss += loss.item()
                 
        if self.valid_evaluator:
            tic = time.time()
            result = self.valid_evaluator.eval()

            logging.info('Performance: ' + str(result) + 'in ' + str(time.time() - tic))

            if result[self.critic[self.params.critic]] >= self.best_metric:
                self.save_classifier()
                self.best_metric = result[self.critic[self.params.critic]]
                self.not_improved_count = 0
            else:
                self.not_improved_count += 1
                if self.not_improved_count > self.params.early_stop:
                    logging.info(
                        f"Validation performance didn\'t improve for {self.params.early_stop} epochs. Training stops.")
            self.last_metric = result[self.critic[self.params.critic]]

        auc = metrics.roc_auc_score(all_labels, all_scores)
        auc_pr = metrics.average_precision_score(all_labels, all_scores)

        weight_norm = sum(map(lambda x: torch.norm(x), model_params))

        return total_loss, auc, auc_pr, weight_norm

    def train(self):
        self.reset_training_state()
        for epoch in range(1, self.params.num_epochs + 1):
            self.params.epoch = epoch
            time_start = time.time()
            loss, auc, auc_pr, weight_norm = self.train_epoch(epoch)
            time_elapsed = time.time() - time_start
            logging.info(
                f'Epoch {epoch} with loss: {loss}, training auc: {auc}, training auc_pr: {auc_pr}, best validation AUC: {self.best_metric}, weight_norm: {weight_norm} in {time_elapsed}')

            if epoch % self.params.save_every == 0:  
                torch.save(self.graph_classifier, os.path.join(self.params.exp_dir, 'graph_classifier_chk.pth'))

    def save_classifier(self):
        torch.save(self.graph_classifier, os.path.join(self.params.exp_dir, 'best_graph_classifier.pth'))
        logging.info(f'Better models found w.r.t {self.critic[self.params.critic]}. Saved it!')
