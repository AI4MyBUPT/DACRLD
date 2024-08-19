import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


def cos_sim(a: torch.Tensor, b: torch.Tensor):
    """
    Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.
    :return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
    return torch.mm(a_norm, b_norm.transpose(0, 1))


class OrdinalCrossEntropyLoss(nn.Module):
    def __init__(self,label_smoothing=0.):
        super().__init__()
        self.label_smoothing = label_smoothing
        self.crossentropy_smoothed = nn.CrossEntropyLoss(reduction='none',label_smoothing=label_smoothing)
        self.crossentropy = nn.CrossEntropyLoss(reduction='none')

    def forward(self,logits,labels):
        # logits: (batch_size, num_classes, *)
        # labels: (batch_size, *)
        # return: scalar
        # |true_class-pred_class|/num_classes*corssentropy(logits,labels)+crossentropy_smoothed(logits,labels)
        num_classes = logits.size(1)
        pred_class = logits.argmax(dim=1) # (batch_size, *)
        loss_1 = torch.abs(labels-pred_class)/num_classes*self.crossentropy(logits,labels)
        loss_2 = self.crossentropy_smoothed(logits,labels)
        loss = loss_1+loss_2
        return loss.mean()
    
class NTXent(nn.Module):
    def __init__(self, temperature=0.07):
        super(NTXent, self).__init__()
        self.loss = nn.LogSoftmax(dim=1)
        self.tau = temperature

    def forward(self, audio_embeds, text_embeds, labels):

        n = audio_embeds.shape[0]

        a2t = cos_sim(audio_embeds, text_embeds) / self.tau
        t2a = cos_sim(text_embeds, audio_embeds) / self.tau

        mask = labels.expand(n, n).eq(labels.expand(n, n).t()).to(a2t.device)
        mask_diag = mask.diag()
        mask_diag = torch.diag_embed(mask_diag)
        mask = mask ^ mask_diag

        a2t_loss = - self.loss(a2t).masked_fill(mask, 0).diag().mean()
        t2a_loss = - self.loss(t2a).masked_fill(mask, 0).diag().mean()

        loss = 0.5 * a2t_loss + 0.5 * t2a_loss

        return loss

class BLEUSimMatrixLoss(nn.Module):
    """
    get BLEU similarity matrix of GT captions
    get similarity matrix of latent features
    calculate MSE loss between two similarity matrix
    """
    def __init__(self, max_ngram=4):
        super().__init__()
        self.max_ngram = max_ngram
        self.loss = nn.MSELoss()

    def forward(self, audio_embeds, text_captions):
        """
        Args:
            audio_embeds: (batch_size, embed_dim)
            text_captions: [str]*batch_size
        """
        # get BLEU similarity matrix of GT captions
        gt_sim_matrix = self.get_bleu_sim_matrix(text_captions)
        # get similarity matrix of latent features
        latent_sim_matrix = self.get_latent_sim_matrix(audio_embeds)
        # calculate MSE loss between two similarity matrix
        loss = self.loss(gt_sim_matrix, latent_sim_matrix)
        return loss
    
    def get_bleu_sim_matrix(self, text_captions):
        """
        Args:
            text_captions: [str]*batch_size
        Return:
            sim_matrix: (batch_size, batch_size)
        """
        batch_size = len(text_captions)
        sim_matrix = np.zeros((batch_size, batch_size))
        for i in range(batch_size):
            for j in range(batch_size):
                sim_matrix[i, j] = self.get_bleu_sim(text_captions[i], text_captions[j])
        return torch.tensor(sim_matrix).float()
    
    def get_latent_sim_matrix(self, audio_embeds):
        """
        Args:
            audio_embeds: (batch_size, embed_dim)
        Return:
            sim_matrix: (batch_size, batch_size)
        """
        batch_size = audio_embeds.size(0)
        sim_matrix = torch.zeros((batch_size, batch_size))
        for i in range(batch_size):
            for j in range(batch_size):
                sim_matrix[i, j] = cos_sim(audio_embeds[i], audio_embeds[j])
        return sim_matrix
    
    def get_bleu_sim(self, caption1, caption2):
        """
        Args:
            caption1: str
            caption2: str
        Return:
            bleu_sim: float
        """
        bleu_sim = sentence_bleu(references=[caption2.split()], hypothesis=caption1.split(), smoothing_function=SmoothingFunction().method4,)
        return bleu_sim
