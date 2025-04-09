import torch
import numpy as np
from tqdm import tqdm
from model import Extractor, Predictor, Discriminator
import torch.optim as optim
import torch.nn as nn

def train_model(opt, source_dl, target_dl):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    extractor = Extractor().to(device)
    predictor = Predictor().to(device)
    discriminator = Discriminator().to(device)
    
    optimizer = optim.Adam(list(extractor.parameters()) +
                        list(predictor.parameters()) +
                        list(discriminator.parameters()), lr=opt.lr)
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda e: opt.lamda ** e)
    criterion = torch.nn.CrossEntropyLoss()

    pbar = tqdm(range(opt.epochs), desc="Training")

    len_dl = min(len(source_dl), len(target_dl))

    for epoch in pbar:
        
        start_steps = epoch * len_dl
        total_steps = opt.epochs * len_dl
        
        extractor.train()
        predictor.train()
        discriminator.train()
        
        total = 0
        d_correct = 0
        
        for ix, (batch_1, batch_2) in enumerate(iter(zip(source_dl, target_dl))):
            
            p = float(ix + start_steps) / total_steps
            lamda = 2. / (1. + np.exp(-10 * p)) - 1
            
            x_s, y_s = batch_1
            x_t, y_t = batch_2
            d_s = torch.zeros(y_s.size(0)).long().to(device).float()
            d_t = torch.ones(y_t.size(0)).long().to(device).float()
            
            combined_x = torch.cat((x_s, x_t), 0)
            combined_d = torch.cat((d_s, d_t), 0)
            source_feature = extractor(x_s)
            combined_feature = extractor(combined_x)
            
            source_pred = predictor(source_feature)
            domain_pred = discriminator(combined_feature, lamda)
            
            loss_y = criterion(source_pred, y_s)
            loss_d = nn.BCELoss()(domain_pred.view(-1), combined_d)
            
            batch_loss = loss_y + opt.alpha * loss_d
            
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            
            d_prediction = torch.where(domain_pred>=0.5, 1.0, 0.0).view(-1)
            d_correct += (d_prediction == combined_d).sum().item()
            total = total + (len(batch_1[0]) + len(batch_2[0]))
            d_acc = 100 * float(d_correct) / total
                        
            pbar.set_postfix({
                'epoch': epoch + 1,
                'task_loss': f"{loss_y.item():.4f}",
                'domain_loss': f"{loss_d.item():.4f}",
                'domain_acc': f"{d_acc:.2f}%"
            })

        scheduler.step()

    return extractor, predictor