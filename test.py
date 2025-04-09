import torch
import numpy as np

def evaluate(test_dl, extractor, predictor):
    
    extractor.eval()
    predictor.eval()

    preds, labels = [], []

    with torch.no_grad():
        for x, y in test_dl:
            feature = extractor(x)
            y_pred = predictor(feature)
            preds.append(torch.argmax(y_pred, 1).cpu())
            labels.append(y.cpu())

    preds = torch.cat(preds).numpy()
    labels = torch.cat(labels).numpy()
    acc = (preds == labels).mean() * 100
    return acc
