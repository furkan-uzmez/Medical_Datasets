import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, roc_auc_score, confusion_matrix
import numpy as np

def evaluate_model(model, dataloader, device, num_classes=None):
    """
    Evaluates a PyTorch model on a dataloader.
    
    Args:
        model: PyTorch model.
        dataloader: PyTorch dataloader.
        device: 'cuda' or 'cpu'.
        num_classes: Number of classes (optional, for AUC).
        
    Returns:
        metrics: Dictionary containing accuracy, f1, precision, recall, and report.
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = [] # For AUC if needed

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloader):
            if i % 10 == 0:
                print(f"Processing batch {i}/{len(dataloader)}...")
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            
            # Assuming outputs are logits
            if num_classes and num_classes > 1:
                 probs = torch.softmax(outputs, dim=1)
            else:
                 probs = torch.sigmoid(outputs) # Binary
                 
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    metrics = {}
    metrics['accuracy'] = accuracy_score(all_labels, all_preds)
    
    # Weighted metrics for handling class imbalance
    metrics['f1_weighted'] = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    metrics['precision_weighted'] = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    metrics['recall_weighted'] = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    metrics['classification_report'] = classification_report(all_labels, all_preds, zero_division=0)
    
    # AUC calculation
    try:
        if num_classes == 2:
             # Binary case, take prob of positive class (index 1)
             # Check if all_probs has shape (N, 2)
             if all_probs.shape[1] == 2:
                 metrics['auc'] = roc_auc_score(all_labels, all_probs[:, 1])
             else:
                 metrics['auc'] = roc_auc_score(all_labels, all_probs)
        elif num_classes and num_classes > 2:
             # Multi-class OvR
             metrics['auc'] = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='weighted')
    except Exception as e:
        print(f"AUC Calculation skipped due to error: {e}")
        metrics['auc'] = None

    return metrics
