import torch
from sklearn.metrics import roc_auc_score, mean_absolute_error, root_mean_squared_error, r2_score

# Train target model - multiclass classification
def train_target(
        model, 
        train_loader, 
        eval_loader, 
        criterion, 
        optimizer, 
        device, 
        num_epochs,
        save_path=None,
        verbose=True,
    ):
    """
    Trains a classifier and evaluates on validation set for each epoch.

    Args:
        model: Classifier model (e.g. MLP, CNN, ViT)
        train_loader: DataLoader for training set
        eval_loader: Dataloader for validation set
        criterion: Loss function for target model
        optimizer: Optimizer for target model
        device: Torch device
        num_epochs: Number of training epochs
        verbose: Boolean whether to print progress
    
    Returns:
        model: Trained model
        history: Dictionary containing training/validation accuracy and loss per epoch
    """
    model = model.to(device)

    history = {"train_loss": [], "train_acc": [], "val_acc": [], "val_auc": [], 
               "train_age": [], "train_sex": [], "val_age": [], "val_sex": []}

    best_auc = 0.0
    
    for epoch in range(num_epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0
        train_ages, train_sexes = [], []

        for imgs, labels, age, sex in train_loader:
            imgs = imgs.to(device)
            labels = labels.float().unsqueeze(1).to(device)

            preds = model(imgs)
            loss = criterion(preds, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            predicted = (torch.sigmoid(preds) > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.numel()

            # store age and sex
            train_ages.append(age)
            train_sexes.append(sex)

        train_loss = total_loss / len(train_loader)
        train_accuracy = correct / total

        # Validation
        model.eval()
        val_preds, val_labels, val_ages, val_sexes = [], [], [], []
        with torch.no_grad():
            for val_imgs, val_lbls, val_age, val_sex in eval_loader:
                val_imgs = val_imgs.to(device)
                val_lbls = val_lbls.unsqueeze(1).to(device)
                val_logits = model(val_imgs)
                val_preds.append(torch.sigmoid(val_logits).cpu())
                val_labels.append(val_lbls.cpu())

                # store age and sex
                val_ages.append(val_age)
                val_sexes.append(val_sex)

        val_preds = torch.cat(val_preds)
        val_labels = torch.cat(val_labels)
        val_pred_labels = (val_preds > 0.5).float()
        val_accuracy = (val_pred_labels == val_labels).float().mean().item()

        try:
            val_auc = roc_auc_score(val_labels, val_preds, average='macro')
        except ValueError:
            val_auc = float('nan')

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_accuracy)
        history["val_acc"].append(val_accuracy)
        history["val_auc"].append(val_auc)

        history["train_age"] = torch.cat(train_ages).tolist()
        history["train_sex"] = torch.cat(train_sexes).tolist()
        history["val_age"] = torch.cat(val_ages).tolist()
        history["val_sex"] = torch.cat(val_sexes).tolist()

        if verbose:
            print(f"Epoch {epoch+1}/{num_epochs}: "
                  f"Loss = {train_loss:.4f}, "
                  f"Train Acc = {train_accuracy:.4f}, "
                  f"Val Acc = {val_accuracy:.4f}, "
                  f"Val AUC (avg) = {val_auc:.4f}")
            
        # Save best model
        if save_path and val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), save_path)

    return model, history


# Train target model - regression

def train_regression(
    model, 
    train_loader, 
    eval_loader, 
    criterion, 
    optimizer, 
    device, 
    num_epochs,
    save_path=None,
    verbose=True,
):
    """
    Trains a model for a regression set and evaluates on validation set for each epoch.

    Args:
        model: Regression model
        train_loader: DataLoader for training set
        eval_loader: Dataloader for validation set
        criterion: Loss function for target model
        optimizer: Optimizer for target model
        device: Torch device
        num_epochs: Number of training epochs
        verbose: Boolean whether to print progress
    
    Returns:
        model: Trained model
        history: Dictionary containing training/validation accuracy and loss per epoch
    """
    model = model.to(device)

    history = {"train_loss": [], "val_loss": [], 
               "val_mae": [], "val_rmse": [], "val_r2": [],
               "train_age": [], "val_age": [], "train_sex": [], "val_sex": []}

    best_r2 = -float("inf")
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        train_ages, train_sexes = [], []

        for imgs, age, sex in train_loader:
            imgs, age = imgs.to(device), age.to(device)
            preds = model(imgs)
            loss = criterion(preds.squeeze(), age)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # store age and sex
            train_ages.append(age)
            train_sexes.append(sex)

        train_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        val_preds, val_labels, val_ages, val_sexes = [], [], [], []
        with torch.no_grad():
            for val_imgs, val_age, val_sex in eval_loader:
                val_imgs, val_age = val_imgs.to(device), val_age.to(device)
                val_outputs = model(val_imgs)
                val_preds.append(val_outputs.cpu())
                val_labels.append(val_age.cpu())

                # store age and sex
                val_ages.append(val_age)
                val_sexes.append(val_sex)

        val_preds = torch.cat(val_preds).squeeze().numpy()
        val_labels = torch.cat(val_labels).squeeze().numpy()
        
        val_loss = criterion(torch.tensor(val_preds).squeeze(), torch.tensor(val_labels)).item()
        val_mae = mean_absolute_error(val_labels, val_preds)
        val_rmse = root_mean_squared_error(val_labels, val_preds)
        val_r2 = r2_score(val_labels, val_preds)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_mae"].append(val_mae)
        history["val_rmse"].append(val_rmse)
        history["val_r2"].append(val_r2)

        history["train_age"].append(torch.cat(train_ages).tolist())
        history["train_sex"].append(torch.cat(train_sexes).tolist())
        history["val_age"].append(torch.cat(val_ages).tolist())
        history["val_sex"].append(torch.cat(val_sexes).tolist())

        if verbose:
            print(f"Epoch {epoch+1}/{num_epochs}: "
                  f"Train Loss = {train_loss:.4f}, "
                  f"Val Loss = {val_loss:.4f}, "
                  f"MAE = {val_mae:.2f}, RMSE = {val_rmse:.2f}, R^2: {val_r2:.4f}")
            
        # Save best model
        if save_path and val_r2 > best_r2:
            best_r2 = val_r2
            torch.save(model.state_dict(), save_path)

    return model, history