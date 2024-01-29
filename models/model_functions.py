import torch
from tqdm.auto import tqdm, trange

def model_training(n_epochs, model, train_loader, val_loader, optimizer, criterion, device):
  train_losses = []
  train_accs = []
  val_losses = []
  val_accs = []
  best_val_acc = 0.0
  best_epoch = 0

  for epoch in range(1, n_epochs + 1):
    train_progress_bar = tqdm(train_loader, desc=f"Training epoch {epoch}/{n_epochs}", leave=False, unit="mini-batch")
    train_acc, train_loss = train(model, train_loader, optimizer, criterion, device, train_progress_bar)
    print(f'Epoch: {epoch:03d}')
    print(f'\tTrain Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%')

    val_progress_bar = tqdm(val_loader, desc=f"Validation epoch {epoch}/{n_epochs}", leave=False, unit="mini-batch")
    _, _, val_acc, val_loss = eval(model, val_loader, criterion, device, val_progress_bar)
    print(f'\tVal Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.2f}%')

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_epoch = epoch
        torch.save(model.state_dict(), 'best_model.pth')

    train_losses.append(train_loss)
    train_accs.append(train_acc)
    val_losses.append(val_loss)
    val_accs.append(val_acc)

  return train_losses, train_accs, val_losses, val_accs, best_epoch


def train(model, data_loader, optimizer, criterion, device, progress_bar):
    total = 0.
    correct = 0.
    total_loss = 0.

    model.train()

    for images, labels in progress_bar:
          images = images.to(device)
          labels = labels.to(device)

          optimizer.zero_grad()

          images_size = images.shape[0]
          total += images_size

          out = model(images)
          loss = criterion(out, labels)

          total_loss += loss.item()
          _, pred = torch.max(out, dim=1)
          correct += torch.sum(pred == labels)

          loss.backward()
          optimizer.step()

    return float(correct * 100. / total), total_loss / len(data_loader)

def eval(model, data_loader, criterion, device, progress_bar=None):
    total = 0.
    correct = 0.
    total_loss = 0.
    true_labels = []
    predicted_labels = []

    if progress_bar is None:
       progress_bar = tqdm(data_loader, desc=f"Test", leave=False, unit="mini-batch")

    model.eval()
    with torch.no_grad():

        for images, labels in progress_bar:

            images = images.to(device)
            labels = labels.to(device)

            images_size = images.shape[0]
            total += images_size

            out = model(images)
            total_loss += criterion(out, labels).item()

            _, pred = torch.max(out, dim=1)
            correct += torch.sum(pred == labels)

            # for test metrics
            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(pred.cpu().numpy())

    return true_labels, predicted_labels, float(correct * 100. / total), total_loss / len(data_loader)

# This function can be made more general by allowing for more flexible layer freezing options.
def freeze(model, classification_layer):
    for param in model.parameters():
      param.requires_grad = False

    for param in classification_layer.parameters():
      param.requires_grad = True

# Prints the number of total parameters and the number of trainable parameters in the model.
def params_info(model):
   total_params = sum(p.numel() for p in model.parameters())
   print(f'{total_params:,} total parameters.')

   total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
   print(f'{total_trainable_params:,} training parameters.')


   
        
    
    


    
    
        