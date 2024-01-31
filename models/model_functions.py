import torch
import time
from tqdm.auto import tqdm

def model_training(n_epochs, model, train_loader, val_loader, optimizer, criterion, scheduler, device):
  train_losses = []
  train_accs = []
  val_losses = []
  val_accs = []
  best_val_acc = 0.0
  best_epoch = 0
  
  start_time = time.time()
  for epoch in range(1, n_epochs + 1):
    train_progress_bar = tqdm(train_loader, desc=f"Training epoch {epoch}/{n_epochs}", leave=False, unit="mini-batch")
    train_acc, train_loss = train(model, train_loader, optimizer, criterion, device, train_progress_bar)
    print(f'Epoch: {epoch:03d}')
    print(f'\tTrain Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%')

    val_progress_bar = tqdm(val_loader, desc=f"Validation epoch {epoch}/{n_epochs}", leave=False, unit="mini-batch")
    _, _, val_acc, val_loss = eval(model, val_loader, criterion, device, val_progress_bar)
    print(f'\tVal Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.2f}%')

    if (scheduler):
        scheduler.step(val_loss)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_epoch = epoch
        torch.save(model.state_dict(), 'best_model.pth')

    train_losses.append(train_loss)
    train_accs.append(train_acc)
    val_losses.append(val_loss)
    val_accs.append(val_acc)

    # Calculate elapsed time and remaining time
    elapsed_time = time.time() - start_time
    avg_time_per_epoch = elapsed_time / epoch
    remaining_epochs = n_epochs - epoch
    remaining_time = avg_time_per_epoch * remaining_epochs

    # Convert elapsed time in hh:mm:ss
    elapsed_time_hr, rem = divmod(elapsed_time, 3600)
    elapsed_time_min, elapsed_time_sec = divmod(rem, 60)
    
    # Convert remaining time in hh:mm:ss
    remaining_time_hr, rem = divmod(remaining_time, 3600)
    remaining_time_min, remaining_time_sec = divmod(rem, 60)

    print(f"\tElapsed time: {elapsed_time_hr:.0f}h {elapsed_time_min:.0f}m {elapsed_time_sec:.0f}s, Remaining Time: {remaining_time_hr:.0f}h {remaining_time_min:.0f}m {remaining_time_sec:.0f}s")
    
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

# Accepts a list of trainable layers. Makes sure that only these layers are not frozen.
def freeze(model, trainable_layers):
    
    for param in model.parameters():
      param.requires_grad = False

    for layer in trainable_layers:
        for param in layer.parameters():
          param.requires_grad = True

    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')

    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} trainable parameters.')






   
        
    
    


    
    
        