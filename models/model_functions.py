import torch

def model_training(n_epochs, model, train_loader, val_loader, optimizer, criterion, device):
  train_losses = []
  train_accs = []
  val_losses = []
  val_accs = []
  best_val_acc = 0.0
  best_epoch = 0

  for epoch in range(1, n_epochs + 1):

    train_acc, train_loss = train(model, train_loader, optimizer, criterion, device)
    val_acc, val_loss = eval(model, val_loader, criterion, device)

    print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
    print(f'Epoch: {epoch:03d}, Train Accuracy: {train_acc:.4f}%, Val Accuracy: {val_acc:.4f}%')

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_epoch = epoch
        torch.save(model.state_dict(), 'best_model.pth')

    train_losses.append(train_loss)
    train_accs.append(train_acc)
    val_losses.append(val_loss)
    val_accs.append(val_acc)

  return train_losses, train_accs, val_losses, val_accs, best_epoch


def train(model, data_loader, optimizer, criterion, device):
    total = 0.
    correct = 0.
    total_loss = 0.

    model.train()

    for images, labels in data_loader:
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

def eval(model, data_loader, criterion, device):
    total = 0.
    correct = 0.
    total_loss = 0.

    model.eval()

    with torch.no_grad():

        for images, labels in data_loader:

            images = images.to(device)
            labels = labels.to(device)

            images_size = images.shape[0]
            total += images_size

            out = model(images)
            total_loss += criterion(out, labels).item()

            _, pred = torch.max(out, dim=1)
            correct += torch.sum(pred == labels)

    return float(correct * 100. / total), total_loss / len(data_loader)