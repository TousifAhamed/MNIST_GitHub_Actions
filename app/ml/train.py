import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import MNISTNet

# Filter out specific warnings
warnings.filterwarnings('ignore', category=UserWarning, message='.*?NumPy.*?')
warnings.filterwarnings('ignore', category=UserWarning, message='.*?Named tensors.*?')

def train_model(epochs=1):
    # Set deterministic behavior
    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    device = torch.device("cpu")
    
    # Simple but effective transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Load dataset with write permissions
    train_dataset = datasets.MNIST(
        './data', 
        train=True, 
        download=True, 
        transform=transform
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        pin_memory=True,
        num_workers=0  # Avoid multiprocessing warnings
    )
    
    model = MNISTNet().to(device)
    param_count = model.count_parameters()
    print(f"\nModel Parameters: {param_count:,}")
    assert param_count < 25000, f"Model has {param_count:,} parameters, exceeding limit of 25,000"
    
    # Simple SGD with momentum
    optimizer = optim.SGD(
        model.parameters(),
        lr=0.1,
        momentum=0.9,
        nesterov=True
    )
    
    # Simple step scheduler
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.1,
        steps_per_epoch=len(train_loader),
        epochs=epochs,
        pct_start=0.1,
        div_factor=10,
        final_div_factor=100,
        anneal_strategy='linear'
    )
    
    criterion = nn.CrossEntropyLoss()
    
    print("\nStarting training...")
    print("-" * 60)
    
    model.train()
    correct = 0
    total = 0
    running_loss = 0.0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        # Forward pass
        optimizer.zero_grad(set_to_none=True)
        output = model(data)
        loss = criterion(output, target)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        # Calculate accuracy
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
        running_loss += loss.item()
        
        if batch_idx % 100 == 0:
            print(f'Progress: [{batch_idx * len(data):>5}/{len(train_loader.dataset)}] '
                  f'({100. * batch_idx / len(train_loader):>3.0f}%) | '
                  f'Loss: {running_loss/100:>7.4f} | '
                  f'Accuracy: {100. * correct / total:>7.2f}%')
            running_loss = 0.0
    
    final_accuracy = 100. * correct / total
    print("-" * 60)
    print(f'Final Training Accuracy: {final_accuracy:.2f}%')
    
    assert final_accuracy >= 95.0, f"Model accuracy {final_accuracy:.2f}% is below required 95%"
    
    return model, final_accuracy

if __name__ == "__main__":
    print("\nMNIST Training with Efficient CNN")
    print("=" * 60)
    model, accuracy = train_model()
    torch.save(model.state_dict(), 'mnist_model.pth')
    print("\nTraining completed successfully!") 