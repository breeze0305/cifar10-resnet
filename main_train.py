import torch
import wandb
import torchvision.models as models

from torchvision.models.resnet import ResNet50_Weights
from tqdm import tqdm
from torch.utils.data import DataLoader

from test_cuda import get_device
from main_dataset import Cifar10Dataset

def get_arg():
    arg = {
        "batch_size" : 512,
        "lr" : 0.001,
        "epoch" : 100,
        "device" : get_device(),
        "number_worker" : 10,
        "root_path" : ".",
        "class" : ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"],
    }
    return arg

def adjust_learning_rate(optimizer, epoch, initial_lr):
    warmup_epochs = 5
    if epoch <= warmup_epochs:
        lr = initial_lr * epoch / warmup_epochs
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    elif epoch % 10 == 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] / 3 * 2

def main(arg: dict):    
    train_dataset = Cifar10Dataset("train", is_transform=True)
    train_dataloader = DataLoader(train_dataset, batch_size=arg["batch_size"], shuffle=True, 
                                    pin_memory=True, num_workers=arg["number_worker"])
    
    test_dataset = Cifar10Dataset("val", is_transform=False)
    test_datalooader = DataLoader(test_dataset, batch_size=arg["batch_size"], shuffle=False, 
                                    pin_memory=True, num_workers=arg["number_worker"])
    
    model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    model.to(arg["device"])
    
    critirien = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=arg["lr"])
    
    for epoch in range(arg["epoch"]):
        model.train()
        total = 0
        correct = 0
        train_loss = 0.0
        train_accuracy = 0.0
        adjust_learning_rate(optimizer, epoch, arg["lr"])
        
        for i, (image, label) in enumerate(tqdm(train_dataloader,desc=f"train epoch: {epoch}")):
            image = image.to(arg["device"])
            label = label.to(arg["device"])
            
            output = model(image)
            loss = critirien(output, label)
            train_loss += loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            _, predict = torch.max(output, 1)
                    
            total += label.size(0)
            correct += (predict == label).sum().item()
                
        train_accuracy = correct / total
        print(f"epoch: {epoch}, loss: {train_loss}, train_accuracy: {train_accuracy}")
        
        # validation--
        if epoch > 30 :
            model.eval()
            test_accuracy = 0.0
            total = 0
            correct = 0
            last_loss = test_loss
            test_loss = 0.0
            
            with torch.no_grad():
                for i, (image, label) in tqdm(enumerate(test_datalooader),desc=f"test epoch: {epoch}"):
                    image = image.to(arg["device"])
                    label = label.to(arg["device"])
                    
                    output = model(image)
                    loss = critirien(output, label)
                    test_loss += loss.item()
                    _, predict = torch.max(output, 1)
                    
                    total += label.size(0)
                    correct += (predict == label).sum().item()
                    
                test_accuracy = correct / total
                print(f"epoch: {epoch}, test_loss: {test_loss}, test_accuracy: {test_accuracy}")
                
                if last_loss > test_loss:
                    torch.save(model.state_dict(), f"weight/model_{epoch}.pth")
        else:
            test_accuracy = 0.0
            test_loss = 0.0
        # validation--
        
        
        wandb.log({"epoch":epoch,"train_loss": train_loss,"test_loss":test_loss, 
                    "learning_rate": optimizer.param_groups[0]['lr'],"train_accuracy": train_accuracy,
                    "test_accuracy": test_accuracy})
        
        
if __name__ == "__main__":
    args = get_arg()
    wandb.init(
        project="cifar10",
        mode = "online",
        config=args
            )

    main(args)