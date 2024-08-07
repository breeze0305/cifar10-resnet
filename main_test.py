import torch
import torchvision.models as models

from tqdm import tqdm
from torch.utils.data import DataLoader

from test_cuda import get_device
from main_dataset import Cifar10Dataset

def get_arg():
    arg = {
        "batch_size" : 1024,
        "lr" : 0.001,
        "epoch" : 100,
        "device" : get_device(),
        "number_worker" : 1,
        "root_path" : ".",
        "class" : ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"],
    }
    return arg

def main(arg: dict):    
    test_dataset = Cifar10Dataset("val", is_transform=False)
    test_datalooader = DataLoader(test_dataset, batch_size=arg["batch_size"], shuffle=False, 
                                    pin_memory=True)
    
    # model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    # model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model = models.resnet50()
    model.load_state_dict(torch.load("weight/model_37.pth"))
    model.to(arg["device"])

    model.eval()
    test_accuracy = 0.0
    total = 0
    correct = 0

    with torch.no_grad():
        for i, (image, label) in enumerate(tqdm(test_datalooader,desc=f"test")):
            image = image.to(arg["device"])
            label = label.to(arg["device"])
            
            output = model(image)
            _, predict = torch.max(output, 1)
            
            total += label.size(0)
            correct += (predict == label).sum().item()
            
        test_accuracy = correct / total
        print(f"test_accuracy: {test_accuracy}")
                
if __name__ == "__main__":
    args = get_arg()
    main(args)