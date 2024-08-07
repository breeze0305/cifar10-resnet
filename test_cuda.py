def get_device():
    import torch 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    return device

if __name__ == "__main__":
    get_device()