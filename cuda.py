import torch 

def main():
    return torch.cuda.is_available()

if __name__ == "__main__":
    print(main())