import torch

print(torch.__version__)
print(torch.version.cuda)

if torch.cuda.is_available():
    print("CUDA is available!")
    device = torch.device("cuda")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"Device {i}: {torch.cuda.get_device_name(i)}")
else:
    print("CUDA is not available.")
    device = torch.device("cpu")

