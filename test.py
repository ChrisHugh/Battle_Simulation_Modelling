import torch
print(torch.cuda.device_count())        # Number of GPUs detected
print(torch.cuda.get_device_name(0))    # Name of the first GPU
print(torch.cuda.current_device())  