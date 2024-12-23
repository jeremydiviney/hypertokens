
print("Test")

import torch
import sys

def check_cuda() -> None:
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        print(f"CUDA is available. Device count: {device_count}")

        for device_index in range(device_count):
            device = torch.cuda.get_device_properties(device_index)
            print(f"\nDevice {device_index}: {device.name}")
            # Print all attributes of the device
            for attr in dir(device):
                if not attr.startswith('_'):  # Skip private attributes
                    value = getattr(device, attr)
                    print(f"  {attr}: {value}")
    else:
        print("CUDA is not available.")

check_cuda() 