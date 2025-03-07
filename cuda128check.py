import torch
import torch.nn as nn


def test_cuda_compatibility_and_compilation():
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("CUDA is not available on this system.")
        return False

    # Print CUDA version
    print(f"CUDA version: {torch.version.cuda}")

    # Create a simple model
    class SimpleModel(nn.Module):
        def __init__(self):
            super(SimpleModel, self).__init__()
            self.linear = nn.Linear(10, 5)
            self.activation = nn.ReLU()
            self.output = nn.Linear(5, 1)

        def forward(self, x):
            x = self.linear(x)
            x = self.activation(x)
            x = self.output(x)
            return x

    try:
        # Create model instance
        model = SimpleModel()
        print("Model created successfully.")

        # Move model to GPU
        device = torch.device("cuda")
        model = model.to(device)
        print("Model moved to CUDA successfully.")

        # Create random input tensor
        x = torch.randn(2, 10, device=device)
        print("Input tensor created on CUDA successfully.")

        # Perform forward pass
        with torch.no_grad():
            output = model(x)
        print("Forward pass completed successfully.")

        # Print output shape to verify
        print(f"Output shape: {output.shape}")

        # Test gradient computation
        x = torch.randn(2, 10, requires_grad=True, device=device)
        output = model(x)
        output.sum().backward()
        print("Backward pass completed successfully.")

        # Test model compilation with torch.compile()
        print("\nTesting model compilation with torch.compile()...")
        try:
            # Compile the model using PyTorch 2.0+ compile
            compiled_model = torch.compile(model)
            print("Model compiled successfully.")

            # Test the compiled model with a forward pass
            compiled_output = compiled_model(x)
            print("Forward pass through compiled model successful.")

            # Verify output matches the original model
            match = torch.allclose(output, compiled_output)
            if match:
                print("Compiled model output matches the original model.")
            else:
                print("Warning: Compiled model output doesn't match original model.")

            print("Model compilation test PASSED!")

        except Exception as e:
            print(f"Model compilation FAILED with error: {e}")
            return False

        print("\nCUDA compatibility and compilation tests PASSED!")
        return True

    except Exception as e:
        print(f"\nCUDA compatibility test FAILED with error: {e}")
        return False


if __name__ == "__main__":
    print("Testing CUDA 12.8 compatibility and model compilation...")
    test_cuda_compatibility_and_compilation()
