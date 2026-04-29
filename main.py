import numpy as np
def main():
    print("Hello World!")
    layers = [64, 64]
    for r in range(1, len(layers)):
        print(f"Layer {r} has {layers[r]} neurons")
if __name__ == "__main__":
    main()