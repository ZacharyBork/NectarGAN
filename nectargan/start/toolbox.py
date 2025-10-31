from nectargan.start.torch_check import validate_torch
validate_torch()

from nectargan.toolbox.run import Interface

def main():
    interface = Interface()
    interface.run()

if __name__ == "__main__":
    main()    
