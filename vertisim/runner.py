import argparse
from .vertisim import VertiSim
import json
import simpy

def parse_args():
    parser = argparse.ArgumentParser(description='Run VertiSim with a configuration file')
    parser.add_argument('-i', '--config', type=str, help='Path to the config.json file')
    return parser.parse_args()

def runner():
    args = parse_args()
    
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        print("Please provide a valid --config argument.")
        return

    env = simpy.Environment()
    
    vertisim = VertiSim(env=env, 
                        config=config)
    vertisim.run()

if __name__ == '__main__':
    runner()
