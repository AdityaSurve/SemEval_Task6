"""
Main entry point for SemEval Task 6
Run this from the SemEval26_Final directory

Usage:
    python run.py augment1     # Augment Task 1 data
    python run.py augment2     # Augment Task 2 data
    python run.py train1       # Train Task 1 model
    python run.py train2       # Train Task 2 model
    python run.py test1        # Test Task 1 model
    python run.py test2        # Test Task 2 model
    python run.py all          # Run everything
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.logger import Logger

logger = Logger()


def show_help():
    print("""
SemEval Task 6 - Evasion Detection
==================================

Usage: python run.py <command>

Commands:
    augment1    - Augment Task 1 (Clarity) training data
    augment2    - Augment Task 2 (Evasion) training data
    train1      - Train Task 1 model
    train2      - Train Task 2 model
    test1       - Test Task 1 model
    test2       - Test Task 2 model
    all         - Run full pipeline (augment + train for both tasks)

Example:
    python run.py train1
    python run.py test2
    """)


def main():
    if len(sys.argv) < 2:
        show_help()
        return
    
    command = sys.argv[1].lower()
    
    if command == "augment1":
        logger.log("Running Task 1 Data Augmentation...", "announce")
        from models.augment_task1 import augment
        augment()
        
    elif command == "augment2":
        logger.log("Running Task 2 Data Augmentation...", "announce")
        from models.augment_task2 import augment
        augment()
        
    elif command == "train1":
        logger.log("Training Task 1 Model...", "announce")
        from models.model1 import train
        train()
        
    elif command == "train2":
        logger.log("Training Task 2 Model...", "announce")
        from models.model2 import train
        train()
        
    elif command == "test1":
        logger.log("Testing Task 1 Model...", "announce")
        from models.test_task1 import test
        test()
        
    elif command == "test2":
        logger.log("Testing Task 2 Model...", "announce")
        from models.test_task2 import test
        test()
        
    elif command == "all":
        logger.log("Running Full Pipeline...", "announce")
        
        logger.log("Step 1/4: Augmenting Task 1 data...", "plain")
        from models.augment_task1 import augment as aug1
        aug1()
        
        logger.log("Step 2/4: Augmenting Task 2 data...", "plain")
        from models.augment_task2 import augment as aug2
        aug2()
        
        logger.log("Step 3/4: Training Task 1 model...", "plain")
        from models.model1 import train as train1
        train1()
        
        logger.log("Step 4/4: Training Task 2 model...", "plain")
        from models.model2 import train as train2
        train2()
        
        logger.log("Pipeline Complete!", "success")
        
    elif command in ["help", "-h", "--help"]:
        show_help()
        
    else:
        logger.log(f"Unknown command: {command}", "error")
        show_help()


if __name__ == "__main__":
    main()


