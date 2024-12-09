# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Trains ML model using training dataset. Saves trained model.
"""

import argparse
from pathlib import Path
import mlflow
import mlflow.sklearn
from story_generator import StoryGenerator

def parse_args():
    parser = argparse.ArgumentParser("train")
    parser.add_argument("--model_output", type=str, help="Path of output model")
    parser.add_argument("--story_parameters", type=str, help="Path to story parameters")
    args = parser.parse_args()
    return args

def main(args):
    # Initialize story generator
    generator = StoryGenerator()
    
    # Log model parameters
    mlflow.log_param("model_name", "Young-Children-Storyteller-Mistral-7B")
    
    # Save the model
    mlflow.sklearn.save_model(
        sk_model=generator,
        path=args.model_output
    )

if __name__ == "__main__":
    mlflow.start_run()
    args = parse_args()
    main(args)
    mlflow.end_run()
    