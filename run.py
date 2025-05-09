import argparse
import os
import subprocess
import sys

def check_environment():
    """Run environment check"""
    print("Checking environment...")
    subprocess.run([sys.executable, "check_environment.py"])

def train_model(args):
    """Train the brain tumor classification model"""
    print("Training model...")
    cmd = [
        sys.executable, "train.py",
        "--data_dir", args.data_dir,
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--image_size", str(args.image_size),
        "--output_dir", args.output_dir
    ]
    
    if args.no_augmentation:
        cmd.append("--no_augmentation")
    
    subprocess.run(cmd)

def run_app():
    """Run the Streamlit app"""
    print("Starting Streamlit app...")
    subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])

def main():
    parser = argparse.ArgumentParser(description="Brain Tumor Classification Project Runner")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Check environment command
    check_parser = subparsers.add_parser("check", help="Check the environment")
    
    # Train model command
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument("--data_dir", type=str, default="Datasets", help="Directory containing the datasets")
    train_parser.add_argument("--epochs", type=int, default=20, help="Number of epochs to train")
    train_parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    train_parser.add_argument("--image_size", type=int, default=150, help="Size of input images")
    train_parser.add_argument("--output_dir", type=str, default="output", help="Directory to save model and results")
    train_parser.add_argument("--no_augmentation", action="store_true", help="Disable data augmentation")
    
    # Run app command
    app_parser = subparsers.add_parser("app", help="Run the Streamlit app")
    
    args = parser.parse_args()
    
    if args.command == "check":
        check_environment()
    elif args.command == "train":
        train_model(args)
    elif args.command == "app":
        run_app()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()