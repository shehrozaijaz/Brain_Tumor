import os
import sys
import importlib
import pkg_resources

def check_python_version():
    """Check Python version"""
    python_version = sys.version.split()[0]
    print(f"Python version: {python_version}")
    
    major, minor, _ = python_version.split('.')
    if int(major) < 3 or (int(major) == 3 and int(minor) < 7):
        print("⚠️ Warning: Python 3.7 or higher is recommended for this project")
    else:
        print("✅ Python version is compatible")

def check_required_packages():
    """Check if all required packages are installed"""
    required_packages = [
        'numpy',
        'pandas',
        'opencv-python',
        'tensorflow',
        'scikit-learn',
        'matplotlib',
        'seaborn',
        'streamlit',
        'pillow',
        'tqdm'
    ]
    
    print("\nChecking required packages:")
    
    installed_packages = {pkg.key: pkg.version for pkg in pkg_resources.working_set}
    
    all_installed = True
    for package in required_packages:
        package_name = package.lower()
        if package_name == 'opencv-python':
            package_name = 'cv2'
        elif package_name == 'pillow':
            package_name = 'PIL'
        elif package_name == 'scikit-learn':
            package_name = 'sklearn'
        
        try:
            if package_name == 'PIL':
                module = importlib.import_module(package_name)
            else:
                module = importlib.import_module(package_name.split('-')[0])
                
            version = getattr(module, '__version__', installed_packages.get(package.lower(), "unknown"))
            print(f"✅ {package} (version: {version})")
        except ImportError:
            print(f"❌ {package} is not installed")
            all_installed = False
    
    return all_installed

def check_gpu():
    """Check if TensorFlow can detect a GPU"""
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"\n✅ GPU detected: {len(gpus)} GPU(s) available")
            for gpu in gpus:
                print(f"   - {gpu}")
        else:
            print("\n⚠️ No GPU detected. Training will use CPU, which may be slower.")
    except ImportError:
        print("\n❌ TensorFlow not installed. Cannot check GPU.")

def check_dataset():
    """Check if the dataset directories exist"""
    base_dir = "Datasets"
    subdirs = ["Training", "Testing"]
    categories = ["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"]
    
    print("\nChecking dataset structure:")
    
    if not os.path.exists(base_dir):
        print(f"❌ Dataset directory '{base_dir}' not found")
        return False
    
    all_good = True
    for subdir in subdirs:
        subdir_path = os.path.join(base_dir, subdir)
        if not os.path.exists(subdir_path):
            print(f"❌ '{subdir}' directory not found")
            all_good = False
            continue
        
        for category in categories:
            category_path = os.path.join(subdir_path, category)
            if not os.path.exists(category_path):
                print(f"❌ '{category}' directory not found in '{subdir}'")
                all_good = False
                continue
            
            files = os.listdir(category_path)
            if not files:
                print(f"⚠️ '{category}' directory in '{subdir}' is empty")
            else:
                print(f"✅ '{category}' in '{subdir}': {len(files)} images found")
    
    return all_good

def main():
    """Main function to check the environment"""
    print("="*60)
    print("BRAIN TUMOR CLASSIFICATION - ENVIRONMENT CHECK")
    print("="*60)
    
    check_python_version()
    
    packages_ok = check_required_packages()
    
    check_gpu()
    
    dataset_ok = check_dataset()
    
    print("\n" + "="*60)
    if packages_ok and dataset_ok:
        print("✅ All checks passed! You're ready to run the project.")
    else:
        print("⚠️ Some checks failed. Please address the issues above.")
    print("="*60)

if __name__ == "__main__":
    main()