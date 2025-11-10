"""
Install MGBT package in R.
"""

import os

# Set R_HOME
R_HOME = r"C:\Program Files\R\R-4.5.1"
os.environ['R_HOME'] = R_HOME

# Add R to PATH
R_BIN = os.path.join(R_HOME, 'bin', 'x64')
if R_BIN not in os.environ['PATH']:
    os.environ['PATH'] = R_BIN + os.pathsep + os.environ['PATH']

print("Installing MGBT package in R...")

try:
    import rpy2.robjects as ro
    from rpy2.robjects.packages import importr
    
    # Import utils for package installation
    utils = importr('utils')
    
    # Set CRAN mirror
    ro.r('options(repos = c(CRAN = "https://cloud.r-project.org"))')
    
    # Install MGBT package
    print("Downloading and installing MGBT from CRAN...")
    utils.install_packages('MGBT')
    
    print("\nMGBT package installed successfully!")
    
    # Verify installation
    mgbt = importr('MGBT')
    print("MGBT package loaded successfully!")
    
except Exception as e:
    print(f"\nError installing MGBT: {e}")
    print("\nAlternative: Install manually in R console:")
    print('  install.packages("MGBT")')
