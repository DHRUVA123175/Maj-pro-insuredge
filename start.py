#!/usr/bin/env python3
"""
InsurEdge AI Startup Script
Handles environment setup and application launch
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version.split()[0]}")
    return True

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = [
        'flask', 'tensorflow', 'opencv-python', 'pymongo', 
        'scikit-learn', 'pillow', 'numpy'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} - Missing")
    
    if missing_packages:
        print(f"\n📦 Installing missing packages: {', '.join(missing_packages)}")
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + missing_packages)
            print("✅ Dependencies installed successfully")
        except subprocess.CalledProcessError:
            print("❌ Failed to install dependencies")
            return False
    
    return True

def check_mongodb():
    """Check MongoDB connection"""
    try:
        from pymongo import MongoClient
        import os
        
        # Check if MONGODB_URI is set in environment
        mongodb_uri = os.getenv('MONGODB_URI')
        
        if mongodb_uri:
            print("🔗 Using MongoDB Atlas connection...")
            client = MongoClient(mongodb_uri, serverSelectionTimeoutMS=10000)
        else:
            print("🔗 Using local MongoDB connection...")
            client = MongoClient('mongodb://localhost:27017/', serverSelectionTimeoutMS=5000)
        
        client.admin.command('ping')
        print("✅ MongoDB connection successful")
        client.close()
        return True
    except Exception as e:
        print(f"⚠️  MongoDB connection failed: {e}")
        print("💡 Make sure MongoDB is running or set MONGODB_URI environment variable")
        print("📖 Check mongodb_config.txt for configuration template")
        return False

def create_directories():
    """Create necessary directories"""
    directories = ['uploads', 'models']
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created directory: {directory}")

def setup_environment():
    """Setup environment variables"""
    env_file = Path('.env')
    
    if not env_file.exists():
        print("📝 Creating .env file...")
        print("💡 Please copy the content from mongodb_config.txt to .env file")
        print("🔑 Update MONGODB_URI with your MongoDB Atlas connection string")
        print("🔐 Update JWT_SECRET_KEY with a secure random string")
        
        # Create a basic .env file
        env_content = """# InsurEdge AI Environment Configuration
# Copy from mongodb_config.txt and update with your actual values

JWT_SECRET_KEY=your-super-secret-jwt-key-change-this-in-production
MONGODB_URI=mongodb+srv://your_username:your_password@your_cluster.mongodb.net/insuredge_db?retryWrites=true&w=majority
FLASK_ENV=development
FLASK_DEBUG=True
"""
        env_file.write_text(env_content)
        print("✅ Created .env file with template values")
        print("⚠️  IMPORTANT: Update the .env file with your actual MongoDB Atlas credentials!")
    else:
        print("✅ .env file already exists")
        print("💡 Make sure MONGODB_URI is set to your MongoDB Atlas connection string")

def start_application():
    """Start the InsurEdge AI application"""
    print("\n🚀 Starting InsurEdge AI...")
    print("=" * 50)
    
    try:
        # Import and run the application
        from app_enhanced import app
        
        print("📊 ML Models: Loading...")
        print("🗄️  Database: Connecting...")
        print("🌐 Server: Starting on http://localhost:8000")
        print("=" * 50)
        
        # Run the application
        app.run(debug=True, host='0.0.0.0', port=8000)
        
    except KeyboardInterrupt:
        print("\n🛑 Application stopped by user")
    except Exception as e:
        print(f"❌ Error starting application: {e}")
        return False
    
    return True

def main():
    """Main startup function"""
    print("🎯 InsurEdge AI - Vehicle Insurance Platform")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    print("\n🔍 Checking dependencies...")
    if not check_dependencies():
        print("❌ Failed to install dependencies")
        sys.exit(1)
    
    print("\n🔍 Checking MongoDB...")
    check_mongodb()  # Warning only, don't exit
    
    print("\n📁 Setting up directories...")
    create_directories()
    
    print("\n⚙️  Setting up environment...")
    setup_environment()
    
    print("\n🎉 Setup complete! Starting application...")
    time.sleep(2)
    
    # Start the application
    start_application()

if __name__ == "__main__":
    main() 