#!/usr/bin/env python3
"""
Fix MongoDB Atlas authentication for InsurEdge AI
"""

import os
from dotenv import load_dotenv

def fix_mongodb_atlas():
    """Guide user to fix MongoDB Atlas authentication"""
    print("🔧 MongoDB Atlas Authentication Fix")
    print("=" * 50)
    
    print("❌ Authentication failed - Let's fix this!")
    print("\n🔍 Common issues and solutions:")
    
    print("\n1. 🔑 Wrong Username/Password")
    print("   - Go to MongoDB Atlas Dashboard")
    print("   - Click 'Database Access' in left menu")
    print("   - Check your database user credentials")
    print("   - Reset password if needed")
    
    print("\n2. 🌐 IP Address Not Whitelisted")
    print("   - Go to 'Network Access' in MongoDB Atlas")
    print("   - Add your current IP address")
    print("   - Or add 0.0.0.0/0 for all IPs (less secure)")
    
    print("\n3. 🗄️ Wrong Database Name")
    print("   - Make sure database name in connection string is correct")
    
    print("\n4. 🔗 Connection String Format")
    print("   - Should look like:")
    print("   mongodb+srv://username:password@cluster.mongodb.net/database")
    
    print("\n" + "=" * 50)
    print("🛠️  Let's create a new .env file with correct credentials")
    
    response = input("\nDo you want to update your MongoDB connection? (y/N): ").lower()
    
    if response == 'y':
        update_env_file()
    else:
        print("\n💡 Manual fix steps:")
        print("1. Go to https://cloud.mongodb.com/")
        print("2. Login to your account")
        print("3. Select your cluster")
        print("4. Click 'Connect' -> 'Connect your application'")
        print("5. Copy the new connection string")
        print("6. Update your .env file with the new string")

def update_env_file():
    """Update .env file with new MongoDB credentials"""
    print("\n📝 Enter your MongoDB Atlas details:")
    
    # Get cluster info
    cluster_url = input("Cluster URL (e.g., cluster1.tblfwtr.mongodb.net): ").strip()
    username = input("Database Username: ").strip()
    password = input("Database Password: ").strip()
    database = input("Database Name (default: insuredge_ai): ").strip() or "insuredge_ai"
    
    if not all([cluster_url, username, password]):
        print("❌ All fields are required!")
        return
    
    # Build connection string
    mongodb_uri = f"mongodb+srv://{username}:{password}@{cluster_url}/{database}?retryWrites=true&w=majority"
    
    # Read current .env
    load_dotenv()
    jwt_secret = os.getenv('JWT_SECRET_KEY', 'your-secret-key-change-in-production')
    
    # Create new .env content
    env_content = f"""# InsurEdge AI Environment Configuration
# MongoDB Atlas Connection String
MONGODB_URI={mongodb_uri}

# JWT Secret Key
JWT_SECRET_KEY={jwt_secret}

# Flask Configuration
FLASK_ENV=development
FLASK_DEBUG=True

# File Upload Configuration
MAX_CONTENT_LENGTH=16777216
UPLOAD_FOLDER=uploads

# Server Configuration
HOST=0.0.0.0
PORT=8000
"""
    
    # Write new .env file
    try:
        with open('.env', 'w') as f:
            f.write(env_content)
        print("✅ .env file updated!")
        
        # Test new connection
        print("\n🧪 Testing new connection...")
        os.system("python test_mongodb.py")
        
    except Exception as e:
        print(f"❌ Error updating .env file: {e}")

def main():
    fix_mongodb_atlas()

if __name__ == "__main__":
    main()