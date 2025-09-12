#!/usr/bin/env python3
"""
MongoDB Atlas Setup Script for InsurEdge AI
Helps configure the MongoDB Atlas connection
"""

import os
from pathlib import Path

def create_env_file():
    """Create .env file with user input"""
    print("🔧 InsurEdge AI - MongoDB Atlas Setup")
    print("=" * 50)
    
    # Check if .env already exists
    env_file = Path('.env')
    if env_file.exists():
        print("⚠️  .env file already exists!")
        response = input("Do you want to overwrite it? (y/N): ").lower()
        if response != 'y':
            print("Setup cancelled.")
            return
    
    print("\n📝 Please provide your MongoDB Atlas configuration:")
    print("💡 You can find this in your MongoDB Atlas dashboard")
    
    # Get MongoDB Atlas connection string
    print("\n🔗 MongoDB Atlas Connection String:")
    print("Format: mongodb+srv://username:password@cluster.mongodb.net/database")
    mongodb_uri = input("Enter your connection string: ").strip()
    
    if not mongodb_uri:
        print("❌ Connection string is required!")
        return
    
    # Get JWT secret key
    print("\n🔐 JWT Secret Key:")
    print("This should be a long, random string for security")
    jwt_secret = input("Enter JWT secret key (or press Enter for auto-generate): ").strip()
    
    if not jwt_secret:
        import secrets
        jwt_secret = secrets.token_urlsafe(32)
        print(f"🔑 Auto-generated JWT secret: {jwt_secret}")
    
    # Create .env content
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
    
    # Write .env file
    try:
        env_file.write_text(env_content)
        print(f"\n✅ .env file created successfully!")
        print(f"📁 Location: {env_file.absolute()}")
        
        # Test MongoDB connection
        print("\n🧪 Testing MongoDB connection...")
        test_mongodb_connection(mongodb_uri)
        
    except Exception as e:
        print(f"❌ Error creating .env file: {e}")

def test_mongodb_connection(connection_string):
    """Test the MongoDB connection"""
    try:
        from pymongo import MongoClient
        
        print("🔗 Connecting to MongoDB Atlas...")
        client = MongoClient(connection_string, serverSelectionTimeoutMS=10000)
        
        # Test connection
        client.admin.command('ping')
        print("✅ MongoDB Atlas connection successful!")
        
        # Get database info
        db = client.get_database()
        print(f"🗄️  Database: {db.name}")
        
        # List collections (will be empty initially)
        collections = db.list_collection_names()
        print(f"📚 Collections: {collections if collections else 'None (will be created automatically)'}")
        
        client.close()
        
    except Exception as e:
        print(f"❌ MongoDB connection failed: {e}")
        print("💡 Please check your connection string and try again")

def main():
    """Main setup function"""
    try:
        create_env_file()
    except KeyboardInterrupt:
        print("\n\n🛑 Setup cancelled by user")
    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
    
    print("\n📖 Next steps:")
    print("1. Make sure your .env file is configured correctly")
    print("2. Run: python start.py")
    print("3. Your InsurEdge AI platform will be available at http://localhost:8000")

if __name__ == "__main__":
    main()




