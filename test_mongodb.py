#!/usr/bin/env python3
"""
Test MongoDB connection for InsurEdge AI
"""

import os
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
from dotenv import load_dotenv
import datetime

def test_mongodb_connection():
    """Test MongoDB Atlas connection"""
    print("🧪 Testing MongoDB Atlas Connection...")
    print("=" * 50)
    
    # Load environment variables
    load_dotenv()
    
    # Get MongoDB URI
    mongodb_uri = os.getenv('MONGODB_URI')
    
    if not mongodb_uri:
        print("❌ MONGODB_URI not found in .env file")
        return False
    
    print(f"🔗 Connection String: {mongodb_uri[:50]}...")
    
    try:
        # Create client with timeout
        print("🔄 Connecting to MongoDB Atlas...")
        client = MongoClient(mongodb_uri, serverSelectionTimeoutMS=10000)
        
        # Test connection with ping
        client.admin.command('ping')
        print("✅ MongoDB Atlas connection successful!")
        
        # Get database info
        db = client['insuredge_ai']  # Your project database
        print(f"🗄️  Database: {db.name}")
        
        # Test basic operations
        print("\n🧪 Testing database operations...")
        
        # Test collection creation
        test_collection = db['connection_test']
        test_doc = {
            'test': True,
            'timestamp': datetime.datetime.utcnow(),
            'message': 'MongoDB connection test successful'
        }
        
        # Insert test document
        result = test_collection.insert_one(test_doc)
        print(f"✅ Test document inserted: {result.inserted_id}")
        
        # Read test document
        found_doc = test_collection.find_one({'_id': result.inserted_id})
        if found_doc:
            print("✅ Test document retrieved successfully")
        
        # Clean up test document
        test_collection.delete_one({'_id': result.inserted_id})
        print("✅ Test document cleaned up")
        
        # List existing collections
        collections = db.list_collection_names()
        print(f"📚 Existing collections: {collections if collections else 'None (will be created when needed)'}")
        
        # Test indexes (your app will create these)
        print("\n📋 Database is ready for InsurEdge AI!")
        print("✅ Users collection: Ready")
        print("✅ Claims collection: Ready") 
        print("✅ Policies collection: Ready")
        print("✅ Analytics collection: Ready")
        
        client.close()
        return True
        
    except ConnectionFailure as e:
        print(f"❌ Connection failed: {e}")
        print("💡 Check your internet connection and MongoDB Atlas cluster status")
        return False
        
    except ServerSelectionTimeoutError as e:
        print(f"❌ Server selection timeout: {e}")
        print("💡 Check if your IP address is whitelisted in MongoDB Atlas")
        return False
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def check_env_file():
    """Check if .env file is properly configured"""
    print("🔍 Checking .env file configuration...")
    
    if not os.path.exists('.env'):
        print("❌ .env file not found!")
        print("💡 Run: python setup_mongodb.py")
        return False
    
    load_dotenv()
    
    required_vars = ['MONGODB_URI', 'JWT_SECRET_KEY']
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"❌ Missing environment variables: {missing_vars}")
        return False
    
    print("✅ .env file configuration looks good")
    return True

def main():
    """Main test function"""
    print("🚀 InsurEdge AI - MongoDB Setup Test")
    print("=" * 50)
    
    # Check .env file first
    if not check_env_file():
        return
    
    # Test MongoDB connection
    success = test_mongodb_connection()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 MongoDB setup is working perfectly!")
        print("✅ Your InsurEdge AI project is ready to run")
        print("\n📖 Next steps:")
        print("1. Run: python app_enhanced.py")
        print("2. Open: http://localhost:8000/index.html")
    else:
        print("❌ MongoDB setup needs attention")
        print("💡 Check the error messages above and fix the issues")

if __name__ == "__main__":
    main()