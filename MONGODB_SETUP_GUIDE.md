# 🗄️ MongoDB Atlas Setup Guide for InsurEdge AI

## 📋 Prerequisites
- MongoDB Atlas account (free tier available)
- Your MongoDB Atlas connection string
- Python 3.8+ installed

## 🚀 Quick Setup (Recommended)

### Option 1: Automated Setup Script
1. **Run the setup script:**
   ```bash
   python setup_mongodb.py
   ```
   
2. **Or use the Windows batch file:**
   ```bash
   setup_mongodb.bat
   ```

3. **Follow the prompts:**
   - Enter your MongoDB Atlas connection string
   - Choose JWT secret key (or auto-generate)
   - The script will test your connection

### Option 2: Manual Setup
1. **Create `.env` file** in your project root
2. **Copy content from `mongodb_config.txt`**
3. **Update with your actual values**

## 🔗 Getting Your MongoDB Atlas Connection String

### Step 1: Access MongoDB Atlas
1. Go to [mongodb.com/atlas](https://www.mongodb.com/atlas)
2. Sign in to your account (or create one)

### Step 2: Get Connection String
1. **Click "Connect"** on your cluster
2. **Choose "Connect your application"**
3. **Copy the connection string**
4. **Replace `<password>` with your database user password**
5. **Replace `<dbname>` with `insuredge_db`

### Example Connection String:
```
mongodb+srv://username:password@cluster.mongodb.net/insuredge_db?retryWrites=true&w=majority
```

## 🔐 JWT Secret Key
- **Auto-generate**: Press Enter when prompted (recommended)
- **Custom**: Enter a long, random string (at least 32 characters)
- **Security**: Never share or commit this key to version control

## 🧪 Testing Your Setup

### Test MongoDB Connection:
```bash
python setup_mongodb.py
```
The script will automatically test your connection.

### Test Full Application:
```bash
python start.py
```

## 📁 File Structure After Setup
```
main project/
├── .env                    # Your configuration (created)
├── mongodb_config.txt     # Configuration template
├── setup_mongodb.py       # Setup script
├── setup_mongodb.bat      # Windows batch file
├── app_enhanced.py        # Main application
├── ml_models.py          # Machine learning models
├── database.py            # Database operations
└── ... (other files)
```

## ❌ Common Issues & Solutions

### Connection Failed
- **Check connection string format**
- **Verify username/password**
- **Ensure IP whitelist includes your IP**
- **Check cluster status**

### Authentication Failed
- **Verify database user exists**
- **Check user permissions**
- **Ensure correct password**

### Network Issues
- **Check firewall settings**
- **Verify internet connection**
- **Try different network**

## 🔒 Security Best Practices

1. **Environment Variables**: Never commit `.env` to version control
2. **Strong Passwords**: Use complex database passwords
3. **IP Whitelist**: Restrict access to your IP addresses
4. **Regular Updates**: Keep dependencies updated
5. **JWT Secret**: Use a long, random string

## 📞 Need Help?

1. **Check MongoDB Atlas documentation**
2. **Verify your connection string format**
3. **Test with MongoDB Compass (GUI tool)**
4. **Check the error messages in the setup script**

## 🎯 Next Steps

After successful MongoDB setup:
1. **Run the application**: `python start.py`
2. **Access your platform**: http://localhost:8000
3. **Register a new account**
4. **Test the AI features**

---

**🎉 Congratulations!** Your InsurEdge AI platform is now ready to connect to MongoDB Atlas!




