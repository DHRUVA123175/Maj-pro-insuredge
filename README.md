# InsurEdge AI - Vehicle Insurance Platform

🚗 **AI-Powered Vehicle Insurance Claim Processing Platform**

InsurEdge AI is a comprehensive vehicle insurance platform that leverages machine learning to automate damage assessment, streamline claim processing, and provide intelligent recommendations for vehicle insurance claims.

## 🌟 Features

### 🤖 AI-Powered Damage Detection
- **Computer Vision Analysis**: Uses ResNet50-based CNN for damage classification
- **Severity Assessment**: ML models predict damage severity levels
- **Cost Estimation**: AI-powered repair cost estimation
- **Multi-damage Support**: Collision, theft, vandalism, fire, flood, hail, and other damage types

### 📊 Smart Dashboard
- Real-time claim tracking
- AI accuracy metrics
- Policy management
- Analytics and insights
- Recent claims overview

### 🔐 Secure Authentication
- JWT-based authentication
- Password hashing
- User session management
- Profile management

### 📱 Modern UI/UX
- Responsive design
- Beautiful gradient animations
- Intuitive navigation
- Mobile-friendly interface

## 🛠️ Technology Stack

### Backend
- **Flask**: Python web framework
- **TensorFlow/Keras**: Deep learning models
- **OpenCV**: Image processing
- **Scikit-learn**: Machine learning algorithms
- **MongoDB**: NoSQL database
- **JWT**: Authentication

### Frontend
- **HTML5/CSS3**: Modern web standards
- **JavaScript**: Dynamic interactions
- **Responsive Design**: Mobile-first approach

## 📋 Prerequisites

Before running InsurEdge AI, ensure you have:

- Python 3.8 or higher
- MongoDB (local or cloud)
- Git
- pip (Python package manager)

## 🚀 Installation & Setup

### 1. Clone the Repository
```bash
git clone <repository-url>
cd insuredge-ai
```

### 2. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 3. MongoDB Setup

#### Option A: Local MongoDB
1. Install MongoDB on your system
2. Start MongoDB service:
```bash
# Windows
net start MongoDB

# macOS/Linux
sudo systemctl start mongod
```

#### Option B: MongoDB Atlas (Cloud)
1. Create a free account at [MongoDB Atlas](https://www.mongodb.com/atlas)
2. Create a new cluster
3. Get your connection string
4. Set environment variable:
```bash
export MONGODB_URI="mongodb+srv://username:password@cluster.mongodb.net/"
```

### 4. Environment Configuration
Create a `.env` file in the root directory:
```env
JWT_SECRET_KEY=your-super-secret-jwt-key-change-this
MONGODB_URI=mongodb://localhost:27017/
FLASK_ENV=development
```

### 5. Initialize ML Models
The ML models will be automatically created and trained on first run:
```bash
python app_enhanced.py
```

## 🏃‍♂️ Running the Application

### Start the Backend Server
```bash
python app_enhanced.py
```

The server will start on `http://localhost:8000`

### Access the Frontend
Open your browser and navigate to:
- **Homepage**: `http://localhost:8000/index.html`
- **Login**: `http://localhost:8000/login.html`
- **Register**: `http://localhost:8000/register.html`
- **Dashboard**: `http://localhost:8000/dashboard.html`
- **File Claim**: `http://localhost:8000/claim.html`

## 📚 API Documentation

### Authentication Endpoints

#### Register User
```http
POST /api/register
Content-Type: application/json

{
  "name": "John Doe",
  "email": "john@example.com",
  "password": "securepassword",
  "phone": "+1234567890",
  "address": "123 Main St"
}
```

#### Login
```http
POST /api/login
Content-Type: application/json

{
  "email": "john@example.com",
  "password": "securepassword"
}
```

### Dashboard Endpoints

#### Get Dashboard Data
```http
GET /api/dashboard
Authorization: Bearer <jwt_token>
```

#### Get User Profile
```http
GET /api/user/profile
Authorization: Bearer <jwt_token>
```

### Claims Endpoints

#### Submit Claim
```http
POST /api/claims
Authorization: Bearer <jwt_token>
Content-Type: multipart/form-data

Form Data:
- policy_id: "VP2025001"
- vehicle_info: "Honda Civic 2020"
- claim_type: "collision"
- description: "Front bumper damage"
- evidence: [image files]
```

#### Get User Claims
```http
GET /api/claims?limit=10
Authorization: Bearer <jwt_token>
```

### AI Analysis Endpoints

#### Analyze Damage
```http
POST /api/analyze-damage
Authorization: Bearer <jwt_token>
Content-Type: multipart/form-data

Form Data:
- image: [image file]
```

## 🧠 Machine Learning Models

### Damage Classification Model
- **Architecture**: ResNet50 + Custom layers
- **Input**: 224x224 RGB images
- **Output**: 7 damage type classifications
- **Training**: Transfer learning with ImageNet weights

### Severity Prediction Model
- **Algorithm**: Random Forest Regressor
- **Features**: Image statistics, edge density, color histograms
- **Output**: Severity score (0-1)

### Cost Estimation Model
- **Algorithm**: Random Forest Regressor
- **Features**: Damage type, severity, image features
- **Output**: Estimated repair cost

## 📁 Project Structure

```
insuredge-ai/
├── app.py                 # Original Flask app
├── app_enhanced.py        # Enhanced Flask app with ML integration
├── ml_models.py          # Machine learning models
├── database.py           # Database operations
├── requirements.txt      # Python dependencies
├── README.md            # This file
├── .env                 # Environment variables
├── uploads/             # Uploaded files directory
├── models/              # Trained ML models
├── index.html           # Homepage
├── login.html           # Login page
├── register.html        # Registration page
├── dashboard.html       # User dashboard
├── claim.html           # Claim submission page
└── styles.css           # Global styles
```

## 🔧 Configuration

### Database Configuration
- **Database Name**: `insuredge_ai`
- **Collections**: `users`, `claims`, `policies`, `analytics`
- **Indexes**: Automatically created for performance

### File Upload Configuration
- **Max File Size**: 16MB
- **Allowed Extensions**: PNG, JPG, JPEG, GIF, BMP
- **Upload Directory**: `uploads/`

### ML Model Configuration
- **Model Directory**: `models/`
- **Image Size**: 224x224 pixels
- **Batch Size**: 1 (for inference)

## 🧪 Testing

### Health Check
```bash
curl http://localhost:8000/api/health
```

### Test Registration
```bash
curl -X POST http://localhost:8000/api/register \
  -H "Content-Type: application/json" \
  -d '{"name":"Test User","email":"test@example.com","password":"testpass"}'
```

## 🚀 Deployment

### Production Setup
1. Set production environment variables
2. Use Gunicorn for WSGI server
3. Configure reverse proxy (Nginx)
4. Set up SSL certificates
5. Configure MongoDB for production

### Docker Deployment
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "app_enhanced:app"]
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support and questions:
- Create an issue in the repository
- Contact the development team
- Check the documentation

## 🔮 Future Enhancements

- [ ] Real-time claim status updates
- [ ] Mobile app development
- [ ] Advanced analytics dashboard
- [ ] Integration with external APIs
- [ ] Blockchain for claim verification
- [ ] Voice-based claim filing
- [ ] AR-based damage assessment

---

**Made with ❤️ for the future of vehicle insurance** 