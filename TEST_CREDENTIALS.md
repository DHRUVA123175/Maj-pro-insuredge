# 🔐 Test Credentials (For Development Only)

## ⚠️ IMPORTANT
These credentials are for testing purposes only. They are NOT displayed on the frontend anymore.

## User Accounts

### Regular User (For Testing Claims)
- **Email:** demo@insuredge.ai
- **Password:** demo123
- **Role:** user
- **Access:** Can submit claims, view own dashboard
- **2FA:** Yes (use code: 123456 for demo)

### Admin User (For Admin Panel)
- **Email:** admin@insuredge.ai
- **Password:** admin123
- **Role:** admin
- **Access:** Full admin panel, view all users, manage all claims
- **2FA:** No

## How to Test

### Test Regular User Flow:
1. Go to http://localhost:5000/login
2. Login with demo@insuredge.ai / demo123
3. Enter 2FA code: 123456
4. You'll be redirected to user dashboard
5. Submit a claim to test fraud detection

### Test Admin Flow:
1. Go to http://localhost:5000/login
2. Login with admin@insuredge.ai / admin123
3. You'll be redirected to admin dashboard
4. Click on users to expand and see their claims
5. Approve/reject claims

### Test Registration:
1. Go to http://localhost:5000/register
2. Create a new account
3. Login with your new credentials
4. Submit claims

## Fraud Detection Test Cases

### Should be HIGH RISK (Rejected):
```
Description: "My car fell into the river and is completely submerged"
Expected: Fraud Score 0.5+, Status: Rejected
```

### Should be MEDIUM RISK (Review):
```
Description: "Not sure what happened, maybe someone hit it"
Expected: Fraud Score 0.25-0.49, Status: Under Investigation
```

### Should be LOW RISK (Approved):
```
Description: "Minor collision with another vehicle at intersection"
Expected: Fraud Score < 0.25, Status: Processing → Auto-approve in 24h
```

## What Was Fixed

✅ **Removed demo credentials from login page** - No longer visible to users
✅ **Fixed user registration** - New users now get 'role': 'user' properly
✅ **Login works for all users** - Both admin and regular users can login
✅ **Enhanced fraud detection** - Now analyzes text descriptions
✅ **Auto-approval system** - Claims auto-approve based on risk level

## Security Notes

- Passwords are hashed using werkzeug.security
- JWT tokens expire after 24 hours
- Session-based authentication for demo
- Admin routes check role before allowing access
- Users can only see their own claims
- Admins can see all users and claims

## Production Recommendations

Before deploying to production:
1. Change all default passwords
2. Use environment variables for secrets
3. Implement real database (MongoDB/PostgreSQL)
4. Add rate limiting
5. Enable HTTPS
6. Implement real 2FA (Google Authenticator, SMS)
7. Add email verification
8. Implement password reset functionality
9. Add audit logging
10. Use proper JWT refresh tokens
