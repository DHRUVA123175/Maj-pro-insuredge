# ⏰ Auto-Approval System

## How It Works

Your InsurEdge AI platform now has an **intelligent auto-approval system** that automatically processes claims based on their risk level and time elapsed.

## Approval Timeline

### 🟢 LOW RISK Claims (Green)
- **Status:** "Processing"
- **Fraud Score:** < 30%
- **Auto-Approval Time:** **24 hours**
- **What happens:** Claim automatically approves after 24 hours if no admin intervention

### 🟡 MEDIUM RISK Claims (Orange)
- **Status:** "Under Investigation"
- **Fraud Score:** 30% - 70%
- **Auto-Approval Time:** **72 hours (3 days)**
- **What happens:** Claim gets extra scrutiny, then auto-approves after 3 days

### 🔴 HIGH RISK Claims (Red)
- **Status:** "Rejected - Fraud Detected" or "Requires Manual Review"
- **Fraud Score:** > 70%
- **Auto-Approval Time:** **NEVER**
- **What happens:** Requires mandatory admin review, will NEVER auto-approve

## Real-Time Status Updates

Claims show countdown timers:
- "Processing (Auto-approve in 18h)" - Low risk, 18 hours remaining
- "Under Investigation (Auto-approve in 48h)" - Medium risk, 48 hours remaining

## Admin Override

Admins can **manually approve or reject** any claim at any time, overriding the auto-approval timer.

## Configuration

Current settings (in `demo_app.py`):
```python
AUTO_APPROVAL_CONFIG = {
    'enabled': True,
    'low_risk_delay_hours': 24,      # 24 hours for low risk
    'medium_risk_delay_hours': 72,   # 72 hours for medium risk
    'high_risk_never': True          # High risk never auto-approves
}
```

## Benefits

✅ **Faster Processing:** Low-risk claims don't wait for manual review
✅ **Reduced Admin Workload:** Only suspicious claims need attention
✅ **Transparent Timeline:** Users know exactly when to expect approval
✅ **Safety First:** High-risk claims always require human review
✅ **Fraud Prevention:** Suspicious claims get extended review period

## Example Scenarios

### Scenario 1: Clean Claim
- User submits claim with clear damage photo
- AI detects: Low fraud risk (10%)
- Status: "Processing (Auto-approve in 24h)"
- After 24 hours: Automatically changes to "Auto-Approved (Low Risk)"

### Scenario 2: Suspicious Claim
- User submits claim with questionable details
- AI detects: Medium fraud risk (50%)
- Status: "Under Investigation (Auto-approve in 72h)"
- After 72 hours: Changes to "Auto-Approved (Medium Risk - Reviewed)"
- OR admin can reject earlier if fraud confirmed

### Scenario 3: Fraudulent Claim
- User submits claim with fake/future date
- AI detects: High fraud risk (85%)
- Status: "Rejected - Fraud Detected"
- Result: NEVER auto-approves, requires admin review

## Testing

To test the system:
1. Submit a claim with valid data → Should show 24h timer
2. Submit a claim with suspicious data → Should show 72h timer
3. Submit a claim with future date → Should be rejected immediately

## Notes

- The system checks auto-approvals every time a user views their dashboard
- The system checks auto-approvals every time admin views statistics
- Timers are calculated in real-time based on claim creation timestamp
- Once auto-approved, claims are marked with "Auto-Approved" in the status
