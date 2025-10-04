"""Test script for financial guidance."""
import requests
import json

BASE_URL = "http://localhost:8000"

profiles = [
    {
        "name": "Age 30, Income 15k/month",
        "age": 30,
        "monthly_income": 15000,
        "monthly_expenses": 10000,
        "current_savings": 50000,
        "dependents": 0,
        "has_health_insurance": False,
        "has_life_insurance": False
    },
    {
        "name": "Age 25, Income 5k/month",
        "age": 25,
        "monthly_income": 5000,
        "monthly_expenses": 4000,
        "current_savings": 10000,
        "dependents": 0,
        "has_health_insurance": True,
        "has_life_insurance": False
    },
    {
        "name": "Age 40, Income 25k/month, 2 kids",
        "age": 40,
        "monthly_income": 25000,
        "monthly_expenses": 18000,
        "current_savings": 200000,
        "dependents": 2,
        "has_health_insurance": True,
        "has_life_insurance": True,
        "debt": 50000
    }
]

def test_profile(profile):
    """Test a single profile."""
    print(f"\n{'='*70}")
    print(f"Testing: {profile['name']}")
    print(f"{'='*70}")
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/financial-guidance",
            json=profile,
            timeout=120
        )
        
        if response.status_code == 200:
            result = response.json()
            
            print("\nCalculations:")
            for key, value in result['calculations'].items():
                if isinstance(value, (int, float)):
                    print(f"  {key}: ${value:,.2f}")
                else:
                    print(f"  {key}: {value}")
            
            print("\nPriority Actions:")
            for i, action in enumerate(result['priority_actions'], 1):
                print(f"  {i}. {action}")
            
            print("\nGuidance Preview:")
            print(result['guidance'][:400] + "...")
            
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
    
    except Exception as e:
        print(f"Exception: {e}")

def main():
    print("Financial Planning Assistant - Test Suite")
    
    # Test system health
    try:
        health = requests.get(f"{BASE_URL}/api/system/info")
        if health.status_code == 200:
            print(f"System Status: {health.json()['status']}")
        else:
            print("System not responding")
            return
    except:
        print("Cannot connect. Run: python app/main.py")
        return
    
    # Test each profile
    for profile in profiles:
        test_profile(profile)
        input("\nPress Enter for next profile...")

if __name__ == "__main__":
    main()