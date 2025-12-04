'''
This is a configuration file for API requesting
'''
import requests
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo


class Config:
    # OAuth2 Configuration - Fill these in with your credentials
    CLIENT_ID = '53c390d4-c92d-4d65-8505-7168af28abc8'  # Application (client) ID
    CLIENT_SECRET = 'hzN8Q~B4IcyCWLJg1ObICDbg5Rm3HZO.C~vmgbHT'  # Client secret
    
    # Willow API Base URL (from Swagger UI)
    BASE_URL = 'https://northernarizonauniversity.app.willowinc.com/api/v3'
    
    # Token endpoint - Willow's own OAuth2 endpoint (from Swagger UI)
    TOKEN_ENDPOINT = 'https://northernarizonauniversity.app.willowinc.com/api/v3/oauth2/token'
    
    # SCOPE - Check Swagger UI "Authorize" dialog for available scopes
    # For client credentials flow, scope might be optional or a specific value
    # Try empty string, or check what Swagger shows when you click "Authorize"
    SCOPE = ''  # Will try empty first, update if Swagger shows a required scope
    
    # Class variables for storing tokens
    LIVE_DATA_TOKEN = ''
    HISTORICAL_DATA_TOKEN = ''
    ID = 'PNTM9M2jMkKm3GLHpLiq3J4gE'
    
    @classmethod
    def _get_bearer_token(cls, scope=None):
        """
        Internal class method to fetch a bearer token using OAuth2 client credentials flow.
        Uses Willow's own OAuth2 endpoint as shown in Swagger UI.
        
        Args:
            scope: OAuth2 scope to request. If None, uses cls.SCOPE
            
        Returns:
            str: Bearer token (access token) or None if failed
        """
        if scope is None:
            scope = cls.SCOPE
            
        if not cls.CLIENT_ID or not cls.CLIENT_SECRET:
            raise ValueError("Please set CLIENT_ID and CLIENT_SECRET in Config class")
        
        # Build payload - scope is optional for client credentials
        payload = {
            'client_id': cls.CLIENT_ID,
            'client_secret': cls.CLIENT_SECRET,
            'grant_type': 'client_credentials'
        }
        
        # Only add scope if it's provided and not empty
        if scope:
            payload['scope'] = scope
        
        headers = {
            'Content-Type': 'application/x-www-form-urlencoded',
            'Accept': 'application/json'
        }
        
        try:
            print(f"Attempting to get token from: {cls.TOKEN_ENDPOINT}")
            response = requests.post(cls.TOKEN_ENDPOINT, data=payload, headers=headers)
            
            # Print response for debugging
            print(f"Status Code: {response.status_code}")
            
            if response.status_code == 200:
                token_data = response.json()
                access_token = token_data.get('access_token')
                
                if access_token:
                    print(f"✓ Successfully obtained token!")
                    return access_token
                else:
                    print(f"Response JSON: {token_data}")
                    raise ValueError("No access_token in response")
            else:
                print(f"Error Response: {response.text}")
                response.raise_for_status()
                
        except requests.exceptions.RequestException as e:
            print(f"✗ Failed to get token: {e}")
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_json = e.response.json()
                    print(f"Error details: {error_json}")
                except:
                    print(f"Error response text: {e.response.text}")
            return None
        except Exception as e:
            print(f"✗ Unexpected error: {e}")
            return None

    # Access willow api to get live data token and store in LIVE_DATA_TOKEN
    @classmethod
    def get_live_data_token(cls):
        """Fetch and store live data bearer token."""
        token = cls._get_bearer_token()
        if token:
            cls.LIVE_DATA_TOKEN = token
            return token
        return None

    # Access willow api to get historical data token and store in HISTORICAL_DATA_TOKEN based on start and end date
    @classmethod
    def get_historical_data_token(cls, start_date=None, end_date=None):
        """
        Fetch and store historical data bearer token.
        
        Args:
            start_date: Optional start date (for logging/debugging)
            end_date: Optional end date (for logging/debugging)
            
        Returns:
            str: Bearer token or None if failed
        """
        # Note: start_date and end_date are included for API compatibility,
        # but OAuth2 tokens are typically time-scoped, not data-range-scoped
        token = cls._get_bearer_token()
        if token:
            cls.HISTORICAL_DATA_TOKEN = token
            return token
        return None
    
    @staticmethod
    def get_start_and_end_date_mst(hours_back=1):
        """
        Generate start and end dates in MST timezone for API requests.
        
        Args:
            hours_back: Number of hours to go back from now for start_date
            
        Returns:
            tuple: (start_date_str, end_date_str) in ISO 8601 format with Z suffix
        """
        mst = ZoneInfo('America/Denver')
        now_mst = datetime.now(mst)
        start_mst = now_mst - timedelta(hours=hours_back)
        
        # Format as ISO 8601 with Z (UTC indicator)
        # Convert to UTC for API (API typically expects UTC)
        now_utc = now_mst.astimezone(ZoneInfo('UTC'))
        start_utc = start_mst.astimezone(ZoneInfo('UTC'))
        
        start_str = start_utc.strftime('%Y-%m-%dT%H:%M:%SZ')
        end_str = now_utc.strftime('%Y-%m-%dT%H:%M:%SZ')
        
        return start_str, end_str

