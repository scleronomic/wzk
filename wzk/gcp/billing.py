import requests


key = 'AIzaSyBx5TzAuFb1KwjFu88m1nGNWqqEZih5wrE'
url = 'https://cloudbilling.googleapis.com'

# https://cloudbilling.googleapis.com/$discovery/rest?version=v1
# oath =

oauth2 = '377873609285-49jufngjj4a3o4g3kirmsb34oo48i200.apps.googleusercontent.com'
secret = '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'


billing_account = 'TH Deggendorf - Cloudwürdig'

get_billing = "/v1/{name={" + billing_account + "}/*}"

get_billing_accounts = "/v1/billingAccounts"
response = requests.get(f"{url}{get_billing_accounts}?oauth_token={oauth2}")

print(response.json())