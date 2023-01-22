import os
import requests

# see notes for keys, store them safely
# billing_account = 'TH Deggendorf - Cloudwürdig'
#
# get_billing = "/v1/{name={" + billing_account + "}/*}"
#
# get_billing_accounts = "/v1/billingAccounts"
# response = requests.get(f"{url}{get_billing_accounts}?oauth_token={oauth2}")
#
# print(response.json())


# #############################################################################
OAUTH2 = os.environ.get("GCP_OAUTH2")


def get_monthly_usage(*, billing_account,
                      start_date, end_date,
                      verbose=0):
    """
    Get the monthly usage of a billing account.
    """
    url = "https://cloudbilling.googleapis.com/v1/billingAccounts/"
    url += billing_account + "/usage/"
    url += "dates=" + start_date + ":" + end_date
    url += "&oauth_token=" + OAUTH2
    response = requests.get(url)
    if verbose > 0:
        print(response.json())
    return response.json()
