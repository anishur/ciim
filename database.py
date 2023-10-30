import deta
from deta import Deta
#import streamlit as st

DETA_KEY = "a0q6lrwn5hf_4EQc8A6LF4HwKS5RHBr7tDwnAaVpdCAj"

deta = Deta(DETA_KEY)

db = deta.Base(name="biodegradable_urban_waste")

##db = deta.Base(name="biodegradable_urban_waste")
##db = deta.Base(name="monthly_report")
##db = deta.Base(name="monthly_reports")

# Functions insert data into the NoSQL database


def insert_period(period, regiaos, comment):
    """Returns the report on a successful creation, otherwise raises an error"""

    return db.put({"key": period, "regiaos": regiaos, "comment": comment})


def fetch_all_periods():
    """Returns a dict of all periods"""

    res = db.fetch()

    return res.items


def get_period(period):
    """If not found, function will return None"""

    return db.get(period)


##### ====== Database interface ===== ######

def get_all_periods():

    items = fetch_all_periods()

    periods = [item["key"] for item in items]

    return periods
