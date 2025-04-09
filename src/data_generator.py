import pandas as pd
import numpy as np
from faker import Faker
from random import choice, randint, uniform
from datetime import datetime, timedelta

fake = Faker()
np.random.seed(42)

NUM_CUSTOMERS = 300
NUM_COURIERS = 50
NUM_ORDERS = 2000
REGIONS = ['Downtown', 'Uptown', 'Eastside', 'Westside', 'Suburbs']
INCIDENT_TYPES = ['Late delivery', 'Courier not found', 'Food missing', 'Order canceled']

# 1. Customers
customers = [{
    'customer_id': i,
    'name': fake.name(),
    'region': choice(REGIONS),
    'signup_date': fake.date_between(start_date='-2y', end_date='today')
} for i in range(1, NUM_CUSTOMERS + 1)]
df_customers = pd.DataFrame(customers)
df_customers.to_csv('data/customers.csv', index=False)

# 2. Couriers
couriers = [{
    'courier_id': i,
    'name': fake.name(),
    'vehicle_type': choice(['Bike', 'Motorcycle', 'Car']),
    'region': choice(REGIONS),
    'active_since': fake.date_between(start_date='-3y', end_date='-6m')
} for i in range(1, NUM_COURIERS + 1)]
df_couriers = pd.DataFrame(couriers)
df_couriers.to_csv('data/couriers.csv', index=False)

# 3. Orders
orders = []
for i in range(1, NUM_ORDERS + 1):
    order_time = fake.date_time_between(start_date='-30d', end_date='now')
    delivery_time = order_time + timedelta(minutes=randint(15, 90))
    orders.append({
        'order_id': i,
        'customer_id': randint(1, NUM_CUSTOMERS),
        'courier_id': randint(1, NUM_COURIERS),
        'region': choice(REGIONS),
        'order_time': order_time,
        'delivery_time': delivery_time,
        'total_amount': round(uniform(10, 120), 2),
        'customer_rating': choice([1, 2, 3, 4, 5, None])
    })
df_orders = pd.DataFrame(orders)
df_orders.to_csv('data/orders.csv', index=False)

# 4. Incidents (simulando 10% dos pedidos com incidentes)
incidents = []
incident_orders = np.random.choice(df_orders['order_id'], size=int(NUM_ORDERS * 0.1), replace=False)
for order_id in incident_orders:
    incidents.append({
        'incident_id': fake.uuid4(),
        'order_id': order_id,
        'incident_type': choice(INCIDENT_TYPES),
        'description': fake.sentence(),
        'reported_at': fake.date_time_between(start_date='-30d', end_date='now')
    })
df_incidents = pd.DataFrame(incidents)
df_incidents.to_csv('data/incidents.csv', index=False)

print("✅ Data simulation complete! CSV files saved in /data")
