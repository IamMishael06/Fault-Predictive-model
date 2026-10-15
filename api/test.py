import csv
import asyncio

print("hello")

def readings_simulator():
    with open('../cleaned_csv/synthetic_oil_stream_1s_30_predictions.csv', mode='r', encoding="utf-8", newline='') as file:
        reader = csv.DictReader(file)
           # Set up a CSV writer targeting our memory buffer
        dict_values = dict()
        writer = csv.DictWriter(dict_values, fieldnames=reader.fieldnames)
        writer.writeheader()
    
        for row in reader:
            # Let's say we only want to buffer employees from IT
                # WRITE the filtered data directly into the RAM buffer
                writer.writerow(row)
                print(dict_values)


readings_simulator()