from google.transit import gtfs_realtime_pb2

feed = gtfs_realtime_pb2.FeedMessage()

with open("VehiclePositions.pb", "rb") as f:
    feed.ParseFromString(f.read())

print("Feed version:", feed.header.gtfs_realtime_version)
print("Total entities:", len(feed.entity))

for entity in feed.entity[:5]:
    if not entity.HasField("vehicle"):
        continue

    bus = entity.vehicle
    pos = bus.position if bus.HasField("position") else None
    trip = bus.trip if bus.HasField("trip") else None

    print(f"\nEntity ID: {entity.id}")
    print("Vehicle ID:", bus.vehicle.id if bus.HasField("vehicle") else "N/A")
    print("Route ID:", trip.route_id if trip else "N/A")
    print("Trip ID:", trip.trip_id if trip else "N/A")

    if pos:
        print(f"Latitude:  {pos.latitude}")
        print(f"Longitude: {pos.longitude}")
        print(f"Speed:     {pos.speed:.2f} m/s" if pos.HasField("speed") else "Speed: N/A")
        print(f"Bearing:   {pos.bearing:.1f}°" if pos.HasField("bearing") else "Bearing: N/A")

