

# V2X-Real meta-class mapping.
# Raw object classes are grouped into vehicle / pedestrian / truck.
SUPER_CLASS_MAP = {
    "vehicle": ["LongVehicle", "Car", "PoliceCar"],
    "pedestrian": ["Child", "RoadWorker", "Pedestrian", "Scooter",
                   "ScooterRider", "Motorcycle", "MotorcyleRider",
                   "BicycleRider"],
    "truck": ["Truck", "Van", "TrashCan", "ConcreteTruck", "Bus"],
}
