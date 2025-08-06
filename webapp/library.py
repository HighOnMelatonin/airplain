def predict_value(zero_to_ten_proximity, ten_to_twenty_proximity, twenty_to_fifty_proximity, pop_density, public_transport):
    # features are ['0 to 10 km','>10 to 20 km','>20 to 50km', 'Population Density', 'Public Transport Travel']
    betas = [2663.96729991, -300.20830527, 250.48136665, 23.49259679, -252.31296831, 46.18437178]
    return betas[0] + betas[1] * zero_to_ten_proximity + betas[2] * ten_to_twenty_proximity + betas[3] * twenty_to_fifty_proximity + betas[4] * pop_density + betas[5] * public_transport

    