#! Discrete action space

Deep_Sea_Treasure_v0 = {
    'treasure':{"lower_bound": 0, "upper_bound": 2, "fine_grit": 0.1},
    'time_penalty':{"lower_bound": 0, "upper_bound": 2, "fine_grit": 0.1}
}

Four_Room_v0 = {
    'blue':{"lower_bound": 0, "upper_bound": 2, "fine_grit": 0.5},
    'green':{"lower_bound": 0, "upper_bound": 2, "fine_grit": 0.5},
    'red':{"lower_bound": 0, "upper_bound": 2, "fine_grit": 0.5}
}

Minecart_v0 = {
    'ore1':{"lower_bound": 0, "upper_bound": 2, "fine_grit": 0.2},
    'ore2':{"lower_bound": 0, "upper_bound": 2, "fine_grit": 0.5},
    'fuel':{"lower_bound": 0, "upper_bound": 2, "fine_grit": 0.2}
}

LunarLander_v2 = {
    'landed':{"lower_bound": 1, "upper_bound": 1, "fine_grit": 0.5},
    'shaped_reward':{"lower_bound": 0, "upper_bound": 2, "fine_grit": 0.5},
    'main_engine_fuel':{"lower_bound": 0, "upper_bound": 2, "fine_grit": 0.5},
    'side_engine_fuel':{"lower_bound": 0, "upper_bound": 2, "fine_grit": 0.5}
}
Supermario_v0 = {
    'x_pos':{"lower_bound": 1, "upper_bound": 1, "fine_grit": 0.5},
    'time':{"lower_bound": 1, "upper_bound": 1, "fine_grit": 0.5},
    'death':{"lower_bound": 1, "upper_bound": 1, "fine_grit": 0.5},
    'coin':{"lower_bound": 1, "upper_bound": 1, "fine_grit": 0.5},
    'enemy':{"lower_bound": 1, "upper_bound": 1, "fine_grit": 0.5},
}

#! Continuous action space 
Hopper_v4 = {
    'velocity':{"lower_bound": 1, "upper_bound": 1, "fine_grit": 0.5},
    'height':{"lower_bound": 1, "upper_bound": 1, "fine_grit": 0.5},
    'energy':{"lower_bound": 1, "upper_bound": 1, "fine_grit": 0.5},
}
Halfcheetah_v4 = {
    'velocity':{"lower_bound": 1, "upper_bound": 1, "fine_grit": 0.5},
    'energy':{"lower_bound": 1, "upper_bound": 1, "fine_grit": 0.5},
}

Factor_dictionary = {
    'deep-sea-treasure-v0':Deep_Sea_Treasure_v0,
    'four-room-v0':Four_Room_v0,
    'minecart-v0':Minecart_v0,
    'mo-lunar-lander-v2':LunarLander_v2,
    'mo-hopper-v4':Hopper_v4,
    'mo-halfcheetah-v4':Halfcheetah_v4,
}

