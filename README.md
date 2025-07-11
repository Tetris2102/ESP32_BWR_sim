# ESP32_BWR_sim
A simple Boiling Water Reactor simulator using an ESP32 MCU

This project is designed more as an engaging education tool than a serious simulator.
Much of its physics is therefore oversimplified.
One has to note that, obviously, the systems are also very simplified and basically made up.
It utilizes full 6-group point kinetics equations for a U-235-based reactor with a nominal power of 1000 kW.
For systems there are two heat transfer loops, one pressuriser, one reserve water tank for loop A, one common coolant relief tank for both loops and one turbine for power generation.

For now, it's impossible to cause meltdown, but this can be potentially changed by increasing ReactorCore.rho_max to 0.05, for example. =)

![Image](https://github.com/user-attachments/assets/3b65c726-c692-486c-86b5-cb8e4bfa4ffe)

<img width="1156" alt="Image" src="https://github.com/user-attachments/assets/d4473f26-9b66-4aaf-af8a-b7f6404f9bea" />

![Image](https://github.com/user-attachments/assets/8e7e51bb-6524-4dc5-8cfb-d498f9d982a3)
