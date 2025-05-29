# ESP32_BWR_sim
A simple Boiling Water Reactor simulator using an ESP32 MCU

This project is designed more as an engaging education tool than a serious simulator.
Much of its physics is therefore oversimplified.
One has to not that, obviously, the systems are also very simplified and basically made up.
It utilizes full 6-group point kinetics equations for a U-235-based reactor with a maximum power of 1000 kW.
For systems there are two heat transfer loops, one pressuriser, one reserve water tank for loop A, a common coolant relief tank for both loops and one turbine for power generation.
