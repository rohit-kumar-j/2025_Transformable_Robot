

# Robot Setup & Configuration Guide

> **Note:** This procedure is experimental and may not be complete. Please verify each step before implementation.

## Table of Contents

- [Software Preparation](#software-preparation)
- [ESP32 MicroPython Setup](#esp32-micropython-setup)
- [Development Tools](#development-tools)
- [Hardware Configuration](#hardware-configuration)
  - [Client/Server / MQTT Broker](#clientserver--mqtt-broker)
  - [Wiring Diagram](#wiring-diagram)
- [Flashing the ESP32](#flashing-the-esp32)
- [Optional OTA Setup](#optional-ota-setup)

---

## Software Preparation

Ensure you have the following software installed:

- **Docker(optional**
- **MQTT Broker**  
  Example: [Mosquitto MQTT](https://mosquitto.org)
- **Jupyter Notebook**

---

## ESP32 MicroPython Setup

For flashing the ESP32 with MicroPython, use the following firmware:

- **Firmware File:** `LOLIN_C3_MINI-20240602-v1.23.0.bin`  
  _(Download from the [MicroPython download page](https://micropython.org/download/))_

---

## Development Tools

- **Thonny IDE**  
  Used for ESP32 development and accessing the file system.

---

## Hardware Configuration

- This setup supports a **PWM Servo Version** configuration.

### Wiring Diagram

- **Battery 1:**  
  Connect to the DC-DC converter, then to the ESP32.

- **Battery 2 (30C):**  
  Connect to the PCA9685, which drives 5 servos.

Remember to double check the voltages. Matching them to the specifications of the boards.

---

## Flashing the ESP32

1. **Flash Firmware:**  
   Use the specified MicroPython firmware to flash the ESP32. Use their tutorial to troubleshoot.

2. **File System Access:**  
   Use Thonny IDE to access and manage the ESP32 file system.

---

## Optional OTA Setup

- **OTA (Over-The-Air) Updates:**  
  Optionally, set up OTA using `webrepl` for remote access and updates.
