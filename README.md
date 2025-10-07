# 🧭 Drone & Robot Mapping and Navigation (NavSLAM System)

---

## 📘 Overview

This project implements a **two-phase mapping and navigation system** that combines **SLAM**, **QR-based localization**, and **A\*** path planning.  

The project involves two main robots:
- 🛩️ **Mavic** — used in *Phase 1* for precise environment mapping and QR scanning  
- 🤖 **TurtleBot3** — used in *Phase 2* for navigation **without GPS or compass**, relying only on camera, LiDAR, and QR codes

---

## 🏗️ Phase 1 — Mapping & QR Localization (Mavic)

**Goal:** Build a LiDAR-based map of a multi-room house and record all door QR code positions.

### Key Features
- Controlled the **Mavic robot** to explore a house with **6–7 rooms**.
- Each **door** contains a **QR code** with a unique number identifier.
- The controller scans all QR codes and records their:
  - Numeric ID
  - Detected 2D position
  - Orientation (if available)
- All data is stored in a structured **`qr_positions.json`** file.
- A **rotating LiDAR** was used to avoid rotating the Mavic body at each waypoint.
- **GPS and Compass** were allowed to ensure:
  - Accurate map construction  
  - High-precision QR code localization

### Outcome
At the end of Phase 1:
- A **LiDAR map** of the full house is generated.
- A **JSON file** stores every QR code’s global position.
- This map and data become the reference for Phase 2.

---

## 🧭 Phase 2 — Autonomous Navigation (TurtleBot3)

**Goal:** Navigate between rooms **without GPS or Compass**, using only camera, LiDAR, and previously saved QR code data.

### Approach

1. **Initial Door Detection**
   - The TurtleBot rotates in place, scanning for a **specific color** that marks door frames.
   - Once the door color is found, the bot aligns itself by centering the color in the camera frame.

2. **QR Code Scan for Rough Localization**
   - As the TurtleBot moves toward the door, a QR code eventually enters view.
   - The code is scanned, and its ID is used to fetch the **door’s known position** from Phase 1.
   - This gives a **rough estimate of the bot’s initial position**.

3. **Pose Refinement**
   - When the QR code disappears from view, a **timer (≈4 seconds)** starts.
   - After the timer ends, the TurtleBot is assumed to be near the **door’s center**.
   - The bot’s position is then updated precisely to match the door’s recorded coordinates.
   - This yields a **3–4 cm localization accuracy** in most trials.

4. **Path Planning**
   - Using the refined position, the **A\*** algorithm plans a path from the current door to the target door.
   - The generated waypoints are followed sequentially until the goal door is reached.

---

## 🎬 Demonstrations

### 🚁 Phase 1 — Mavic Mapping
- Drone explores and maps the environment.
- Records QR code positions.
- [Mavic Mapping Video](https://github.com/JZX100II/Drone2Bot-Navigation-System/blob/main/Recordings%20and%20Figures/Mavic.mp4)

---

### 🤖 Phase 2 — TurtleBot Navigation (No GPS)
- Detects door, scans QR, and plans path.
- Navigates to target door.
- [TurtleBot3 Navigation Video](https://github.com/JZX100II/Drone2Bot-Navigation-System/blob/main/Recordings%20and%20Figures/Turtle.mp4)