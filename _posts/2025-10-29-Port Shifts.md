---
title: "Port Shifts Solution with udev"
categories: tech
tags: [Deployment]
use_math: true
toc: true  # enables the sidebar TOC
toc_label: "On this page"  # optional, custom title for TOC
toc_sticky: true  # optional, makes the TOC stick while scrolling
---

# Using `udev` to Create Persistent Device Paths

## 1. Problem Statement: USB Port Shifting

In Linux, kernel-level device enumeration (e.g., for USB-to-Serial devices) is non-deterministic. A device like a Pixhawk may be assigned `/dev/ttyACM0` on one boot and `/dev/ttyACM1` on another, especially when other serial devices (like a RealSense camera) are present. This breaks static configurations in applications like ROS.

## 2. Solution: Persistent `udev` Symlinks

The `udev` subsystem manages the `/dev` directory. We can create a custom rule to identify a device by its unique, permanent hardware attributes (like `idVendor` and `idProduct`) and create a stable, persistent symbolic link (symlink) to it (e.g., `/dev/pixhawk`).

## 3. Prerequisites

* Root/`sudo` access on an Ubuntu system.
* The physical device (e.g., Pixhawk).
* A terminal.

---

## Step-by-Step Guide

### Step 1: Identify Device Hardware Attributes

1.  Connect the device to your computer.
2.  Identify its current kernel-assigned path.
    ```bash
    ls /dev/ttyACM*
    ```
    (This might also be `ttyUSB*` for other devices. Note the path, e.g., `/dev/ttyACM0`).

3.  Use `udevadm` to query the device's attributes. **Replace `/dev/ttyACM0` with your device's path.**
    ```bash
    udevadm info -a -n /dev/ttyACM0 | grep -E "idVendor|idProduct|serial"
    ```

4.  Analyze the output. Find the attribute block for the main TTY device. Record the `ATTRS{idVendor}` and `ATTRS{idProduct}` values.

    ```
    # Example Output
    ATTRS{idVendor}=="26ac"
    ATTRS{idProduct}=="0032"
    ATTRS{serial}=="00000000001A"
    ```
    **You must use the values from your command output.**

---

### Step 2: Create `udev` Rule File

1.  Create a new rules file in `/etc/udev/rules.d/`. The `99-` prefix ensures it runs after system defaults.
    ```bash
    sudo nano /etc/udev/rules.d/99-pixhawk.rules
    ```

---

### Step 3: Define the Rule

1.  Paste the following line into the file, **replacing `<VENDOR_ID>` and `<PRODUCT_ID>`** with the values you found in Step 1.

    ```
    KERNEL=="ttyACM*", ATTRS{idVendor}=="<VENDOR_ID>", ATTRS{idProduct}=="<PRODUCT_ID>", MODE:="0666", GROUP:="dialout", SYMLINK+="pixhawk"
    ```

    * **Example Rule:**
        ```
        KERNEL=="ttyACM*", ATTRS{idVendor}=="26ac", ATTRS{idProduct}=="0032", MODE:="0666", GROUP:="dialout", SYMLINK+="pixhawk"
        ```

2.  **Rule Breakdown:**
    * `KERNEL=="ttyACM*"`: Match any device the kernel names `ttyACM...`.
    * `ATTRS{...}`: Filter to match *only* the device with this specific Vendor and Product ID.
    * `MODE:="0666"`: Set file permissions to read/write for all users.
    * `GROUP:="dialout"`: Assign group ownership to `dialout`. ROS/Mavros users are typically in this group for serial port access.
    * `SYMLINK+="pixhawk"`: Create the persistent symlink at `/dev/pixhawk`.

3.  Save and exit the editor (in `nano`: `Ctrl+O`, `Enter`, `Ctrl+X`).

---

### Step 4: Apply and Verify the New Rule

1.  Reload the `udev` rules:
    ```bash
    sudo udevadm control --reload-rules
    ```
2.  Trigger the new rules to apply them to currently connected devices:
    ```bash
    sudo udevadm trigger
    ```
3.  **Test by hotplugging:** Unplug your device, wait a few seconds, and plug it back in. This is the most reliable way to verify the rule.
4.  Check if the symlink was created:
    ```bash
    ls -l /dev/pixhawk
    ```
    * **Success:** `lrwxrwxrwx 1 root root ... /dev/pixhawk -> ttyACM0` (or `-> ttyACM1`)
    * **Failure:** `ls: cannot access '/dev/pixhawk': No such file or directory` (See Troubleshooting).

---

### Step 5: (Optional) Configure User Permissions

If your user (running ROS) does not have permission, ensure it is in the `dialout` group.

1.  Add your user to the group:
    ```bash
    sudo usermod -a -G dialout $USER
    ```
2.  **IMPORTANT:** Group changes only apply after you **log out and log back in** or reboot.

---

### Step 6: Update Application Configuration (ROS Example)

Modify your ROS launch file (e.g., `mavros/px4.launch`) to use the persistent path.

* **Before:**
    ```xml
    <param name="fcu_url" type="string" value="serial:///dev/ttyACM0:57600" />
    ```
* **After:**
    ```xml
    <param name="fcu_url" type="string" value="serial:///dev/pixhawk:57600" />
    ```
This ensures your application reliably connects to the Pixhawk, regardless of its kernel-assigned `ttyACMx` path.

---

## Troubleshooting & Advanced Cases

* **Rule Not Working:**
    1.  Double-check `idVendor`/`idProduct` for typos in your `.rules` file.
    2.  Ensure you reloaded rules (`udevadm control --reload-rules`) and re-plugged the device.
    3.  Check permissions. Ensure your user is in the `dialout` group (and you've logged out/in).

* **Multiple Identical Devices:**
    If you have two identical Pixhawks, `idVendor` and `idProduct` are not unique. In this case, use the `ATTRS{serial}` value (from Step 1) to differentiate them.
    ```
    # Rule for device A
    KERNEL=="ttyACM*", ATTRS{idVendor}=="26ac", ATTRS{idProduct}=="0032", ATTRS{serial}=="00000000001A", MODE:="0666", GROUP:="dialout", SYMLINK+="pixhawk_A"

    # Rule for device B
    KERNEL=="ttyACM*", ATTRS{idVendor}=="26ac", ATTRS{idProduct}=="0032", ATTRS{serial}=="00000000001B", MODE:="0666", GROUP:="dialout", SY