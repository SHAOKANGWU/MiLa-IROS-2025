# MiLa-IROS-2025

This repository contains the code for the paper:

**"One-Shot Robust Imitation Learning for Long-Horizon Visuomotor Tasks from Unsegmented Demonstrations"**  
Accepted at **IROS 2025**

---

## 🔧 Repository Structure

- **`MiLa-ISRR/Experiment-Given-Order/`**  
  Contains our main experiments and baseline implementations.

  - **`MiLa-ISRR/Experiment-Given-Order/MiLa/MiLa.py`**  
    Entry point to train the MiLa model. You can run this file to start training:
    ```bash
    bash MiLa-ISRR/Experiment-Given-Order/MiLa/run_script.bash
    ```

- **`old_version/`**  
  Contains archived or previously tried code that is no longer actively maintained but may still be useful for reference.

- **`util_folder/`**  
  Includes utility functions for extracting Movement Primitives (MPs) and implementing Gaussian Mixture Models (GMMs).

---

## 📄 Notes

- Please refer to the `MiLa-ISRR` directory for full experiments and configurations.
- Baseline comparisons are organized inside `Experiment-Given-Order`.
- The MiLa training pipeline is modular and can be adapted to custom datasets and environments.

---