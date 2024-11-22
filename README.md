# AITIS-Data-Science-Intern-2024

Welcome to the 2024 Internship Data Access Repository! This repository contains all the necessary instructions and guidelines for accessing and utilizing the data required for your internship.

This repository serves as a guide for interns to access and use the data provided for the 2024 internship program. It includes instructions on prerequisites, installation, data access methods, and guidelines on data usage.

### Assumptions

The camera is tilted above ground, providing a bird's eye view over the environement. This is important for the initialisation of the vanishing points' locations

### Accessing Data

To access the data, follow these steps:

1. **Request Access**

    - Ensure you have been granted access by contacting your internship coordinator.

2. **Download Data**

    - You will receive a folder to download the data or credentials to access a data server.


3. **Load Data into Your Project**

    - Use the following template to load data in your Python scripts:

    ```python
    import pandas as pd

    data_path = 'data/dataset1.csv'
    df = pd.read_csv(data_path)
    print(df.head())
    ```
