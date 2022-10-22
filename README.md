# TM Research 2
#### Repository for theoretical mechanics' research 2

## Setup

1. Make sure you have installed:
    - **Python 3.9+** (make sure command `python3 -V` or `python -V`)
    - **PIP** (make sure command `pip`)
    - **Python Venv** for setup virtual environment (check this [useful link](https://docs.python.org/3/library/venv.html))

2. Clone the repo into your folder:
    ```shell
    git clone https://github.com/someilay/TM_Research_2.git
    cd ./TM_Research_2
    ```

3. Setup virtual environment:
    ```shell
    python3 -m venv venv
    ```

4. Activate environment ([guide](https://docs.python.org/3/tutorial/venv.html#creating-virtual-environments)) 
   and install Manim package ([installation](https://docs.manim.community/en/stable/installation.html)).

5. Install [matplotlib](https://matplotlib.org), version 3.5.0
    ```shell
    pip install matplotlib==3.5.0
    ```

## Rendering graphs

1.  Render graphs by executing:
    ```shell
    python3 main.py
    ```
    Rendered graphs will be stored in the `plots` directory

## Rendering animation

1.  Render scenes by executing:
    ```shell
    manim -pqh animation.py Main
    ```
    Render results for Task1 would be located in `media/videos/animation/1080p60/Main.mp4`

    Be aware, rendering can take a couple of minutes
