## Installation

### Prerequisites

- Python 3.10
- An OpenAI API key ([sign up here](https://beta.openai.com/signup/))
- [Gurobi](https://support.gurobi.com/hc/en-us/articles/14799677517585-Getting-Started-with-Gurobi-Optimizer) license (free for academic use)

### Simulation (no hardware required)

1. **Clone the repository**:
```bash
git clone https://github.com/wxzhao03/VernaCopter.git
git clone https://github.com/wxzhao03/stlpy.git
cd VernaCopter
```

2. **Create a virtual environment**:
```bash
conda create -n vernacopter_env python=3.10 -y
conda activate vernacopter_env
```

3. **Install dependencies**:

First install stlpy from the cloned folder:
```bash
cd ../stlpy
pip install -e .
cd ../VernaCopter
```

Then install the remaining dependencies:
```bash
pip install -r requirements.txt
```

4. **Set environment variables**:

Run these once — they will be loaded automatically every time you activate the environment. Then re-activate to apply them in the current session.
```bash
conda env config vars set OPENAI_API_KEY="your_openai_api_key" PYTHONUTF8=1
conda activate vernacopter_env
```

### Real Drone Deployment (additional setup)

If you want to fly a real Crazyflie drone, complete the simulation setup first, then install the hardware dependencies:

```bash
pip install -r requirements-hardware.txt
```

> **Note:** If you are on Python 3.10 + Windows 64-bit, `motioncapture` may need to be recompiled from source. 


## Usage

Run the system with:

```bash
python -m examples.one_shot_automatic
```

### Configuration

Adjust the flags in [basics/config.py](basics/config.py) under `One_shot_parameters` to switch between modes:

**Input mode** — controls how the task is provided:

| `automated_user` | Behaviour |
|---|---|
| `False` | Voice input from microphone |
| `True` | Uses predefined task |

**Execution mode** — controls what happens after the trajectory is generated:

| `use_simulation` | Behaviour |
|---|---|
| `True` | PyBullet simulation |


## Authors and Acknowledgments

- **Author 1** - *Initial work* - [Teun van de Laar](https://github.com/TeunvdL)
- **Author 2** - *Follow-up work* - Jake Rap
- **Author 3** - *Follow-up work* - [Sofie Haeseart](https://github.com/shaesaert)

## Contact Information

For any questions, please contact [tavdlaar@gmail.com](mailto:tavdlaar@gmail.com).
