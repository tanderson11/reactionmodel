# Pitch

This package makes it easy to specify any physical system that can be written in terms of reactions that convert reactants to products. What's nice about this solution is that
1. you don't have to think about how will you simulate the model while you're specifying it
2. but when you do go to simulate, the right choices have been made to make it easy without giving up performance

## The three key types

To use this package, you will interface with its three core types: `Species`, `Reaction`, and `Model`. They are described below:

- `Species`: a countable thing within the system. Examples include rabbits, methanol, and HIV free virions.
```python
s = Species(name, description="an optional longer description")
```
- `Reaction`: a process that converts one or more `Species` to one or more other different `Species` according to a rate law. Examples include the death of rabbits due to old age, the oxidation of methanol to formaldehyde, and the entry of HIV free virus into CD4+ target cells.
```python
r = Reaction(reactants, products, description="an optional informative description")
```
- `Model`: a collection of reactions.
```python
m = Model(species, reactions)
```

## Example 1
Here's an example of specifying a [Lotka-Volterra](https://en.wikipedia.org/wiki/Lotka–Volterra_equations) predator prey model:

```python
from reactionmodel.model import Species, Model, Reaction

## Model is specified
prey = Species('Prey')
predator = Species('Predator')

reactions = []
# when a single species appears more than once on one side of reaction
# specify a tuple of (Species, multiplicity)
reactions.append(Reaction([prey], [(prey, 2)], k="alpha", description="prey birth"))
reactions.append(Reaction([prey, predator], [predator], k="beta", description="prey death"))
reactions.append(Reaction([predator, prey], [(predator, 2)], k="delta", description="predator birth"))
# when no products are formed or no reactants are involved, provide the empty set []
reactions.append(Reaction([predator], [], k="gamma", description="predator death"))

m = Model([prey, predator], reactions)

## Now I want to simulate it
import scipy.integrate
import matplotlib.pyplot as plt

parameters = {
    'alpha': 1.1,
    'beta': 0.4,
    'delta': 0.1,
    'gamma': 0.4
}

dydt = m.get_dydt(parameters=parameters)
result = scipy.integrate.solve_ivp(dydt, [0.0, 100.0], y0=[10.0, 10.0])
plt.plot(result.t, result.y.T)
plt.legend(m.legend())
```
![Solution to our Lotka Volterra equations](examples/lotka.png)

## Example 2
As seen above, it's easy to build a model directly in a Python script or notebook. When collaborating, however, it's often better to specify models in standalone files that can be shared and reviewed by non-programmers while still being tracked by version control. `reactionmodel` supports writing and reading Models in `.yaml` and `.json` files.

```python
from reactionmodel.model import Species, Reaction, Model
A = Species('A')
r = Reaction([A], [], description='death', k='gamma')
m = Model([A], [r])

m.save('model.yaml')
```
This produces the file:
```yaml
reactions:
- description: death
  k: gamma
  products: []
  reactants:
  - A
species:
- name: A
```
In [`examples/`](examples/) you can explore several systems that show different features of the file specification, including the ability to programatically generate groups of related species and reactions.

# Installation

If you're already using [Poetry](https://python-poetry.org), run
```bash
poetry add git+https://github.com/tanderson11/reactionmodel.git
```
in your project directory.

Otherwise, read on:

You should add `reactionmodel` as a requirement for your Python project and use `reactionmodel` to specify the model of your system while you use the rest of your code to simulate / test / explore the model. The pain free way of doing this is to use [Poetry](https://python-poetry.org) to manage your dependencies. Poetry manages Python "virtual environments" (i.e. isolated containers with different versions of Python and different collections of packages that won't clash with any other container).

## Installing Poetry

Mac OS:
```bash
brew install pipx
pipx ensurepath
pipx install poetry
```
Linux:
```bash
sudo apt update
sudo apt install pipx
pipx ensurepath
pipx install poetry
```
Windows:
```bash
scoop install pipx
pipx ensurepath
pipx install poetry
```

(Note: when you run these commands, you are aiming to install Poetry on your system -- not through another virtual environment manager like Anaconda.)

## Using Poetry

Now that you have Poetry installed, you may use it to manage Python environments on a per-project basis. Navigate to your project and initialize a poetry environment. (When you run `poetry init` you will be asked a series of questions about your package. You can always go back and change your choices.)
```bash
cd project/
poetry init
```
Add your project's existing dependencies:
```bash
poetry add numpy
poetry add {whateverelse}
```
Finally install your project and all its dependencies in the environment
```bash
poetry install
```
Now to run your project code, prepend `poetry run` to any invocation of Python to let your command line know that you intend to run the version of Python managed by Poetry:
```bash
poetry run python my_project_code.py
```

## Adding `reactionmodel` as a dependency to your project

To any Python project managed by Poetry (see above), you can add `reactionmodel` as a dependency by running this command:
```bash
poetry add git+https://github.com/tanderson11/reactionmodel.git
```
Or by manually adding this line to your `pyproject.toml` file:
```toml
reactionmodel = { git = "https://github.com/tanderson11/reactionmodel.git" }
```
and subsequently running
```bash
poetry update
```

## Installing as a standalone package for development

If you want to develop this package further rather than using it in the context of a project, simply clone it and install dependencies with Poetry:
```
git clone https://github.com/tanderson11/reactionmodel.git
cd reactionmodel/
poetry install
```